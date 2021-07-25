// -----------------------------------------------------------------------------------------
// NVEnc by rigaya
// -----------------------------------------------------------------------------------------
//
// The MIT License
//
// Copyright (c) 2020 rigaya
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
#include "convert_csp.h"
#include "NVEncFilterSmooth.h"
#include "NVEncParam.h"
#pragma warning (push)
#pragma warning (disable: 4819)
#include "cuda_runtime.h"
#if __CUDACC_VER_MAJOR__ >= 10
#include "cuda_fp16.h"
#include "cuda_fp16.hpp"
#endif
#include "device_launch_parameters.h"
#pragma warning (pop)
#include "rgy_cuda_util_kernel.h"

#define ENABLE_CUDA_FP16_DEVICE (__CUDACC_VER_MAJOR__ >= 10 && __CUDA_ARCH__ >= 530)
#define ENABLE_CUDA_FP16_HOST   (__CUDACC_VER_MAJOR__ >= 10)

#define SPP_THREAD_BLOCK_X (8) //blockDim.x
#define SPP_THREAD_BLOCK_Y (8) //blockDim.y

#define SPP_BLOCK_SIZE_X (8) //ひとつのスレッドブロックの担当するx方向の8x8ブロックの数

#define SPP_SHARED_BLOCK_NUM_X (SPP_BLOCK_SIZE_X+2) //sharedメモリ上のx方向の8x8ブロックの数
#define SPP_SHARED_BLOCK_NUM_Y (2)                  //sharedメモリ上のy方向の8x8ブロックの数

#define SPP_LOOP_COUNT_BLOCK (8)

//CUDA Sampleより拝借
#define C_a 1.387039845322148f //!< a = (2^0.5) * cos(    pi / 16);  Used in forward and inverse DCT.
#define C_b 1.306562964876377f //!< b = (2^0.5) * cos(    pi /  8);  Used in forward and inverse DCT.
#define C_c 1.175875602419359f //!< c = (2^0.5) * cos(3 * pi / 16);  Used in forward and inverse DCT.
#define C_d 0.785694958387102f //!< d = (2^0.5) * cos(5 * pi / 16);  Used in forward and inverse DCT.
#define C_e 0.541196100146197f //!< e = (2^0.5) * cos(3 * pi /  8);  Used in forward and inverse DCT.
#define C_f 0.275899379282943f //!< f = (2^0.5) * cos(7 * pi / 16);  Used in forward and inverse DCT.

//Normalization constant that is used in forward and inverse DCT
#define C_norm 0.3535533905932737f // 1 / (8^0.5)


template<typename T> __device__ __inline__ T setval(float val);
template<> __device__ __inline__ float setval(float val) { return val; };
#if ENABLE_CUDA_FP16_HOST
template<> __device__ __inline__ __half2 setval(float val) { return __float2half2_rn(val); }
__device__ __inline__
__half2 fabs(__half2 val) {
    __half2 h;
    RGY_HALF2_TO_UI(h) = RGY_HALF2_TO_UI(val) & 0x7fff7fffu;
    return h;
}
#endif //#if ENABLE_CUDA_FP16_HOST

template<typename T, int Step>
__device__ void CUDAsubroutineInplaceDCTvector(T *Vect0) {
    T *Vect1 = Vect0 + Step;
    T *Vect2 = Vect1 + Step;
    T *Vect3 = Vect2 + Step;
    T *Vect4 = Vect3 + Step;
    T *Vect5 = Vect4 + Step;
    T *Vect6 = Vect5 + Step;
    T *Vect7 = Vect6 + Step;

    T X07P = (*Vect0) + (*Vect7);
    T X16P = (*Vect1) + (*Vect6);
    T X25P = (*Vect2) + (*Vect5);
    T X34P = (*Vect3) + (*Vect4);

    T X07M = (*Vect0) - (*Vect7);
    T X61M = (*Vect6) - (*Vect1);
    T X25M = (*Vect2) - (*Vect5);
    T X43M = (*Vect4) - (*Vect3);

    T X07P34PP = X07P + X34P;
    T X07P34PM = X07P - X34P;
    T X16P25PP = X16P + X25P;
    T X16P25PM = X16P - X25P;

    (*Vect0) = setval<T>(C_norm) * (X07P34PP + X16P25PP);
    (*Vect2) = setval<T>(C_norm) * (setval<T>(C_b) * X07P34PM + setval<T>(C_e) * X16P25PM);
    (*Vect4) = setval<T>(C_norm) * (X07P34PP - X16P25PP);
    (*Vect6) = setval<T>(C_norm) * (setval<T>(C_e) * X07P34PM - setval<T>(C_b) * X16P25PM);

    (*Vect1) = setval<T>(C_norm) * (setval<T>(C_a) * X07M - setval<T>(C_c) * X61M + setval<T>(C_d) * X25M - setval<T>(C_f) * X43M);
    (*Vect3) = setval<T>(C_norm) * (setval<T>(C_c) * X07M + setval<T>(C_f) * X61M - setval<T>(C_a) * X25M + setval<T>(C_d) * X43M);
    (*Vect5) = setval<T>(C_norm) * (setval<T>(C_d) * X07M + setval<T>(C_a) * X61M + setval<T>(C_f) * X25M - setval<T>(C_c) * X43M);
    (*Vect7) = setval<T>(C_norm) * (setval<T>(C_f) * X07M + setval<T>(C_d) * X61M + setval<T>(C_c) * X25M + setval<T>(C_a) * X43M);
}

template<typename T, int Step>
__device__ void CUDAsubroutineInplaceIDCTvector(T *Vect0) {
    T *Vect1 = Vect0 + Step;
    T *Vect2 = Vect1 + Step;
    T *Vect3 = Vect2 + Step;
    T *Vect4 = Vect3 + Step;
    T *Vect5 = Vect4 + Step;
    T *Vect6 = Vect5 + Step;
    T *Vect7 = Vect6 + Step;

    T Y04P = (*Vect0) + (*Vect4);
    T Y2b6eP = setval<T>(C_b) * (*Vect2) + setval<T>(C_e) * (*Vect6);

    T Y04P2b6ePP = Y04P + Y2b6eP;
    T Y04P2b6ePM = Y04P - Y2b6eP;
    T Y7f1aP3c5dPP = setval<T>(C_f) * (*Vect7) + setval<T>(C_a) * (*Vect1) + setval<T>(C_c) * (*Vect3) + setval<T>(C_d) * (*Vect5);
    T Y7a1fM3d5cMP = setval<T>(C_a) * (*Vect7) - setval<T>(C_f) * (*Vect1) + setval<T>(C_d) * (*Vect3) - setval<T>(C_c) * (*Vect5);

    T Y04M = (*Vect0) - (*Vect4);
    T Y2e6bM = setval<T>(C_e) * (*Vect2) - setval<T>(C_b) * (*Vect6);

    T Y04M2e6bMP = Y04M + Y2e6bM;
    T Y04M2e6bMM = Y04M - Y2e6bM;
    T Y1c7dM3f5aPM = setval<T>(C_c) * (*Vect1) - setval<T>(C_d) * (*Vect7) - setval<T>(C_f) * (*Vect3) - setval<T>(C_a) * (*Vect5);
    T Y1d7cP3a5fMM = setval<T>(C_d) * (*Vect1) + setval<T>(C_c) * (*Vect7) - setval<T>(C_a) * (*Vect3) + setval<T>(C_f) * (*Vect5);

    (*Vect0) = setval<T>(C_norm) * (Y04P2b6ePP + Y7f1aP3c5dPP);
    (*Vect7) = setval<T>(C_norm) * (Y04P2b6ePP - Y7f1aP3c5dPP);
    (*Vect4) = setval<T>(C_norm) * (Y04P2b6ePM + Y7a1fM3d5cMP);
    (*Vect3) = setval<T>(C_norm) * (Y04P2b6ePM - Y7a1fM3d5cMP);

    (*Vect1) = setval<T>(C_norm) * (Y04M2e6bMP + Y1c7dM3f5aPM);
    (*Vect5) = setval<T>(C_norm) * (Y04M2e6bMM - Y1d7cP3a5fMM);
    (*Vect2) = setval<T>(C_norm) * (Y04M2e6bMM + Y1d7cP3a5fMM);
    (*Vect6) = setval<T>(C_norm) * (Y04M2e6bMP - Y1c7dM3f5aPM);
}

#if ENABLE_CUDA_FP16_HOST
#if !ENABLE_CUDA_FP16_DEVICE
template<> __device__ void CUDAsubroutineInplaceDCTvector<__half2, 1>(__half2 *Vect0) { };
template<> __device__ void CUDAsubroutineInplaceDCTvector<__half2, 9>(__half2 *Vect0) { };
template<> __device__ void CUDAsubroutineInplaceIDCTvector<__half2, 1>(__half2 *Vect0) { };
template<> __device__ void CUDAsubroutineInplaceIDCTvector<__half2, 9>(__half2 *Vect0) { };
#endif
#endif

template<typename T>
__device__ void dct8x8(T shared_tmp[8][9], int thWorker) {
    CUDAsubroutineInplaceDCTvector<T, 1>((T *)&shared_tmp[thWorker][0]); // row
    CUDAsubroutineInplaceDCTvector<T, 9>((T *)&shared_tmp[0][thWorker]); // column
}

template<typename T>
__device__ void idct8x8(T shared_tmp[8][9], int thWorker) {
    CUDAsubroutineInplaceIDCTvector<T, 9>((T *)&shared_tmp[0][thWorker]);  // column
    CUDAsubroutineInplaceIDCTvector<T, 1>((T *)&shared_tmp[thWorker][0]); // row
}
__device__ float calcThreshold(const float qp, const float threshA, const float threshB) {
    return clamp(threshA * qp + threshB, 0.0f, qp);
}

__device__ void threshold8x8(float shared_tmp[8][9], int thWorker, const float threshold) {
    #pragma unroll
    for (int y = 0; y < 8; y++) {
        if (y > 0 || thWorker > 0) {
            float *ptr = &shared_tmp[y][thWorker];
            const float val = ptr[0];
            if (fabs(val) <= threshold) {
                ptr[0] = 0.0f;
            }
        }
    }
}

#if ENABLE_CUDA_FP16_HOST
__device__ void threshold8x8(__half2 shared_tmp[8][9], int thWorker, const __half2 threshold) {
#if ENABLE_CUDA_FP16_DEVICE
    #pragma unroll
    for (int y = 0; y < 8; y++) {
        if (y > 0 || thWorker > 0) {
            __half2 *ptr = &shared_tmp[y][thWorker];
            const __half2 val = ptr[0];
            ptr[0] = __hgt2(fabs(val), threshold) * val;
        }
    }
#endif //#if ENABLE_CUDA_FP16_DEVICE
}
#endif //#if ENABLE_CUDA_FP16_HOST

__constant__ uchar2 SPP_DEBLOCK_OFFSET[127] = {
  { 0,0 },                                                         // quality = 0

  { 0,0 },{ 4,4 },                                                 // quality = 1

  { 0,0 },{ 2,2 },{ 6,4 },{ 4,6 },                                 // quality = 2

  { 0,0 },{ 5,1 },{ 2,2 },{ 7,3 },{ 4,4 },{ 1,5 },{ 6,6 },{ 3,7 }, // quality = 3

  { 0,0 },{ 4,0 },{ 1,1 },{ 5,1 },{ 3,2 },{ 7,2 },{ 2,3 },{ 6,3 }, // quality = 4
  { 0,4 },{ 4,4 },{ 1,5 },{ 5,5 },{ 3,6 },{ 7,6 },{ 2,7 },{ 6,7 },

  { 0,0 },{ 0,2 },{ 0,4 },{ 0,6 },{ 1,1 },{ 1,3 },{ 1,5 },{ 1,7 }, // quality = 5
  { 2,0 },{ 2,2 },{ 2,4 },{ 2,6 },{ 3,1 },{ 3,3 },{ 3,5 },{ 3,7 },
  { 4,0 },{ 4,2 },{ 4,4 },{ 4,6 },{ 5,1 },{ 5,3 },{ 5,5 },{ 5,7 },
  { 6,0 },{ 6,2 },{ 6,4 },{ 6,6 },{ 7,1 },{ 7,3 },{ 7,5 },{ 7,7 },

  { 0,0 },{ 4,4 },{ 0,4 },{ 4,0 },{ 2,2 },{ 6,6 },{ 2,6 },{ 6,2 }, // quality = 6
  { 0,2 },{ 4,6 },{ 0,6 },{ 4,2 },{ 2,0 },{ 6,4 },{ 2,4 },{ 6,0 },
  { 1,1 },{ 5,5 },{ 1,5 },{ 5,1 },{ 3,3 },{ 7,7 },{ 3,7 },{ 7,3 },
  { 1,3 },{ 5,7 },{ 1,7 },{ 5,3 },{ 3,1 },{ 7,5 },{ 3,5 },{ 7,1 },
  { 0,1 },{ 4,5 },{ 0,5 },{ 4,1 },{ 2,3 },{ 6,7 },{ 2,7 },{ 6,3 },
  { 0,3 },{ 4,7 },{ 0,7 },{ 4,3 },{ 2,1 },{ 6,5 },{ 2,5 },{ 6,1 },
  { 1,0 },{ 5,4 },{ 1,4 },{ 5,0 },{ 3,2 },{ 7,6 },{ 3,6 },{ 7,2 },
  { 1,2 },{ 5,6 },{ 1,6 },{ 5,2 },{ 3,0 },{ 7,4 },{ 3,4 },{ 7,0 },
};

#define STMP(x, y) (shared_tmp[(y)][(x)])
#define SIN(x, y)  (shared_in[(y) & (8 * SPP_SHARED_BLOCK_NUM_Y - 1)][(x)])
#define SOUT(x, y) (shared_out[(y) & (8 * SPP_SHARED_BLOCK_NUM_Y - 1)][(x)])

__device__ void load_8x8(float shared_in[8 * SPP_SHARED_BLOCK_NUM_Y][8 * SPP_SHARED_BLOCK_NUM_X], cudaTextureObject_t texSrc, int thWorker, int shared_bx, int shared_by, int src_global_bx, int src_global_by) {
    #pragma unroll
    for (int y = 0; y < 8; y++) {
        SIN(shared_bx * 8 + thWorker, shared_by * 8 + y) = tex2D<float>(texSrc, src_global_bx * 8 + thWorker, src_global_by * 8 + y);
    }
}
__device__ void zero_8x8(float shared_out[8 * SPP_SHARED_BLOCK_NUM_Y][8 * SPP_SHARED_BLOCK_NUM_X], int thWorker, int shared_bx, int shared_by) {
    #pragma unroll
    for (int y = 0; y < 8; y++) {
        SOUT(shared_bx * 8 + thWorker, shared_by * 8 + y) = 0.0f;
    }
}
__device__ void load_8x8tmp(float shared_tmp[8][9], float shared_in[8 * SPP_SHARED_BLOCK_NUM_Y][8 * SPP_SHARED_BLOCK_NUM_X], int thWorker, int shared_bx, int shared_by, int offset1_x, int offset1_y, int offset2_x, int offset2_y) {
    #pragma unroll
    for (int y = 0; y < 8; y++) {
        STMP(thWorker, y) = SIN(shared_bx*8 + offset1_x + thWorker, shared_by*8 + offset1_y + y);
    }
}
__device__ void add_8x8tmp(float shared_out[8 * SPP_SHARED_BLOCK_NUM_Y][8 * SPP_SHARED_BLOCK_NUM_X], float shared_tmp[8][9], int thWorker, int shared_bx, int shared_by, int offset1_x, int offset1_y, int offset2_x, int offset2_y) {
    #pragma unroll
    for (int y = 0; y < 8; y++) {
        SOUT(shared_bx * 8 + offset1_x + thWorker, shared_by * 8 + offset1_y + y) += STMP(thWorker, y);
    }
}
#if ENABLE_CUDA_FP16_HOST
__device__ void load_8x8tmp(__half2 shared_tmp[8][9], float shared_in[8 * SPP_SHARED_BLOCK_NUM_Y][8 * SPP_SHARED_BLOCK_NUM_X], int thWorker, int shared_bx, int shared_by, int offset1_x, int offset1_y, int offset2_x, int offset2_y) {
#if ENABLE_CUDA_FP16_DEVICE
    #pragma unroll
    for (int y = 0; y < 8; y++) {
        float v0 = SIN(shared_bx * 8 + offset1_x + thWorker, shared_by * 8 + offset1_y + y);
        float v1 = SIN(shared_bx * 8 + offset2_x + thWorker, shared_by * 8 + offset2_y + y);
        STMP(thWorker, y) = __floats2half2_rn(v0, v1);
    }
#endif //#if ENABLE_CUDA_FP16_DEVICE
}
__device__ void add_8x8tmp(float shared_out[8 * SPP_SHARED_BLOCK_NUM_Y][8 * SPP_SHARED_BLOCK_NUM_X], __half2 shared_tmp[8][9], int thWorker, int shared_bx, int shared_by, int offset1_x, int offset1_y, int offset2_x, int offset2_y) {
#if ENABLE_CUDA_FP16_DEVICE
    #pragma unroll
    for (int y = 0; y < 8; y++) {
        __half2 v = STMP(thWorker, y);
        float v0 = __half2float(v.x);
        float v1 = __half2float(v.y);
        SOUT(shared_bx * 8 + offset1_x + thWorker, shared_by * 8 + offset1_y + y) += v0;
        SOUT(shared_bx * 8 + offset2_x + thWorker, shared_by * 8 + offset2_y + y) += v1;
    }
#endif //#if ENABLE_CUDA_FP16_DEVICE
}
#endif //#if ENABLE_CUDA_FP16_HOST
template<typename TypePixel, int bit_depth>
__device__ void store_8x8(char *__restrict__ pDst, int dstPitch, int dstWidth, int dstHeight, float shared_out[8 * SPP_SHARED_BLOCK_NUM_Y][8 * SPP_SHARED_BLOCK_NUM_X], int thWorker, int shared_bx, int shared_by, int dst_global_bx, int dst_global_by, int quality) {
    const int dst_global_x = dst_global_bx * 8 + thWorker;
    if (dst_global_x < dstWidth) {
        const int dst_block_offset = (dst_global_by * 8) * dstPitch + dst_global_x * sizeof(TypePixel);
        char *ptrDst = pDst + dst_block_offset;

        const int y_max = dstHeight - dst_global_by * 8;
        #pragma unroll
        for (int y = 0; y < 8; y++, ptrDst += dstPitch) {
            if (y < y_max) {
                *(TypePixel *)ptrDst = (TypePixel)clamp(SOUT(shared_bx * 8 + thWorker, shared_by * 8 + y) * (float)(1 << (bit_depth - quality)), 0.0f, (float)((1 << bit_depth) - 0.5f));
            }
        }
    }
}

template<typename TypePixel, int bit_depth, typename TypeDct, bool usefp16, typename TypeQP>
__global__ void kernel_spp(
    char *__restrict__ ptrDst,
    cudaTextureObject_t texSrc,
    const int dstPitch,
    const int dstWidth,
    const int dstHeight,
    const char *__restrict__ ptrQP,
    const int qpPitch,
    const int qpWidth,
    const int qpHeight,
    const int qpBlockShift,
    const float qpMul,
    const int quality,
    const float strength,
    const float threshA, const float threshB) {
    const int thWorker = threadIdx.x; // SPP_THREAD_BLOCK_X
    const int local_bx = threadIdx.y; // SPP_THREAD_BLOCK_Y
    const int global_bx = blockIdx.x * SPP_BLOCK_SIZE_X + local_bx;
    int global_by = blockIdx.y * SPP_LOOP_COUNT_BLOCK;
    const int count = 1 << quality;

    __shared__ TypeDct shared_tmp[SPP_THREAD_BLOCK_Y][8][9];
    __shared__ float shared_in[8 * SPP_SHARED_BLOCK_NUM_Y][8 * SPP_SHARED_BLOCK_NUM_X];
    __shared__ float shared_out[8 * SPP_SHARED_BLOCK_NUM_Y][8 * SPP_SHARED_BLOCK_NUM_X];

    load_8x8(shared_in, texSrc, thWorker, local_bx, 0, global_bx - 1, global_by - 1);
    zero_8x8(shared_out, thWorker, local_bx, 0);
    if (local_bx < (SPP_SHARED_BLOCK_NUM_X - SPP_BLOCK_SIZE_X)) {
        load_8x8(shared_in, texSrc, thWorker, local_bx + SPP_BLOCK_SIZE_X, 0, global_bx + SPP_BLOCK_SIZE_X - 1, global_by - 1);
        zero_8x8(shared_out, thWorker, local_bx + SPP_BLOCK_SIZE_X, 0);
    }

    for (int local_by = 0; local_by <= SPP_LOOP_COUNT_BLOCK; local_by++, global_by++) {
        const TypeQP qp = *(TypeQP *)(ptrQP + min(global_by >> qpBlockShift, qpHeight) * qpPitch + min(global_bx >> qpBlockShift, qpWidth) * sizeof(TypeQP));
        const TypeDct threshold = setval<TypeDct>((1.0f / (8.0f * (float)(1<<bit_depth))) * (calcThreshold((float)qp * qpMul, threshA, threshB) * ((float)(1 << 2) + strength) - 1.0f));

        load_8x8(shared_in, texSrc, thWorker, local_bx, local_by+1, global_bx - 1, global_by);
        zero_8x8(shared_out, thWorker, local_bx, local_by+1);
        if (local_bx < (SPP_SHARED_BLOCK_NUM_X - SPP_BLOCK_SIZE_X)) {
            load_8x8(shared_in, texSrc, thWorker, local_bx + SPP_BLOCK_SIZE_X, local_by+1, global_bx + SPP_BLOCK_SIZE_X - 1, global_by);
            zero_8x8(shared_out, thWorker, local_bx + SPP_BLOCK_SIZE_X, local_by+1);
        }
        __syncthreads();

        //fp16では、icount2つ分をSIMD的に2並列で処理する
        for (int icount = 0; icount < count; icount += (usefp16) ? 2 : 1) {
            const uchar2 offset = SPP_DEBLOCK_OFFSET[count - 1 + icount];
            const int offset1_x = offset.x;
            const int offset1_y = offset.y;
            int offset2_x = 0;
            int offset2_y = 0;
            if (usefp16) {
                const uchar2 offset2 = SPP_DEBLOCK_OFFSET[count + icount];
                offset2_x = offset2.x;
                offset2_y = offset2.y;
            }

            //fp16では、icount2つ分をSIMD的に2並列で処理するが、
            //add_8x8tmpで衝突する可能性がある
            //衝突するのは、warp間の書き込み先がオーバーラップした場合なので、
            //そこで、warp間を1ブロック空けて処理することでオーバーラップが起こらないようにする
            //1warp=32threadで、SPP_THREAD_BLOCK_X(blockDim)=8なので、
            //warp1=local_bx[0-3], warp2=local_bx[4-7]
            //local_bx 3と4の間をひとつ開けるようにする
            //どのみち、1ブロックは別に処理する必要があるので、都合がよい
            int target_bx = (local_bx < 4) ? local_bx : local_bx + 1;
            load_8x8tmp(shared_tmp[local_bx], shared_in, thWorker, target_bx, local_by, offset1_x, offset1_y, offset2_x, offset2_y);
            dct8x8<TypeDct>(shared_tmp[local_bx], thWorker);
            threshold8x8(shared_tmp[local_bx], thWorker, threshold);
            idct8x8<TypeDct>(shared_tmp[local_bx], thWorker);
            add_8x8tmp(shared_out, shared_tmp[local_bx], thWorker, target_bx, local_by, offset1_x, offset1_y, offset2_x, offset2_y);
            if (usefp16) {
                __syncthreads();
            }
            if (local_bx < 1) {
                target_bx = 4;
                load_8x8tmp(shared_tmp[local_bx], shared_in, thWorker, target_bx, local_by, offset1_x, offset1_y, offset2_x, offset2_y);
                dct8x8<TypeDct>(shared_tmp[local_bx], thWorker);
                threshold8x8(shared_tmp[local_bx], thWorker, threshold);
                idct8x8<TypeDct>(shared_tmp[local_bx], thWorker);
                add_8x8tmp(shared_out, shared_tmp[local_bx], thWorker, target_bx, local_by, offset1_x, offset1_y, offset2_x, offset2_y);
            }
            __syncthreads();
        }
        if (local_by > 0) {
            store_8x8<TypePixel, bit_depth>(ptrDst, dstPitch, dstWidth, dstHeight, shared_out, thWorker, local_bx+1, local_by, global_bx, global_by-1, quality);
        }
        __syncthreads();
    }
}

template<typename TypePixel>
cudaError_t setTexFieldSmooth(cudaTextureObject_t& texSrc, const RGYFrameInfo *pFrame) {
    texSrc = 0;

    cudaResourceDesc resDescSrc;
    memset(&resDescSrc, 0, sizeof(resDescSrc));
    resDescSrc.resType = cudaResourceTypePitch2D;
    resDescSrc.res.pitch2D.desc = cudaCreateChannelDesc<TypePixel>();
    resDescSrc.res.pitch2D.pitchInBytes = pFrame->pitch;
    resDescSrc.res.pitch2D.width = pFrame->width;
    resDescSrc.res.pitch2D.height = pFrame->height;
    resDescSrc.res.pitch2D.devPtr = (uint8_t *)pFrame->ptr;

    cudaTextureDesc texDescSrc;
    memset(&texDescSrc, 0, sizeof(texDescSrc));
    texDescSrc.addressMode[0]   = cudaAddressModeWrap;
    texDescSrc.addressMode[1]   = cudaAddressModeWrap;
    texDescSrc.filterMode       = cudaFilterModePoint;
    texDescSrc.readMode         = cudaReadModeNormalizedFloat;
    texDescSrc.normalizedCoords = 0;

    return cudaCreateTextureObject(&texSrc, &resDescSrc, &texDescSrc, nullptr);
}

template<typename TypePixel, int bit_depth, typename TypeDct, bool usefp16, typename TypeQP>
cudaError_t run_spp(RGYFrameInfo *pOutputPlane,
    const RGYFrameInfo *pSrc,
    const RGYFrameInfo *pQP,
    const int qpBlockShift,
    const float qpMul,
    const int quality,
    const float strength,
    const float threshold,
    cudaStream_t stream) {
    cudaTextureObject_t texSrc = 0;
    auto cudaerr = setTexFieldSmooth<TypePixel>(texSrc, pSrc);
    if (cudaerr != cudaSuccess) {
        return cudaerr;
    }

    dim3 blockSize(SPP_THREAD_BLOCK_X, SPP_THREAD_BLOCK_Y);
    dim3 gridSize(divCeil(pOutputPlane->width, 8 * SPP_BLOCK_SIZE_X), divCeil(pOutputPlane->height, 8 * SPP_LOOP_COUNT_BLOCK));

    const float W = 5.0f;
    const float thresh_a = (threshold + W) / (2.0f * W);
    const float thresh_b = (W * W - threshold * threshold) / (2.0f * W);

    kernel_spp<TypePixel, bit_depth, TypeDct, usefp16, TypeQP><<<gridSize, blockSize, 0, stream>>>(
        (char * )pOutputPlane->ptr,
        texSrc,
        pOutputPlane->pitch,
        pOutputPlane->width,
        pOutputPlane->height,
        (const char *)pQP->ptr,
        pQP->pitch,
        pQP->width,
        pQP->height,
        qpBlockShift, qpMul,
        quality, strength,
        thresh_a, thresh_b);
    cudaerr = cudaGetLastError();
    cudaDestroyTextureObject(texSrc);
    return cudaerr;
}

template<typename TypePixel, int bit_depth, typename TypeDct, bool usefp16, typename TypeQP>
cudaError_t run_spp_frame(RGYFrameInfo *pOutputFrame,
    const RGYFrameInfo *pSrc,
    const CUFrameBuf *qpFrame,
    const float qpMul,
    const int quality,
    const float strength,
    const float threshold,
    cudaStream_t stream) {
    const auto planeSrcY = getPlane(pSrc, RGY_PLANE_Y);
    const auto planeSrcU = getPlane(pSrc, RGY_PLANE_U);
    const auto planeSrcV = getPlane(pSrc, RGY_PLANE_V);
    auto planeOutputY = getPlane(pOutputFrame, RGY_PLANE_Y);
    auto planeOutputU = getPlane(pOutputFrame, RGY_PLANE_U);
    auto planeOutputV = getPlane(pOutputFrame, RGY_PLANE_V);

    const int qpBlockShiftY = 1;
    const int qpBlockShiftUV = RGY_CSP_CHROMA_FORMAT[pSrc->csp] == RGY_CHROMAFMT_YUV420 ? 0 : 1;

    auto cudaerr = run_spp<TypePixel, bit_depth, TypeDct, usefp16, TypeQP>(&planeOutputY, &planeSrcY, &qpFrame->frame, qpBlockShiftY, qpMul, quality, strength, threshold, stream);
    if (cudaerr != cudaSuccess) {
        return cudaerr;
    }
    cudaerr = run_spp<TypePixel, bit_depth, TypeDct, usefp16, TypeQP>(&planeOutputU, &planeSrcU, &qpFrame->frame, qpBlockShiftUV, qpMul, quality, strength, threshold, stream);
    if (cudaerr != cudaSuccess) {
        return cudaerr;
    }
    cudaerr = run_spp<TypePixel, bit_depth, TypeDct, usefp16, TypeQP>(&planeOutputV, &planeSrcV, &qpFrame->frame, qpBlockShiftUV, qpMul, quality, strength, threshold, stream);
    if (cudaerr != cudaSuccess) {
        return cudaerr;
    }
    return cudaerr;
}

template<typename TypeQP4>
__global__ void kernel_gen_qp_table(
    char *__restrict__ ptrQPDst,
    const int qpDstPitch,
    const int qpDstWidth,
    const int qpDstHeight,
    const char *__restrict__ ptrQPSrc,
    const char *__restrict__ ptrQPSrcB,
    const int qpSrcPitch,
    const int qpSrcWidth,
    const int qpSrcHeight,
    const float qpMul,
    const float bRatio) {
    const int thx = threadIdx.x;
    const int thy = threadIdx.y;
    const int qpx = (blockIdx.x * blockDim.x + thx) * 4;
    const int qpy = blockIdx.y * blockDim.y + thy;

    const int qpsx = max(qpx, qpSrcWidth);
    const int qpsy = max(qpy, qpSrcHeight);

    ptrQPDst  += (qpy  * qpDstPitch + qpx  * sizeof(decltype(TypeQP4::x)));
    ptrQPSrc  += (qpsy * qpSrcPitch + qpsx * sizeof(decltype(TypeQP4::x)));
    ptrQPSrcB += (qpsy * qpSrcPitch + qpsx * sizeof(decltype(TypeQP4::x)));

    TypeQP4 qpSrc  = *(TypeQP4 *)ptrQPSrc;
    TypeQP4 qpSrcB = *(TypeQP4 *)ptrQPSrcB;

    TypeQP4 qp4;
    qp4.x = max(1, (int)(lerpf(qpSrcB.x, qpSrc.x, bRatio) * qpMul + 0.5f));
    qp4.y = max(1, (int)(lerpf(qpSrcB.y, qpSrc.y, bRatio) * qpMul + 0.5f));
    qp4.z = max(1, (int)(lerpf(qpSrcB.z, qpSrc.z, bRatio) * qpMul + 0.5f));
    qp4.w = max(1, (int)(lerpf(qpSrcB.w, qpSrc.w, bRatio) * qpMul + 0.5f));
    if (qpx < qpDstWidth && qpy < qpDstHeight) {
        *(TypeQP4 *)ptrQPDst = qp4;
    }
}

template<typename TypeQP4>
cudaError_t run_gen_qp_table(
    RGYFrameInfo *pQPDst,
    const RGYFrameInfo *pQPSrc,
    const RGYFrameInfo *pQPSrcB,
    const float qpMul,
    const float bRatio,
    cudaStream_t stream) {
    dim3 blockSize(32, 8);
    dim3 gridSize(divCeil(pQPDst->width, 4 * blockSize.x), divCeil(pQPDst->height, blockSize.y));

    kernel_gen_qp_table<TypeQP4> << <gridSize, blockSize, 0, stream >> > (
        (char *)pQPDst->ptr,
        pQPDst->pitch,
        (pQPDst->width + 3) & (~3),
        pQPDst->height,
        (char *)pQPSrc->ptr,
        (char *)pQPSrcB->ptr,
        pQPSrc->pitch,
        (pQPSrc->width + 3) & (~3),
        pQPSrc->height,
        qpMul, bRatio);
    auto cudaerr = cudaGetLastError();
    return cudaerr;
}

template<typename TypeQP4>
__global__ void kernel_set_qp(
    char *__restrict__ ptrQP,
    const int qpPitch,
    const int qpWidth,
    const int qpHeight,
    const int qp) {
    const int thx = threadIdx.x;
    const int thy = threadIdx.y;
    const int qpx = (blockIdx.x * blockDim.x + thx) * 4;
    const int qpy = blockIdx.y * blockDim.y + thy;

    TypeQP4 qp4;
    qp4.x = qp;
    qp4.y = qp;
    qp4.z = qp;
    qp4.w = qp;
    if (qpx < qpWidth && qpy < qpHeight) {
        ptrQP += (qpy * qpPitch + qpx * sizeof(decltype(TypeQP4::x)));
        *(TypeQP4 *)ptrQP = qp4;
    }
}

template<typename TypeQP4>
cudaError_t run_set_qp(
    RGYFrameInfo *pQP,
    const int qp,
    cudaStream_t stream) {
    dim3 blockSize(32, 8);
    dim3 gridSize(divCeil(pQP->width, 4 * blockSize.x), divCeil(pQP->height, blockSize.y));

    kernel_set_qp<TypeQP4> << <gridSize, blockSize, 0, stream >> > (
        (char *)pQP->ptr,
        pQP->pitch,
        (pQP->width + 3) & (~3),
        pQP->height,
        qp);
    auto cudaerr = cudaGetLastError();
    return cudaerr;
}

NVEncFilterSmooth::NVEncFilterSmooth() : m_qp(), m_qpSrc(), m_qpSrcB(), m_qpTableRef(nullptr), m_qpTableErrCount(0) {
    m_sFilterName = _T("smooth");
}

NVEncFilterSmooth::~NVEncFilterSmooth() {
    close();
}

RGY_ERR NVEncFilterSmooth::check_param(shared_ptr<NVEncFilterParamSmooth> prm) {
    if (prm->frameOut.height <= 0 || prm->frameOut.width <= 0) {
        AddMessage(RGY_LOG_ERROR, _T("Invalid parameter.\n"));
        return RGY_ERR_INVALID_PARAM;
    }
    if (prm->smooth.quality < 0 || prm->smooth.quality > VPP_SMOOTH_MAX_QUALITY_LEVEL) {
        AddMessage(RGY_LOG_ERROR, _T("Invalid parameter (quality).\n"));
        return RGY_ERR_INVALID_PARAM;
    }
    if (prm->smooth.qp <= 0 || prm->smooth.qp > 63) {
        AddMessage(RGY_LOG_ERROR, _T("Invalid parameter (qp).\n"));
        return RGY_ERR_INVALID_PARAM;
    }
#if !ENABLE_CUDA_FP16_HOST
    if (prm->smooth.prec == VPP_FP_PRECISION_FP16) {
        AddMessage(RGY_LOG_WARN, _T("prec=fp16 not compiled in this build, switching to fp32.\n"));
        prm->smooth.prec = VPP_FP_PRECISION_FP32;
    }
#endif
    return RGY_ERR_NONE;
}

RGY_ERR NVEncFilterSmooth::init(shared_ptr<NVEncFilterParam> pParam, shared_ptr<RGYLog> pPrintMes) {
    RGY_ERR sts = RGY_ERR_NONE;
    m_pPrintMes = pPrintMes;
    auto prm = std::dynamic_pointer_cast<NVEncFilterParamSmooth>(pParam);
    if (!prm) {
        AddMessage(RGY_LOG_ERROR, _T("Invalid parameter type.\n"));
        return RGY_ERR_INVALID_PARAM;
    }
    //パラメータチェック
    if (check_param(prm) != RGY_ERR_NONE) {
        return RGY_ERR_INVALID_PARAM;
    }
    if (prm->smooth.prec == VPP_FP_PRECISION_AUTO) {
        prm->smooth.prec =
#if ENABLE_CUDA_FP16_HOST
            RGY_CSP_BIT_DEPTH[pParam->frameIn.csp] <= 8 &&
            prm->smooth.quality > 0 &&
            ((prm->compute_capability.first == 6 && prm->compute_capability.second == 0)
                || prm->compute_capability.first >= 7)
            ? VPP_FP_PRECISION_FP16 : VPP_FP_PRECISION_FP32;
#else
            VPP_FP_PRECISION_FP32;
#endif
    }
    if (!m_pParam
        || cmpFrameInfoCspResolution(&m_pParam->frameIn, &pParam->frameIn)
        || (std::dynamic_pointer_cast<NVEncFilterParamSmooth>(m_pParam)
            && std::dynamic_pointer_cast<NVEncFilterParamSmooth>(m_pParam)->smooth != prm->smooth)) {

        auto cudaerr = AllocFrameBuf(prm->frameOut, 1);
        if (cudaerr != cudaSuccess) {
            AddMessage(RGY_LOG_ERROR, _T("failed to allocate memory for output: %s.\n"), char_to_tstring(cudaGetErrorName(cudaerr)).c_str());
            return RGY_ERR_MEMORY_ALLOC;
        }
        prm->frameOut.pitch = m_pFrameBuf[0]->frame.pitch;
        AddMessage(RGY_LOG_DEBUG, _T("allocated output buffer: %dx%pixym1[3], pitch %pixym1[3], %s.\n"),
            m_pFrameBuf[0]->frame.width, m_pFrameBuf[0]->frame.height, m_pFrameBuf[0]->frame.pitch, RGY_CSP_NAMES[m_pFrameBuf[0]->frame.csp]);

        cudaerr = m_qp.alloc(qp_size(pParam->frameIn.width), qp_size(pParam->frameIn.height), RGY_CSP_Y8);
        if (cudaerr != cudaSuccess) {
            AddMessage(RGY_LOG_ERROR, _T("failed to allocate memory for qp table: %s.\n"), char_to_tstring(cudaGetErrorName(cudaerr)).c_str());
            return RGY_ERR_MEMORY_ALLOC;
        }
        AddMessage(RGY_LOG_DEBUG, _T("allocated qp table buffer: %dx%pixym1[3], pitch %pixym1[3], %s.\n"),
            m_qp.frame.width, m_qp.frame.height, m_qp.frame.pitch, RGY_CSP_NAMES[m_qp.frame.csp]);

        setFilterInfo(pParam->print());
    }
    m_qpTableRef = prm->qpTableRef;
    m_pParam = pParam;
    return sts;
}

tstring NVEncFilterParamSmooth::print() const {
    return smooth.print();
}

float NVEncFilterSmooth::getQPMul(int qpScaleType) {
    switch (qpScaleType) {
    case 0/*mpeg1*/: return 4.0f;
    case 1/*mpeg2*/: return 2.0f;
    case 2/*h264*/:  return 1.0f;
    case 3/*VP56*/:  //return (63 - qscale + 2);
    default:
        return 0.0f;
    }
}

RGY_ERR NVEncFilterSmooth::run_filter(const RGYFrameInfo *pInputFrame, RGYFrameInfo **ppOutputFrames, int *pOutputFrameNum, cudaStream_t stream) {
    RGY_ERR sts = RGY_ERR_NONE;

    auto prm = std::dynamic_pointer_cast<NVEncFilterParamSmooth>(m_pParam);
    if (!prm) {
        AddMessage(RGY_LOG_ERROR, _T("Invalid parameter type.\n"));
        return RGY_ERR_INVALID_PARAM;
    }

    if (pInputFrame->ptr == nullptr) {
        //終了
        *pOutputFrameNum = 0;
        ppOutputFrames[0] = nullptr;
        return sts;
    }
    //エラーチェック
    const auto memcpyKind = getCudaMemcpyKind(pInputFrame->deivce_mem, m_pFrameBuf[0]->frame.deivce_mem);
    if (memcpyKind != cudaMemcpyDeviceToDevice) {
        AddMessage(RGY_LOG_ERROR, _T("only supported on device memory.\n"));
        return RGY_ERR_INVALID_CALL;
    }
    if (m_pParam->frameOut.csp != m_pParam->frameIn.csp) {
        AddMessage(RGY_LOG_ERROR, _T("csp does not match.\n"));
        return RGY_ERR_INVALID_PARAM;
    }

    //出力先のフレーム
    CUFrameBuf *pOutFrame = nullptr;
    *pOutputFrameNum = 1;
    if (ppOutputFrames[0] == nullptr) {
        pOutFrame = m_pFrameBuf[m_nFrameIdx].get();
        ppOutputFrames[0] = &pOutFrame->frame;
        ppOutputFrames[0]->picstruct = pInputFrame->picstruct;
        m_nFrameIdx = (m_nFrameIdx + 1) % m_pFrameBuf.size();
    }

    //入力フレームのQPテーブルへの参照を取得
    std::shared_ptr<RGYFrameDataQP> qpInput;
    if (prm->smooth.useQPTable) {
        for (auto &data : pInputFrame->dataList) {
            if (data->dataType() == RGY_FRAME_DATA_QP) {
                auto ptr = dynamic_cast<RGYFrameDataQP *>(data.get());
                if (ptr == nullptr) {
                    AddMessage(RGY_LOG_ERROR, _T("Failed to get RGYFrameDataQP.\n"));
                    return RGY_ERR_UNSUPPORTED;
                }
                auto ptrRef = m_qpTableRef->get(ptr);
                if (!ptrRef) {
                    AddMessage(RGY_LOG_ERROR, _T("Failed to get ref to RGYFrameDataQP.\n"));
                    return RGY_ERR_UNSUPPORTED;
                }
                qpInput = std::move(ptrRef);
            }
        }
        if (!qpInput) {
            m_qpTableErrCount++;
            AddMessage(RGY_LOG_DEBUG, _T("Failed to get qp table from input file %d: inputID %d, %lld\n"), m_qpTableErrCount, pInputFrame->inputFrameId, pInputFrame->timestamp);
            if (m_qpTableErrCount >= prm->smooth.maxQPTableErrCount) {
                AddMessage(RGY_LOG_ERROR, _T("Failed to get qp table from input file for more than %d times, please specify \"qp\" for --vpp-smooth.\n"), m_qpTableErrCount);
                return RGY_ERR_UNSUPPORTED;
            }
            //ひとまず、前のQPテーブルで代用する
            qpInput = m_qpSrc;
        } else {
            m_qpTableErrCount = 0;
        }
    }

    //実際に計算用に使用するQPテーブルの選択、あるいは作成
    CUFrameBuf *targetQPTable = nullptr;
    float qpMul = 1.0f;
    if (!!qpInput) {
        auto cudaerr = cudaStreamWaitEvent(stream, qpInput->event(), 0);
        if (cudaerr != cudaSuccess) {
            AddMessage(RGY_LOG_ERROR, _T("error in cudaStreamWaitEvent(): %s.\n"), char_to_tstring(cudaGetErrorName(cudaerr)).c_str());
            return RGY_ERR_MEMORY_ALLOC;
        }
        qpMul = getQPMul(qpInput->qpScaleType());
        if (qpMul <= 0.0f) {
            AddMessage(RGY_LOG_ERROR, _T("Unsupported qp scale type.\n"));
            return RGY_ERR_UNSUPPORTED;
        }
        const bool isBFrame = qpInput->frameType() == 3;
        if (isBFrame) {
            m_qpSrcB = qpInput;
            targetQPTable = &m_qp;
            cudaerr = run_gen_qp_table<uchar4>(&m_qp.frame, &m_qpSrc->qpDev()->frame, &m_qpSrcB->qpDev()->frame, qpMul, prm->smooth.bratio, stream);
            if (cudaerr != cudaSuccess) {
                AddMessage(RGY_LOG_ERROR, _T("error in run_set_qp(): %s.\n"), char_to_tstring(cudaGetErrorName(cudaerr)).c_str());
                return RGY_ERR_MEMORY_ALLOC;
            }
            qpMul = 1.0f; //run_gen_qp_tableの中で反映済み
        } else {
            m_qpSrc = qpInput;
            targetQPTable = m_qpSrc->qpDev();
        }
    } else {
        targetQPTable = &m_qp;
        auto cudaerr = run_set_qp<uchar4>(&m_qp.frame, prm->smooth.qp, stream);
        if (cudaerr != cudaSuccess) {
            AddMessage(RGY_LOG_ERROR, _T("error in run_set_qp(): %s.\n"), char_to_tstring(cudaGetErrorName(cudaerr)).c_str());
            return RGY_ERR_MEMORY_ALLOC;
        }
    }

    static const std::map<RGY_CSP, decltype(run_spp_frame<uint8_t, 8, float, false, uint8_t>)*> func_list_fp32 = {
        { RGY_CSP_YV12,      run_spp_frame<uint8_t,   8, float, false, uint8_t> },
        { RGY_CSP_YV12_16,   run_spp_frame<uint16_t, 16, float, false, uint8_t> },
        { RGY_CSP_YUV444,    run_spp_frame<uint8_t,   8, float, false, uint8_t> },
        { RGY_CSP_YUV444_16, run_spp_frame<uint16_t, 16, float, false, uint8_t> }
    };
#if ENABLE_CUDA_FP16_HOST
    static const std::map<RGY_CSP, decltype(run_spp_frame<uint8_t, 8, float, false, uint8_t>) *> func_list_fp16 = {
        { RGY_CSP_YV12,      run_spp_frame<uint8_t,   8, __half2, true, uint8_t> },
        { RGY_CSP_YV12_16,   run_spp_frame<uint16_t, 16, __half2, true, uint8_t> },
        { RGY_CSP_YUV444,    run_spp_frame<uint8_t,   8, __half2, true, uint8_t> },
        { RGY_CSP_YUV444_16, run_spp_frame<uint16_t, 16, __half2, true, uint8_t> },
    };
    const auto &func_list = (prm->smooth.prec == VPP_FP_PRECISION_FP32) ? func_list_fp32 : func_list_fp16;
#else
    const auto &func_list = func_list_fp32;
#endif
    if (func_list.count(pInputFrame->csp) == 0) {
        AddMessage(RGY_LOG_ERROR, _T("unsupported csp %s.\n"), RGY_CSP_NAMES[pInputFrame->csp]);
        return RGY_ERR_UNSUPPORTED;
    }
    auto cudaerr = func_list.at(pInputFrame->csp)(ppOutputFrames[0],
        pInputFrame,
        targetQPTable,
        qpMul,
        prm->smooth.quality,
        prm->smooth.strength,
        prm->smooth.threshold,
        stream
        );
    if (cudaerr != cudaSuccess) {
        AddMessage(RGY_LOG_ERROR, _T("error in run_spp_frame(): %s.\n"), char_to_tstring(cudaGetErrorName(cudaerr)).c_str());
        return RGY_ERR_MEMORY_ALLOC;
    }

    return sts;
}

void NVEncFilterSmooth::close() {
    AddMessage(RGY_LOG_DEBUG, _T("closed smooth filter.\n"));
}
