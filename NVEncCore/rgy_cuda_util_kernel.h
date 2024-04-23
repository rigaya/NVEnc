// -----------------------------------------------------------------------------------------
// QSVEnc/NVEnc by rigaya
// -----------------------------------------------------------------------------------------
// The MIT License
//
// Copyright (c) 2011-2016 rigaya
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
// --------------------------------------------------------------------------------------------

#pragma once
#ifndef __RGY_CUDA_UTIL_KERNEL_H__
#define __RGY_CUDA_UTIL_KERNEL_H__

#include "cuda_fp16.h"

static const int WARP_SIZE_2N = 5;
static const int WARP_SIZE = (1<<WARP_SIZE_2N);

#define RGY_FLT_EPS (1e-6)

struct __align__(sizeof(int) * 8) int8 {
    int s0, s1, s2, s3, s4, s5, s6, s7;
};

struct __align__(sizeof(float) * 8) float8 {
    float s0, s1, s2, s3, s4, s5, s6, s7;
    float8() {};
    __device__ float8(float val) {
        s0 = val;
        s1 = val;
        s2 = val;
        s3 = val;
        s4 = val;
        s5 = val;
        s6 = val;
        s7 = val;
    };
    __device__ float8(float v0, float v1, float v2, float v3, float v4, float v5, float v6, float v7) {
        this->s0 = v0;
        this->s1 = v1;
        this->s2 = v2;
        this->s3 = v3;
        this->s4 = v4;
        this->s5 = v5;
        this->s6 = v6;
        this->s7 = v7;
    }
};

struct __align__(sizeof(__half) * 8) __half8 {
    __half s0, s1, s2, s3, s4, s5, s6, s7;
    __half8() {};
    __device__ __half8(__half val) {
        s0 = val;
        s1 = val;
        s2 = val;
        s3 = val;
        s4 = val;
        s5 = val;
        s6 = val;
        s7 = val;
    };
};

struct __align__(sizeof(__half2) * 4) __half2x4 {
    __half2 s0, s1, s2, s3;
};

union __align__(sizeof(__half8)) half8 {
    __half8 h;
    __half2x4 h2;
public:
    half8() {};
    __device__ half8(const half8& src) {
        this->h2.s0 = src.h2.s0;
        this->h2.s1 = src.h2.s1;
        this->h2.s2 = src.h2.s2;
        this->h2.s3 = src.h2.s3;
    }
    __device__ half8(const __half8& src) {
        this->h = src;
    }
    __device__ half8(__half val) {
        __half2 vh2 = { val, val };
        this->h2.s0 = vh2;
        this->h2.s1 = vh2;
        this->h2.s2 = vh2;
        this->h2.s3 = vh2;
    }
#if defined(__CUDACC__) && (__CUDA_ARCH__ >= 530 || !defined(__CUDA_ARCH__)) && !defined(__CUDA_NO_HALF2_OPERATORS__)
    __device__ half8(float val) {
        __half2 vh2 = __float2half2_rn(val);
        this->h2.s0 = vh2;
        this->h2.s1 = vh2;
        this->h2.s2 = vh2;
        this->h2.s3 = vh2;
    }
#else
    __device__ half8(float val) { };
#endif
    __device__ half8(__half v0, __half v1, __half v2, __half v3, __half v4, __half v5, __half v6, __half v7) {
        this->h.s0 = v0;
        this->h.s1 = v1;
        this->h.s2 = v2;
        this->h.s3 = v3;
        this->h.s4 = v4;
        this->h.s5 = v5;
        this->h.s6 = v6;
        this->h.s7 = v7;
    }
    __device__ half8 &operator=(const half8& src) {
        this->h2.s0 = src.h2.s0;
        this->h2.s1 = src.h2.s1;
        this->h2.s2 = src.h2.s2;
        this->h2.s3 = src.h2.s3;
        return *this;
    }

};

static __device__ float2 operator*(float2 a, float b) {
    a.x *= b;
    a.y *= b;
    return a;
}

static __device__ float2 operator*(float2 a, float2 b) {
    a.x *= b.x;
    a.y *= b.y;
    return a;
}

static __device__ float2& operator*=(float2& a, float b) {
    a.x *= b;
    a.y *= b;
    return a;
}

static __device__ float2& operator*=(float2& a, float2 b) {
    a.x *= b.x;
    a.y *= b.y;
    return a;
}

static __device__ float2 operator+(float2 a, float b) {
    a.x += b;
    a.y += b;
    return a;
}

static __device__ float2 operator+(float2 a, float2 b) {
    a.x += b.x;
    a.y += b.y;
    return a;
}

static __device__ float2& operator+=(float2& a, float b) {
    a.x += b;
    a.y += b;
    return a;
}

static __device__ float2& operator+=(float2& a, float2 b) {
    a.x += b.x;
    a.y += b.y;
    return a;
}

static __device__ float2 operator-(float2 a, float b) {
    a.x -= b;
    a.y -= b;
    return a;
}

static __device__ float2 operator-(float2 a, float2 b) {
    a.x -= b.x;
    a.y -= b.y;
    return a;
}

static __device__ float2& operator-=(float2& a, float b) {
    a.x -= b;
    a.y -= b;
    return a;
}

static __device__ float2& operator-=(float2& a, float2 b) {
    a.x -= b.x;
    a.y -= b.y;
    return a;
}

static __device__ float8 max(float8 a, float8 b) {
    a.s0 = max(a.s0, b.s0);
    a.s1 = max(a.s1, b.s1);
    a.s2 = max(a.s2, b.s2);
    a.s3 = max(a.s3, b.s3);
    a.s4 = max(a.s4, b.s4);
    a.s5 = max(a.s5, b.s5);
    a.s6 = max(a.s6, b.s6);
    a.s7 = max(a.s7, b.s7);
    return a;
}

static __device__ float8 __expf(float8 a) {
    a.s0 = __expf(a.s0);
    a.s1 = __expf(a.s1);
    a.s2 = __expf(a.s2);
    a.s3 = __expf(a.s3);
    a.s4 = __expf(a.s4);
    a.s5 = __expf(a.s5);
    a.s6 = __expf(a.s6);
    a.s7 = __expf(a.s7);
    return a;
}

static __device__ float8 operator*(float8 a, float b) {
    a.s0 *= b;
    a.s1 *= b;
    a.s2 *= b;
    a.s3 *= b;
    a.s4 *= b;
    a.s5 *= b;
    a.s6 *= b;
    a.s7 *= b;
    return a;
}

static __device__ float8 operator*(float8 a, float8 b) {
    a.s0 *= b.s0;
    a.s1 *= b.s1;
    a.s2 *= b.s2;
    a.s3 *= b.s3;
    a.s4 *= b.s4;
    a.s5 *= b.s5;
    a.s6 *= b.s6;
    a.s7 *= b.s7;
    return a;
}

static __device__ float8& operator*=(float8& a, float b) {
    a.s0 *= b;
    a.s1 *= b;
    a.s2 *= b;
    a.s3 *= b;
    a.s4 *= b;
    a.s5 *= b;
    a.s6 *= b;
    a.s7 *= b;
    return a;
}

static __device__ float8& operator*=(float8& a, float8 b) {
    a.s0 *= b.s0;
    a.s1 *= b.s1;
    a.s2 *= b.s2;
    a.s3 *= b.s3;
    a.s4 *= b.s4;
    a.s5 *= b.s5;
    a.s6 *= b.s6;
    a.s7 *= b.s7;
    return a;
}

static __device__ float8 operator+(float8 a, float b) {
    a.s0 += b;
    a.s1 += b;
    a.s2 += b;
    a.s3 += b;
    a.s4 += b;
    a.s5 += b;
    a.s6 += b;
    a.s7 += b;
    return a;
}

static __device__ float8 operator+(float8 a, float8 b) {
    a.s0 += b.s0;
    a.s1 += b.s1;
    a.s2 += b.s2;
    a.s3 += b.s3;
    a.s4 += b.s4;
    a.s5 += b.s5;
    a.s6 += b.s6;
    a.s7 += b.s7;
    return a;
}

static __device__ float8& operator+=(float8& a, float b) {
    a.s0 += b;
    a.s1 += b;
    a.s2 += b;
    a.s3 += b;
    a.s4 += b;
    a.s5 += b;
    a.s6 += b;
    a.s7 += b;
    return a;
}

static __device__ float8& operator+=(float8& a, float8 b) {
    a.s0 += b.s0;
    a.s1 += b.s1;
    a.s2 += b.s2;
    a.s3 += b.s3;
    a.s4 += b.s4;
    a.s5 += b.s5;
    a.s6 += b.s6;
    a.s7 += b.s7;
    return a;
}

static __device__ float8 operator-(float8 a, float b) {
    a.s0 -= b;
    a.s1 -= b;
    a.s2 -= b;
    a.s3 -= b;
    a.s4 -= b;
    a.s5 -= b;
    a.s6 -= b;
    a.s7 -= b;
    return a;
}

static __device__ float8 operator-(float8 a, float8 b) {
    a.s0 -= b.s0;
    a.s1 -= b.s1;
    a.s2 -= b.s2;
    a.s3 -= b.s3;
    a.s4 -= b.s4;
    a.s5 -= b.s5;
    a.s6 -= b.s6;
    a.s7 -= b.s7;
    return a;
}

static __device__ float8& operator-=(float8& a, float b) {
    a.s0 -= b;
    a.s1 -= b;
    a.s2 -= b;
    a.s3 -= b;
    a.s4 -= b;
    a.s5 -= b;
    a.s6 -= b;
    a.s7 -= b;
    return a;
}

static __device__ float8& operator-=(float8& a, float8 b) {
    a.s0 -= b.s0;
    a.s1 -= b.s1;
    a.s2 -= b.s2;
    a.s3 -= b.s3;
    a.s4 -= b.s4;
    a.s5 -= b.s5;
    a.s6 -= b.s6;
    a.s7 -= b.s7;
    return a;
}

#if defined(__CUDACC__) && (__CUDA_ARCH__ >= 530 || !defined(__CUDA_ARCH__)) && !defined(__CUDA_NO_HALF2_OPERATORS__)

static __device__ half8 operator*(half8 a, float b) {
    __half2 bh2 = __float2half2_rn(b);
    a.h2.s0 *= bh2;
    a.h2.s1 *= bh2;
    a.h2.s2 *= bh2;
    a.h2.s3 *= bh2;
    return a;
}

static __device__ half8 operator*(half8 a, __half b) {
    __half2 bh2 = { b, b };
    a.h2.s0 *= bh2;
    a.h2.s1 *= bh2;
    a.h2.s2 *= bh2;
    a.h2.s3 *= bh2;
    return a;
}

static __device__ half8 operator*(half8 a, half8 b) {
    a.h2.s0 *= b.h2.s0;
    a.h2.s1 *= b.h2.s1;
    a.h2.s2 *= b.h2.s2;
    a.h2.s3 *= b.h2.s3;
    return a;
}

static __device__ half8& operator*=(half8& a, float b) {
    __half2 bh2 = __float2half2_rn(b);
    a.h2.s0 *= bh2;
    a.h2.s1 *= bh2;
    a.h2.s2 *= bh2;
    a.h2.s3 *= bh2;
    return a;
}

static __device__ half8& operator*=(half8& a, __half b) {
    __half2 bh2 = { b, b };
    a.h2.s0 *= bh2;
    a.h2.s1 *= bh2;
    a.h2.s2 *= bh2;
    a.h2.s3 *= bh2;
    return a;
}

static __device__ half8& operator*=(half8& a, half8 b) {
    a.h2.s0 *= b.h2.s0;
    a.h2.s1 *= b.h2.s1;
    a.h2.s2 *= b.h2.s2;
    a.h2.s3 *= b.h2.s3;
    return a;
}

static __device__ half8 operator+(half8 a, float b) {
    __half2 bh2 = __float2half2_rn(b);
    a.h2.s0 += bh2;
    a.h2.s1 += bh2;
    a.h2.s2 += bh2;
    a.h2.s3 += bh2;
    return a;
}

static __device__ half8 operator+(half8 a, __half b) {
    __half2 bh2 = { b, b };
    a.h2.s0 += bh2;
    a.h2.s1 += bh2;
    a.h2.s2 += bh2;
    a.h2.s3 += bh2;
    return a;
}

static __device__ half8 operator+(half8 a, half8 b) {
    a.h2.s0 += b.h2.s0;
    a.h2.s1 += b.h2.s1;
    a.h2.s2 += b.h2.s2;
    a.h2.s3 += b.h2.s3;
    return a;
}

static __device__ half8& operator+=(half8& a, float b) {
    __half2 bh2 = __float2half2_rn(b);
    a.h2.s0 += bh2;
    a.h2.s1 += bh2;
    a.h2.s2 += bh2;
    a.h2.s3 += bh2;
    return a;
}

static __device__ half8& operator+=(half8& a, __half b) {
    __half2 bh2 = { b, b };
    a.h2.s0 += bh2;
    a.h2.s1 += bh2;
    a.h2.s2 += bh2;
    a.h2.s3 += bh2;
    return a;
}

static __device__ half8& operator+=(half8& a, half8 b) {
    a.h2.s0 += b.h2.s0;
    a.h2.s1 += b.h2.s1;
    a.h2.s2 += b.h2.s2;
    a.h2.s3 += b.h2.s3;
    return a;
}

static __device__ half8 operator-(half8 a, float b) {
    __half2 bh2 = __float2half2_rn(b);
    a.h2.s0 -= bh2;
    a.h2.s1 -= bh2;
    a.h2.s2 -= bh2;
    a.h2.s3 -= bh2;
    return a;
}

static __device__ half8 operator-(half8 a, __half b) {
    __half2 bh2 = { b, b };
    a.h2.s0 -= bh2;
    a.h2.s1 -= bh2;
    a.h2.s2 -= bh2;
    a.h2.s3 -= bh2;
    return a;
}

static __device__ half8 operator-(half8 a, half8 b) {
    a.h2.s0 -= b.h2.s0;
    a.h2.s1 -= b.h2.s1;
    a.h2.s2 -= b.h2.s2;
    a.h2.s3 -= b.h2.s3;
    return a;
}

static __device__ half8& operator-=(half8& a, float b) {
    __half2 bh2 = __float2half2_rn(b);
    a.h2.s0 -= bh2;
    a.h2.s1 -= bh2;
    a.h2.s2 -= bh2;
    a.h2.s3 -= bh2;
    return a;
}

static __device__ half8& operator-=(half8& a, __half b) {
    __half2 bh2 = { b, b };
    a.h2.s0 -= bh2;
    a.h2.s1 -= bh2;
    a.h2.s2 -= bh2;
    a.h2.s3 -= bh2;
    return a;
}

static __device__ half8& operator-=(half8& a, half8 b) {
    a.h2.s0 -= b.h2.s0;
    a.h2.s1 -= b.h2.s1;
    a.h2.s2 -= b.h2.s2;
    a.h2.s3 -= b.h2.s3;
    return a;
}
#endif

#if __CUDACC_VER_MAJOR__ >= 9
#define __shfl(x, y)     __shfl_sync(0xFFFFFFFFU, x, y)
#define __shfl_up(x, y)   __shfl_up_sync(0xFFFFFFFFU, x, y)
#define __shfl_down(x, y) __shfl_down_sync(0xFFFFFFFFU, x, y)
#define __shfl_xor(x, y)  __shfl_xor_sync(0xFFFFFFFFU, x, y)
#define __any(x)          __any_sync(0xFFFFFFFFU, x)
#define __all(x)          __all_sync(0xFFFFFFFFU, x)
#endif

// cuda_fp16.hppが定義してくれないことがある
#define RGY_HALF_TO_US(var) *(reinterpret_cast<unsigned short *>(&(var)))
#define RGY_HALF_TO_CUS(var) *(reinterpret_cast<const unsigned short *>(&(var)))
#define RGY_HALF2_TO_UI(var) *(reinterpret_cast<unsigned int *>(&(var)))
#define RGY_HALF2_TO_CUI(var) *(reinterpret_cast<const unsigned int *>(&(var)))

template<typename Type, int width>
__inline__ __device__
Type warp_sum(Type val) {
    static_assert(width <= WARP_SIZE, "width too big for warp_sum");
    if (width >= 32) val += __shfl_xor(val, 16);
    if (width >= 16) val += __shfl_xor(val, 8);
    if (width >=  8) val += __shfl_xor(val, 4);
    if (width >=  4) val += __shfl_xor(val, 2);
    if (width >=  2) val += __shfl_xor(val, 1);
    return val;
}

template<typename Type>
__inline__ __device__
Type warp_sum(Type val, int width) {
    if (width >= 32) val += __shfl_xor(val, 16);
    if (width >= 16) val += __shfl_xor(val, 8);
    if (width >= 8) val += __shfl_xor(val, 4);
    if (width >= 4) val += __shfl_xor(val, 2);
    if (width >= 2) val += __shfl_xor(val, 1);
    return val;
}

template<typename Type, int BLOCK_X, int BLOCK_Y>
__inline__ __device__
Type block_sum(Type val, Type *shared) {
    static_assert(BLOCK_X * BLOCK_Y <= WARP_SIZE * WARP_SIZE, "block size too big for block_sum");
    const int lid = threadIdx.y * BLOCK_X + threadIdx.x;
    const int lane    = lid & (WARP_SIZE - 1);
    const int warp_id = lid >> WARP_SIZE_2N;

    val = warp_sum<Type, WARP_SIZE>(val);

    if (lane == 0) shared[warp_id] = val;

    __syncthreads();

    if (warp_id == 0) {
        val = (lid * WARP_SIZE < BLOCK_X * BLOCK_Y) ? shared[lane] : 0;
        val = warp_sum<Type, BLOCK_X * BLOCK_Y / WARP_SIZE>(val);
    }
    return val;
}

template<typename Type>
__inline__ __device__
Type block_sum(Type val, Type *shared, int blockX, int blockY) {
    const int lid = threadIdx.y * blockX + threadIdx.x;
    const int lane = lid & (WARP_SIZE - 1);
    const int warp_id = lid >> WARP_SIZE_2N;

    val = warp_sum<Type, WARP_SIZE>(val);

    if (lane == 0) shared[warp_id] = val;

    __syncthreads();

    if (warp_id == 0) {
        val = (lid * WARP_SIZE < blockX * blockY) ? shared[lane] : 0;
        val = warp_sum<Type>(val, (blockX * blockY + WARP_SIZE - 1) / WARP_SIZE);
    }
    return val;
}

template<typename Type, int width>
__inline__ __device__
Type warp_min(Type val) {
    static_assert(width <= WARP_SIZE, "width too big for warp_min");
    if (width >= 32) val = min(val, __shfl_xor(val, 16));
    if (width >= 16) val = min(val, __shfl_xor(val, 8));
    if (width >=  8) val = min(val, __shfl_xor(val, 4));
    if (width >=  4) val = min(val, __shfl_xor(val, 2));
    if (width >=  2) val = min(val, __shfl_xor(val, 1));
    return val;
}

template<typename Type, int BLOCK_X, int BLOCK_Y>
__inline__ __device__
Type block_min(Type val, Type *shared) {
    static_assert(BLOCK_X * BLOCK_Y <= WARP_SIZE * WARP_SIZE, "block size too big for block_min");
    const int lid = threadIdx.y * BLOCK_X + threadIdx.x;
    const int lane    = lid & (WARP_SIZE - 1);
    const int warp_id = lid >> WARP_SIZE_2N;

    val = warp_min<Type, WARP_SIZE>(val);

    if (lane == 0) shared[warp_id] = val;

    __syncthreads();

    if (warp_id == 0) {
        val = (lid * WARP_SIZE < BLOCK_X * BLOCK_Y) ? shared[lane] : 0;
        val = warp_min<Type, BLOCK_X * BLOCK_Y / WARP_SIZE>(val);
    }
    return val;
}

template<typename Type, int width>
__inline__ __device__
Type warp_max(Type val) {
    static_assert(width <= WARP_SIZE, "width too big for warp_max");
    if (width >= 32) val = max(val, __shfl_xor(val, 16));
    if (width >= 16) val = max(val, __shfl_xor(val, 8));
    if (width >= 8)  val = max(val, __shfl_xor(val, 4));
    if (width >= 4)  val = max(val, __shfl_xor(val, 2));
    if (width >= 2)  val = max(val, __shfl_xor(val, 1));
    return val;
}

template<typename Type, int BLOCK_X, int BLOCK_Y>
__inline__ __device__
Type block_max(Type val, Type *shared) {
    static_assert(BLOCK_X * BLOCK_Y <= WARP_SIZE * WARP_SIZE, "block size too big for block_max");
    const int lid = threadIdx.y * BLOCK_X + threadIdx.x;
    const int lane = lid & (WARP_SIZE - 1);
    const int warp_id = lid >> WARP_SIZE_2N;

    val = warp_max<Type, WARP_SIZE>(val);

    if (lane == 0) shared[warp_id] = val;

    __syncthreads();

    if (warp_id == 0) {
        val = (lid * WARP_SIZE < BLOCK_X *BLOCK_Y) ? shared[lane] : 0;
        val = warp_max<Type, BLOCK_X *BLOCK_Y / WARP_SIZE>(val);
    }
    return val;
}

static __device__ float lerpf(float v0, float v1, float ratio) {
    return v0 + (v1 - v0) * ratio;
}

static __device__ int wrap_idx(const int idx, const int min, const int max) {
    if (idx < min) {
        return min - idx;
    }
    if (idx > max) {
        return max - (idx - max);
    }
    return idx;
}

template<typename T>
static __device__ T *selectptr(T *ptr0, T *ptr1, T *ptr2, const int idx) {
    if (idx == 1) return ptr1;
    if (idx == 2) return ptr2;
    return ptr0;
}

#endif //__RGY_CUDA_UTIL_KERNEL_H__
