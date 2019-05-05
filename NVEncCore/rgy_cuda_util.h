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
#ifndef __RGY_CUDA_UTIL_H__
#define __RGY_CUDA_UTIL_H__

static const int WARP_SIZE_2N = 5;
static const int WARP_SIZE = (1<<WARP_SIZE_2N);

#if __CUDACC_VER_MAJOR__ >= 9
#define __shfl(x, y)     __shfl_sync(0xFFFFFFFFU, x, y)
#define __shfl_up(x, y)   __shfl_up_sync(0xFFFFFFFFU, x, y)
#define __shfl_down(x, y) __shfl_down_sync(0xFFFFFFFFU, x, y)
#define __shfl_xor(x, y)  __shfl_xor_sync(0xFFFFFFFFU, x, y)
#define __any(x)          __any_sync(0xFFFFFFFFU, x)
#define __all(x)          __all_sync(0xFFFFFFFFU, x)
#endif

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

#endif //__RGY_CUDA_UTIL_H__
