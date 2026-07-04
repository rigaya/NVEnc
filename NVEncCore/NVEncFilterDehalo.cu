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

#include <cmath>
#include <algorithm>
#include <utility>
#include "convert_csp.h"
#include "NVEncFilterDehalo.h"
#pragma warning (push)
#pragma warning (disable: 4819)
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#pragma warning (pop)
#include "rgy_cuda_util_kernel.h"

static const int DEHALO_BLOCK_X = 32;
static const int DEHALO_BLOCK_Y = 8;

template<typename Type>
__device__ __forceinline__ int dehalo_read_pix_clamp(const uint8_t *frame, const int pitch, const int width, const int height, int x, int y) {
    x = clamp(x, 0, width - 1);
    y = clamp(y, 0, height - 1);
    const auto ptr = (const Type *)(frame + y * pitch + x * sizeof(Type));
    return (int)ptr[0];
}

template<typename Type>
__global__ void kernel_dehalo_expand(const uint8_t *src, const int srcPitch, uint8_t *dst, const int dstPitch,
    const int width, const int height, const float rx, const float ry) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;

    const int irx = (int)ceilf(rx);
    const int iry = (int)ceilf(ry);
    const float invRx2 = 1.0f / (rx * rx);
    const float invRy2 = 1.0f / (ry * ry);

    int m = dehalo_read_pix_clamp<Type>(src, srcPitch, width, height, x, y);
    for (int dy = -iry; dy <= iry; dy++) {
        const float dyF = (float)dy;
        const float yTerm = dyF * dyF * invRy2;
        if (yTerm > 1.0f) continue;
        const float xLimitSq = 1.0f - yTerm;
        for (int dx = -irx; dx <= irx; dx++) {
            const float dxF = (float)dx;
            if (dxF * dxF * invRx2 > xLimitSq) continue;
            const int v = dehalo_read_pix_clamp<Type>(src, srcPitch, width, height, x + dx, y + dy);
            if (v > m) m = v;
        }
    }

    auto dstPix = (Type *)(dst + y * dstPitch + x * sizeof(Type));
    dstPix[0] = (Type)m;
}

template<typename Type>
__global__ void kernel_dehalo_inpand(const uint8_t *src, const int srcPitch, uint8_t *dst, const int dstPitch,
    const int width, const int height, const float rx, const float ry) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;

    const int irx = (int)ceilf(rx);
    const int iry = (int)ceilf(ry);
    const float invRx2 = 1.0f / (rx * rx);
    const float invRy2 = 1.0f / (ry * ry);

    int m = dehalo_read_pix_clamp<Type>(src, srcPitch, width, height, x, y);
    for (int dy = -iry; dy <= iry; dy++) {
        const float dyF = (float)dy;
        const float yTerm = dyF * dyF * invRy2;
        if (yTerm > 1.0f) continue;
        const float xLimitSq = 1.0f - yTerm;
        for (int dx = -irx; dx <= irx; dx++) {
            const float dxF = (float)dx;
            if (dxF * dxF * invRx2 > xLimitSq) continue;
            const int v = dehalo_read_pix_clamp<Type>(src, srcPitch, width, height, x + dx, y + dy);
            if (v < m) m = v;
        }
    }

    auto dstPix = (Type *)(dst + y * dstPitch + x * sizeof(Type));
    dstPix[0] = (Type)m;
}

template<typename Type, int bit_depth>
__global__ void kernel_dehalo_mask(const uint8_t *src, const int srcPitch, const uint8_t *expanded, const int expandedPitch,
    const uint8_t *inpand, const int inpandPitch, uint8_t *mask, const int maskPitch,
    const int width, const int height, const int loScaled, const int hiScaled) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;

    static const int max_val = (1 << bit_depth) - 1;
    const int s = dehalo_read_pix_clamp<Type>(src, srcPitch, width, height, x, y);
    const int e = dehalo_read_pix_clamp<Type>(expanded, expandedPitch, width, height, x, y);
    const int i = dehalo_read_pix_clamp<Type>(inpand, inpandPitch, width, height, x, y);
    const int range = e - i;

    int abs_diff = 0;
    if (range > 0) {
        long long num = (long long)(s - i) * (long long)max_val;
        int v = (int)(num / (long long)range);
        v = clamp(v, 0, max_val);
        abs_diff = v;
    }

    int m = 0;
    if (hiScaled > loScaled) {
        long long num = (long long)(abs_diff - loScaled) * (long long)max_val;
        int v = (int)(num / (long long)(hiScaled - loScaled));
        m = clamp(v, 0, max_val);
    } else {
        m = (abs_diff >= loScaled) ? max_val : 0;
    }

    auto maskPix = (Type *)(mask + y * maskPitch + x * sizeof(Type));
    maskPix[0] = (Type)m;
}

template<typename Type, int bit_depth>
__global__ void kernel_dehalo_apply(const uint8_t *src, const int srcPitch, const uint8_t *expanded, const int expandedPitch,
    const uint8_t *inpand, const int inpandPitch, const uint8_t *mask, const int maskPitch,
    uint8_t *dst, const int dstPitch, const int width, const int height, const float darkstr, const float brightstr) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;

    static const int max_val = (1 << bit_depth) - 1;
    const float s = (float)dehalo_read_pix_clamp<Type>(src, srcPitch, width, height, x, y);
    const float e = (float)dehalo_read_pix_clamp<Type>(expanded, expandedPitch, width, height, x, y);
    const float i = (float)dehalo_read_pix_clamp<Type>(inpand, inpandPitch, width, height, x, y);
    const float m = (float)dehalo_read_pix_clamp<Type>(mask, maskPitch, width, height, x, y);
    const float mn = m / (float)max_val;

    float r = s - mn * darkstr * (s - i) + mn * brightstr * (e - s);
    r = clamp(r, 0.0f, (float)max_val);

    auto dstPix = (Type *)(dst + y * dstPitch + x * sizeof(Type));
    dstPix[0] = (Type)(int)(r + 0.5f);
}

template<typename Type>
__global__ void kernel_dehalo_square_morph(const uint8_t *src, const int srcPitch, uint8_t *dst, const int dstPitch,
    const int width, const int height, const int radius, const bool expand) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;

    int m = dehalo_read_pix_clamp<Type>(src, srcPitch, width, height, x, y);
    for (int dy = -radius; dy <= radius; dy++) {
        for (int dx = -radius; dx <= radius; dx++) {
            const int v = dehalo_read_pix_clamp<Type>(src, srcPitch, width, height, x + dx, y + dy);
            m = expand ? ((m < v) ? v : m) : ((m > v) ? v : m);
        }
    }

    auto dstPix = (Type *)(dst + y * dstPitch + x * sizeof(Type));
    dstPix[0] = (Type)m;
}

template<typename Type>
__global__ void kernel_dehalo_square_range(const uint8_t *src, const int srcPitch, uint8_t *dst, const int dstPitch,
    const int width, const int height, const int radiusExpand, const int radiusInpand) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;

    int mn = dehalo_read_pix_clamp<Type>(src, srcPitch, width, height, x, y);
    int mx = mn;
    for (int dy = -radiusExpand; dy <= radiusExpand; dy++) {
        for (int dx = -radiusExpand; dx <= radiusExpand; dx++) {
            const int v = dehalo_read_pix_clamp<Type>(src, srcPitch, width, height, x + dx, y + dy);
            mx = (mx < v) ? v : mx;
        }
    }
    for (int dy = -radiusInpand; dy <= radiusInpand; dy++) {
        for (int dx = -radiusInpand; dx <= radiusInpand; dx++) {
            const int v = dehalo_read_pix_clamp<Type>(src, srcPitch, width, height, x + dx, y + dy);
            mn = (mn > v) ? v : mn;
        }
    }

    auto dstPix = (Type *)(dst + y * dstPitch + x * sizeof(Type));
    dstPix[0] = (Type)(mx - mn);
}

template<typename Type, int bit_depth>
__global__ void kernel_dehalo_alpha_lets(const uint8_t *clp, const int clpPitch, const uint8_t *halos, const int halosPitch,
    const uint8_t *are, const int arePitch, const uint8_t *ugly, const int uglyPitch,
    uint8_t *lets, const int letsPitch, const int width, const int height, const int loScaled, const int highsens) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;

    static const int max_val = (1 << bit_depth) - 1;
    const float clpPix = (float)dehalo_read_pix_clamp<Type>(clp, clpPitch, width, height, x, y);
    const float halosPix = (float)dehalo_read_pix_clamp<Type>(halos, halosPitch, width, height, x, y);
    const float arePix = (float)dehalo_read_pix_clamp<Type>(are, arePitch, width, height, x, y);
    const float uglyPix = (float)dehalo_read_pix_clamp<Type>(ugly, uglyPitch, width, height, x, y);
    const float range_size = (float)max_val + 1.0f;
    const float eps = 0.001f;
    const float soBase = ((arePix - uglyPix) / (arePix + eps) * (float)max_val) - (float)loScaled;
    const float soGain = ((arePix + range_size) / (range_size * 2.0f)) + (float)highsens * 0.01f;
    const float so = clamp(soBase * soGain, 0.0f, (float)max_val);
    const float r = halosPix + (clpPix - halosPix) * (so / (float)max_val);

    auto dstPix = (Type *)(lets + y * letsPitch + x * sizeof(Type));
    dstPix[0] = (Type)(int)(clamp(r, 0.0f, (float)max_val) + 0.5f);
}

template<typename Type, int bit_depth>
__global__ void kernel_dehalo_alpha_clamp(const uint8_t *src, const int srcPitch,
    const uint8_t *limitLow, const int limitLowPitch, const uint8_t *limitHigh, const int limitHighPitch,
    uint8_t *dst, const int dstPitch, const int width, const int height) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;

    static const int max_val = (1 << bit_depth) - 1;
    const int srcPix = dehalo_read_pix_clamp<Type>(src, srcPitch, width, height, x, y);
    const int lowPix = dehalo_read_pix_clamp<Type>(limitLow, limitLowPitch, width, height, x, y);
    const int highPix = dehalo_read_pix_clamp<Type>(limitHigh, limitHighPitch, width, height, x, y);
    const int r = clamp(srcPix, (lowPix < highPix) ? lowPix : highPix, (lowPix < highPix) ? highPix : lowPix);

    auto dstPix = (Type *)(dst + y * dstPitch + x * sizeof(Type));
    dstPix[0] = (Type)clamp(r, 0, max_val);
}

template<typename Type, int bit_depth>
__global__ void kernel_dehalo_alpha_them(const uint8_t *clp, const int clpPitch, const uint8_t *remove, const int removePitch,
    uint8_t *dst, const int dstPitch, const int width, const int height, const float darkstr, const float brightstr) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;

    static const int max_val = (1 << bit_depth) - 1;
    const float xPix = (float)dehalo_read_pix_clamp<Type>(clp, clpPitch, width, height, x, y);
    const float yPix = (float)dehalo_read_pix_clamp<Type>(remove, removePitch, width, height, x, y);
    const float str = (xPix < yPix) ? darkstr : brightstr;
    const float r = xPix - (xPix - yPix) * str;

    auto dstPix = (Type *)(dst + y * dstPitch + x * sizeof(Type));
    dstPix[0] = (Type)(int)(clamp(r, 0.0f, (float)max_val) + 0.5f);
}

template<typename Type, int bit_depth>
static RGY_ERR dehalo_process_y_typed(RGYFrameInfo *pOutputFrame, const RGYFrameInfo *pInputFrame,
    RGYFrameInfo *pExpanded, RGYFrameInfo *pInpand, RGYFrameInfo *pMask,
    const VppDehalo& prm, const int loScaled, const int hiScaled, cudaStream_t stream) {
    dim3 blockSize(DEHALO_BLOCK_X, DEHALO_BLOCK_Y);
    dim3 gridSize(divCeil(pInputFrame->width, blockSize.x), divCeil(pInputFrame->height, blockSize.y));
    const auto width = pInputFrame->width;
    const auto height = pInputFrame->height;

    kernel_dehalo_expand<Type><<<gridSize, blockSize, 0, stream>>>(pInputFrame->ptr[0], pInputFrame->pitch[0],
        pExpanded->ptr[0], pExpanded->pitch[0], width, height, prm.rx, prm.ry);
    auto cudaerr = cudaGetLastError();
    if (cudaerr != cudaSuccess) return err_to_rgy(cudaerr);

    kernel_dehalo_inpand<Type><<<gridSize, blockSize, 0, stream>>>(pInputFrame->ptr[0], pInputFrame->pitch[0],
        pInpand->ptr[0], pInpand->pitch[0], width, height, prm.rx, prm.ry);
    cudaerr = cudaGetLastError();
    if (cudaerr != cudaSuccess) return err_to_rgy(cudaerr);

    kernel_dehalo_mask<Type, bit_depth><<<gridSize, blockSize, 0, stream>>>(pInputFrame->ptr[0], pInputFrame->pitch[0],
        pExpanded->ptr[0], pExpanded->pitch[0], pInpand->ptr[0], pInpand->pitch[0],
        pMask->ptr[0], pMask->pitch[0], width, height, loScaled, hiScaled);
    cudaerr = cudaGetLastError();
    if (cudaerr != cudaSuccess) return err_to_rgy(cudaerr);

    kernel_dehalo_apply<Type, bit_depth><<<gridSize, blockSize, 0, stream>>>(pInputFrame->ptr[0], pInputFrame->pitch[0],
        pExpanded->ptr[0], pExpanded->pitch[0], pInpand->ptr[0], pInpand->pitch[0],
        pMask->ptr[0], pMask->pitch[0], pOutputFrame->ptr[0], pOutputFrame->pitch[0],
        width, height, prm.darkstr, prm.brightstr);
    cudaerr = cudaGetLastError();
    if (cudaerr != cudaSuccess) return err_to_rgy(cudaerr);

    return RGY_ERR_NONE;
}

static RGY_ERR dehalo_process_y(RGYFrameInfo *pOutputFrame, const RGYFrameInfo *pInputFrame,
    RGYFrameInfo *pExpanded, RGYFrameInfo *pInpand, RGYFrameInfo *pMask,
    const VppDehalo& prm, const int loScaled, const int hiScaled, cudaStream_t stream) {
    if (RGY_CSP_BIT_DEPTH[pInputFrame->csp] > 8) {
        return dehalo_process_y_typed<uint16_t, 16>(pOutputFrame, pInputFrame, pExpanded, pInpand, pMask, prm, loScaled, hiScaled, stream);
    } else {
        return dehalo_process_y_typed<uint8_t, 8>(pOutputFrame, pInputFrame, pExpanded, pInpand, pMask, prm, loScaled, hiScaled, stream);
    }
}

template<typename Type>
static RGY_ERR dehalo_alpha_range_y_typed(RGYFrameInfo *pOutputFrame, const RGYFrameInfo *pInputFrame,
    const int radiusExpand, const int radiusInpand, cudaStream_t stream) {
    dim3 blockSize(DEHALO_BLOCK_X, DEHALO_BLOCK_Y);
    dim3 gridSize(divCeil(pInputFrame->width, blockSize.x), divCeil(pInputFrame->height, blockSize.y));
    kernel_dehalo_square_range<Type><<<gridSize, blockSize, 0, stream>>>(pInputFrame->ptr[0], pInputFrame->pitch[0],
        pOutputFrame->ptr[0], pOutputFrame->pitch[0], pInputFrame->width, pInputFrame->height, radiusExpand, radiusInpand);
    auto cudaerr = cudaGetLastError();
    return (cudaerr == cudaSuccess) ? RGY_ERR_NONE : err_to_rgy(cudaerr);
}

static RGY_ERR dehalo_alpha_range_y(RGYFrameInfo *pOutputFrame, const RGYFrameInfo *pInputFrame,
    const int radiusExpand, const int radiusInpand, cudaStream_t stream) {
    if (RGY_CSP_BIT_DEPTH[pInputFrame->csp] > 8) {
        return dehalo_alpha_range_y_typed<uint16_t>(pOutputFrame, pInputFrame, radiusExpand, radiusInpand, stream);
    } else {
        return dehalo_alpha_range_y_typed<uint8_t>(pOutputFrame, pInputFrame, radiusExpand, radiusInpand, stream);
    }
}

static int dehalo_alpha_auto_search_radius(const VppDehalo& prm) {
    return std::max((int)std::lround(std::max(prm.rx, prm.ry)), 3);
}

static std::pair<int, int> dehalo_alpha_search_radius(const VppDehalo& prm) {
    int searchRade = prm.searchRade;
    int searchRadi = prm.searchRadi;
    if (searchRade < 0) {
        searchRade = dehalo_alpha_auto_search_radius(prm);
    }
    if (searchRadi < 0) {
        searchRadi = searchRade;
    }
    return std::make_pair(searchRade, searchRadi);
}

template<typename Type>
static RGY_ERR dehalo_alpha_morph_y_typed(RGYFrameInfo *pOutputFrame, const RGYFrameInfo *pInputFrame,
    const int radius, const bool expand, cudaStream_t stream) {
    dim3 blockSize(DEHALO_BLOCK_X, DEHALO_BLOCK_Y);
    dim3 gridSize(divCeil(pInputFrame->width, blockSize.x), divCeil(pInputFrame->height, blockSize.y));
    kernel_dehalo_square_morph<Type><<<gridSize, blockSize, 0, stream>>>(pInputFrame->ptr[0], pInputFrame->pitch[0],
        pOutputFrame->ptr[0], pOutputFrame->pitch[0], pInputFrame->width, pInputFrame->height, radius, expand);
    auto cudaerr = cudaGetLastError();
    return (cudaerr == cudaSuccess) ? RGY_ERR_NONE : err_to_rgy(cudaerr);
}

static RGY_ERR dehalo_alpha_morph_y(RGYFrameInfo *pOutputFrame, const RGYFrameInfo *pInputFrame,
    const int radius, const bool expand, cudaStream_t stream) {
    if (RGY_CSP_BIT_DEPTH[pInputFrame->csp] > 8) {
        return dehalo_alpha_morph_y_typed<uint16_t>(pOutputFrame, pInputFrame, radius, expand, stream);
    } else {
        return dehalo_alpha_morph_y_typed<uint8_t>(pOutputFrame, pInputFrame, radius, expand, stream);
    }
}

template<typename Type, int bit_depth>
static RGY_ERR dehalo_alpha_lets_y_typed(RGYFrameInfo *pOutputFrame, const RGYFrameInfo *pInputFrame,
    const RGYFrameInfo *pHalos, const RGYFrameInfo *pAre, const RGYFrameInfo *pUgly,
    const int loScaled, const int highsens, cudaStream_t stream) {
    dim3 blockSize(DEHALO_BLOCK_X, DEHALO_BLOCK_Y);
    dim3 gridSize(divCeil(pInputFrame->width, blockSize.x), divCeil(pInputFrame->height, blockSize.y));
    kernel_dehalo_alpha_lets<Type, bit_depth><<<gridSize, blockSize, 0, stream>>>(pInputFrame->ptr[0], pInputFrame->pitch[0],
        pHalos->ptr[0], pHalos->pitch[0], pAre->ptr[0], pAre->pitch[0], pUgly->ptr[0], pUgly->pitch[0],
        pOutputFrame->ptr[0], pOutputFrame->pitch[0], pInputFrame->width, pInputFrame->height, loScaled, highsens);
    auto cudaerr = cudaGetLastError();
    return (cudaerr == cudaSuccess) ? RGY_ERR_NONE : err_to_rgy(cudaerr);
}

static RGY_ERR dehalo_alpha_lets_y(RGYFrameInfo *pOutputFrame, const RGYFrameInfo *pInputFrame,
    const RGYFrameInfo *pHalos, const RGYFrameInfo *pAre, const RGYFrameInfo *pUgly,
    const int loScaled, const int highsens, cudaStream_t stream) {
    if (RGY_CSP_BIT_DEPTH[pInputFrame->csp] > 8) {
        return dehalo_alpha_lets_y_typed<uint16_t, 16>(pOutputFrame, pInputFrame, pHalos, pAre, pUgly, loScaled, highsens, stream);
    } else {
        return dehalo_alpha_lets_y_typed<uint8_t, 8>(pOutputFrame, pInputFrame, pHalos, pAre, pUgly, loScaled, highsens, stream);
    }
}

template<typename Type, int bit_depth>
static RGY_ERR dehalo_alpha_clamp_y_typed(RGYFrameInfo *pOutputFrame, const RGYFrameInfo *pInputFrame,
    const RGYFrameInfo *pLimitLow, const RGYFrameInfo *pLimitHigh, cudaStream_t stream) {
    dim3 blockSize(DEHALO_BLOCK_X, DEHALO_BLOCK_Y);
    dim3 gridSize(divCeil(pInputFrame->width, blockSize.x), divCeil(pInputFrame->height, blockSize.y));
    kernel_dehalo_alpha_clamp<Type, bit_depth><<<gridSize, blockSize, 0, stream>>>(pInputFrame->ptr[0], pInputFrame->pitch[0],
        pLimitLow->ptr[0], pLimitLow->pitch[0], pLimitHigh->ptr[0], pLimitHigh->pitch[0],
        pOutputFrame->ptr[0], pOutputFrame->pitch[0], pInputFrame->width, pInputFrame->height);
    auto cudaerr = cudaGetLastError();
    return (cudaerr == cudaSuccess) ? RGY_ERR_NONE : err_to_rgy(cudaerr);
}

static RGY_ERR dehalo_alpha_clamp_y(RGYFrameInfo *pOutputFrame, const RGYFrameInfo *pInputFrame,
    const RGYFrameInfo *pLimitLow, const RGYFrameInfo *pLimitHigh, cudaStream_t stream) {
    if (RGY_CSP_BIT_DEPTH[pInputFrame->csp] > 8) {
        return dehalo_alpha_clamp_y_typed<uint16_t, 16>(pOutputFrame, pInputFrame, pLimitLow, pLimitHigh, stream);
    } else {
        return dehalo_alpha_clamp_y_typed<uint8_t, 8>(pOutputFrame, pInputFrame, pLimitLow, pLimitHigh, stream);
    }
}

template<typename Type, int bit_depth>
static RGY_ERR dehalo_alpha_them_y_typed(RGYFrameInfo *pOutputFrame, const RGYFrameInfo *pInputFrame,
    const RGYFrameInfo *pRemove, const VppDehalo& prm, cudaStream_t stream) {
    dim3 blockSize(DEHALO_BLOCK_X, DEHALO_BLOCK_Y);
    dim3 gridSize(divCeil(pInputFrame->width, blockSize.x), divCeil(pInputFrame->height, blockSize.y));
    kernel_dehalo_alpha_them<Type, bit_depth><<<gridSize, blockSize, 0, stream>>>(pInputFrame->ptr[0], pInputFrame->pitch[0],
        pRemove->ptr[0], pRemove->pitch[0], pOutputFrame->ptr[0], pOutputFrame->pitch[0],
        pInputFrame->width, pInputFrame->height, prm.darkstr, prm.brightstr);
    auto cudaerr = cudaGetLastError();
    return (cudaerr == cudaSuccess) ? RGY_ERR_NONE : err_to_rgy(cudaerr);
}

static RGY_ERR dehalo_alpha_them_y(RGYFrameInfo *pOutputFrame, const RGYFrameInfo *pInputFrame,
    const RGYFrameInfo *pRemove, const VppDehalo& prm, cudaStream_t stream) {
    if (RGY_CSP_BIT_DEPTH[pInputFrame->csp] > 8) {
        return dehalo_alpha_them_y_typed<uint16_t, 16>(pOutputFrame, pInputFrame, pRemove, prm, stream);
    } else {
        return dehalo_alpha_them_y_typed<uint8_t, 8>(pOutputFrame, pInputFrame, pRemove, prm, stream);
    }
}

NVEncFilterDehalo::NVEncFilterDehalo() :
    m_resizeUp(),
    m_resizeDown(),
    m_resizeAlphaHaloDown(),
    m_resizeAlphaHaloUp(),
    m_resizeAlphaUp(),
    m_resizeAlphaDown(),
    m_supersampled(),
    m_expanded(),
    m_inpand(),
    m_mask(),
    m_corrected(),
    m_alphaHalosSmall(),
    m_alphaHalos(),
    m_alphaAre(),
    m_alphaUgly(),
    m_alphaLets(),
    m_alphaLimitLow(),
    m_alphaLimitHigh(),
    m_alphaLimitLowSS(),
    m_alphaLimitHighSS(),
    m_alphaRemoved(),
    m_alphaHaloW(0),
    m_alphaHaloH(0),
    m_ssW(0),
    m_ssH(0),
    m_ssActive(false) {
    m_name = _T("dehalo");
}

NVEncFilterDehalo::~NVEncFilterDehalo() {
    close();
}

RGY_ERR NVEncFilterDehalo::checkParam(const std::shared_ptr<NVEncFilterParamDehalo> prm) {
    if (prm->frameOut.height <= 0 || prm->frameOut.width <= 0) {
        AddMessage(RGY_LOG_ERROR, _T("Invalid parameter.\n"));
        return RGY_ERR_INVALID_PARAM;
    }
    if (prm->frameOut.height < 4 || prm->frameOut.width < 4) {
        AddMessage(RGY_LOG_ERROR, _T("dehalo requires input width/height >= 4 (got %dx%d).\n"),
            prm->frameOut.width, prm->frameOut.height);
        return RGY_ERR_INVALID_PARAM;
    }
    if (!(prm->dehalo.rx >= 0.5f && prm->dehalo.rx <= 10.0f)) {
        AddMessage(RGY_LOG_ERROR, _T("Invalid rx=%.2f: must be in [0.5, 10.0].\n"), prm->dehalo.rx);
        return RGY_ERR_INVALID_PARAM;
    }
    if (!(prm->dehalo.ry >= 0.5f && prm->dehalo.ry <= 10.0f)) {
        AddMessage(RGY_LOG_ERROR, _T("Invalid ry=%.2f: must be in [0.5, 10.0].\n"), prm->dehalo.ry);
        return RGY_ERR_INVALID_PARAM;
    }
    if (!(prm->dehalo.darkstr >= 0.0f && prm->dehalo.darkstr <= 1.0f)) {
        AddMessage(RGY_LOG_ERROR, _T("Invalid darkstr=%.2f: must be in [0.0, 1.0].\n"), prm->dehalo.darkstr);
        return RGY_ERR_INVALID_PARAM;
    }
    if (!(prm->dehalo.brightstr >= 0.0f && prm->dehalo.brightstr <= 1.0f)) {
        AddMessage(RGY_LOG_ERROR, _T("Invalid brightstr=%.2f: must be in [0.0, 1.0].\n"), prm->dehalo.brightstr);
        return RGY_ERR_INVALID_PARAM;
    }
    if (prm->dehalo.lowsens < 0 || prm->dehalo.lowsens > 100) {
        AddMessage(RGY_LOG_ERROR, _T("Invalid lowsens=%d: must be in [0, 100].\n"), prm->dehalo.lowsens);
        return RGY_ERR_INVALID_PARAM;
    }
    if (prm->dehalo.highsens < 0 || prm->dehalo.highsens > 100) {
        AddMessage(RGY_LOG_ERROR, _T("Invalid highsens=%d: must be in [0, 100].\n"), prm->dehalo.highsens);
        return RGY_ERR_INVALID_PARAM;
    }
    if (!(prm->dehalo.ss >= 1.0f && prm->dehalo.ss <= 4.0f)) {
        AddMessage(RGY_LOG_ERROR, _T("Invalid ss=%.2f: must be in [1.0, 4.0].\n"), prm->dehalo.ss);
        return RGY_ERR_INVALID_PARAM;
    }
    if (!((prm->dehalo.searchRade == FILTER_DEFAULT_DEHALO_SEARCH_RADIUS_AUTO) || (prm->dehalo.searchRade >= 1 && prm->dehalo.searchRade <= 10))) {
        AddMessage(RGY_LOG_ERROR, _T("Invalid search_rade=%d: must be auto or in [1, 10].\n"), prm->dehalo.searchRade);
        return RGY_ERR_INVALID_PARAM;
    }
    if (!((prm->dehalo.searchRadi == FILTER_DEFAULT_DEHALO_SEARCH_RADIUS_AUTO) || (prm->dehalo.searchRadi >= 1 && prm->dehalo.searchRadi <= 10))) {
        AddMessage(RGY_LOG_ERROR, _T("Invalid search_radi=%d: must be auto or in [1, 10].\n"), prm->dehalo.searchRadi);
        return RGY_ERR_INVALID_PARAM;
    }
    return RGY_ERR_NONE;
}

RGY_ERR NVEncFilterDehalo::allocWorkFrame(std::unique_ptr<CUFrameBuf>& frame, const RGYFrameInfo& frameInfo, const TCHAR *label) {
    if (!frame
        || frame->frame.width != frameInfo.width
        || frame->frame.height != frameInfo.height
        || frame->frame.csp != frameInfo.csp) {
        frame = std::make_unique<CUFrameBuf>(frameInfo);
        frame->releasePtr();
        const auto sts = frame->alloc();
        if (sts != RGY_ERR_NONE) {
            AddMessage(RGY_LOG_ERROR, _T("failed to allocate dehalo %s buffer: %s.\n"), label, get_err_mes(sts));
            return RGY_ERR_MEMORY_ALLOC;
        }
    }
    return RGY_ERR_NONE;
}

RGY_ERR NVEncFilterDehalo::init(shared_ptr<NVEncFilterParam> pParam, shared_ptr<RGYLog> pPrintMes) {
    RGY_ERR sts = RGY_ERR_NONE;
    m_pLog = pPrintMes;
    auto prm = std::dynamic_pointer_cast<NVEncFilterParamDehalo>(pParam);
    if (!prm) {
        AddMessage(RGY_LOG_ERROR, _T("Invalid parameter type.\n"));
        return RGY_ERR_INVALID_PARAM;
    }
    sts = checkParam(prm);
    if (sts != RGY_ERR_NONE) {
        return sts;
    }

    prm->frameOut.picstruct = prm->frameIn.picstruct;
    sts = AllocFrameBuf(prm->frameOut, 1);
    if (sts != RGY_ERR_NONE) {
        AddMessage(RGY_LOG_ERROR, _T("failed to allocate memory: %s.\n"), get_err_mes(sts));
        return RGY_ERR_MEMORY_ALLOC;
    }
    for (int i = 0; i < RGY_CSP_PLANES[prm->frameOut.csp]; i++) {
        prm->frameOut.pitch[i] = m_frameBuf[0]->frame.pitch[i];
    }

    m_ssActive = prm->dehalo.ss > 1.0f + 1e-6f;
    if (m_ssActive) {
        m_ssW = ((int)std::lround(prm->frameIn.width  * prm->dehalo.ss) + 1) & ~1;
        m_ssH = ((int)std::lround(prm->frameIn.height * prm->dehalo.ss) + 1) & ~1;
    } else {
        m_ssW = prm->frameIn.width;
        m_ssH = prm->frameIn.height;
    }

    const auto bitDepth = RGY_CSP_BIT_DEPTH[prm->frameIn.csp];
    const auto lumaCsp = (bitDepth > 8) ? RGY_CSP_Y16 : RGY_CSP_Y8;
    auto lumaInfo = prm->frameIn;
    lumaInfo.csp = lumaCsp;

    auto initResize = [&](std::unique_ptr<NVEncFilterResize>& filter, const RGYFrameInfo& frameIn, const RGYFrameInfo& frameOut,
        RGY_VPP_RESIZE_ALGO interp, const TCHAR *label, const float bicubicB = FILTER_DEFAULT_RESIZE_BICUBIC_B, const float bicubicC = FILTER_DEFAULT_RESIZE_BICUBIC_C) {
        auto resizePrm = std::make_shared<NVEncFilterParamResize>();
        resizePrm->frameIn = frameIn;
        resizePrm->frameOut = frameOut;
        resizePrm->interp = interp;
        resizePrm->bicubic.b = bicubicB;
        resizePrm->bicubic.c = bicubicC;
        resizePrm->baseFps = prm->baseFps;
        resizePrm->bOutOverwrite = false;
        filter = std::make_unique<NVEncFilterResize>();
        const auto resizeSts = filter->init(resizePrm, m_pLog);
        if (resizeSts != RGY_ERR_NONE) {
            AddMessage(RGY_LOG_ERROR, _T("failed to init dehalo %s sub-filter: %s.\n"), label, get_err_mes(resizeSts));
        }
        return resizeSts;
    };

    if (prm->dehalo.mode == VPP_DEHALO_MODE_LEGACY) {
        RGYFrameInfo workInfo = prm->frameIn;
        workInfo.width = m_ssW;
        workInfo.height = m_ssH;
        sts = allocWorkFrame(m_expanded, workInfo, _T("expanded"));
        if (sts != RGY_ERR_NONE) return sts;
        sts = allocWorkFrame(m_inpand, workInfo, _T("inpand"));
        if (sts != RGY_ERR_NONE) return sts;
        sts = allocWorkFrame(m_mask, workInfo, _T("mask"));
        if (sts != RGY_ERR_NONE) return sts;

        if (m_ssActive) {
            sts = allocWorkFrame(m_supersampled, workInfo, _T("supersampled"));
            if (sts != RGY_ERR_NONE) return sts;
            sts = allocWorkFrame(m_corrected, workInfo, _T("corrected"));
            if (sts != RGY_ERR_NONE) return sts;

            auto prmUpIn = lumaInfo;
            auto prmUpOut = lumaInfo;
            prmUpOut.width = m_ssW;
            prmUpOut.height = m_ssH;
            sts = initResize(m_resizeUp, prmUpIn, prmUpOut, RGY_VPP_RESIZE_SPLINE36, _T("upscale"));
            if (sts != RGY_ERR_NONE) return sts;

            auto prmDownIn = prmUpOut;
            auto prmDownOut = lumaInfo;
            sts = initResize(m_resizeDown, prmDownIn, prmDownOut, RGY_VPP_RESIZE_SPLINE36, _T("downscale"));
            if (sts != RGY_ERR_NONE) return sts;
        } else {
            m_supersampled.reset();
            m_corrected.reset();
            m_resizeUp.reset();
            m_resizeDown.reset();
        }
    } else {
        m_alphaHaloW = std::max(4, (int)std::lround((double)prm->frameIn.width / (double)prm->dehalo.rx));
        m_alphaHaloH = std::max(4, (int)std::lround((double)prm->frameIn.height / (double)prm->dehalo.ry));

        RGYFrameInfo fullInfo = lumaInfo;
        RGYFrameInfo smallInfo = lumaInfo;
        smallInfo.width = m_alphaHaloW;
        smallInfo.height = m_alphaHaloH;
        RGYFrameInfo ssInfo = lumaInfo;
        ssInfo.width = m_ssW;
        ssInfo.height = m_ssH;

        sts = allocWorkFrame(m_alphaHalosSmall, smallInfo, _T("alpha halos small"));
        if (sts != RGY_ERR_NONE) return sts;
        sts = allocWorkFrame(m_alphaHalos, fullInfo, _T("alpha halos"));
        if (sts != RGY_ERR_NONE) return sts;
        sts = allocWorkFrame(m_alphaAre, fullInfo, _T("alpha are"));
        if (sts != RGY_ERR_NONE) return sts;
        sts = allocWorkFrame(m_alphaUgly, fullInfo, _T("alpha ugly"));
        if (sts != RGY_ERR_NONE) return sts;
        sts = allocWorkFrame(m_alphaLets, fullInfo, _T("alpha lets"));
        if (sts != RGY_ERR_NONE) return sts;
        sts = allocWorkFrame(m_alphaLimitLow, fullInfo, _T("alpha limit low"));
        if (sts != RGY_ERR_NONE) return sts;
        sts = allocWorkFrame(m_alphaLimitHigh, fullInfo, _T("alpha limit high"));
        if (sts != RGY_ERR_NONE) return sts;
        sts = allocWorkFrame(m_alphaRemoved, fullInfo, _T("alpha removed"));
        if (sts != RGY_ERR_NONE) return sts;

        sts = initResize(m_resizeAlphaHaloDown, fullInfo, smallInfo, RGY_VPP_RESIZE_BICUBIC, _T("alpha halo downscale"), 1.0f / 3.0f, 1.0f / 3.0f);
        if (sts != RGY_ERR_NONE) return sts;
        sts = initResize(m_resizeAlphaHaloUp, smallInfo, fullInfo, RGY_VPP_RESIZE_BICUBIC, _T("alpha halo upscale"), 1.0f, 0.0f);
        if (sts != RGY_ERR_NONE) return sts;

        if (m_ssActive) {
            sts = allocWorkFrame(m_supersampled, ssInfo, _T("alpha supersampled"));
            if (sts != RGY_ERR_NONE) return sts;
            sts = allocWorkFrame(m_corrected, ssInfo, _T("alpha corrected"));
            if (sts != RGY_ERR_NONE) return sts;
            sts = allocWorkFrame(m_alphaLimitLowSS, ssInfo, _T("alpha limit low ss"));
            if (sts != RGY_ERR_NONE) return sts;
            sts = allocWorkFrame(m_alphaLimitHighSS, ssInfo, _T("alpha limit high ss"));
            if (sts != RGY_ERR_NONE) return sts;
            sts = initResize(m_resizeAlphaUp, fullInfo, ssInfo, RGY_VPP_RESIZE_LANCZOS3, _T("alpha upscale"));
            if (sts != RGY_ERR_NONE) return sts;
            sts = initResize(m_resizeAlphaDown, ssInfo, fullInfo, RGY_VPP_RESIZE_LANCZOS3, _T("alpha downscale"));
            if (sts != RGY_ERR_NONE) return sts;
        } else {
            m_supersampled.reset();
            m_corrected.reset();
            m_alphaLimitLowSS.reset();
            m_alphaLimitHighSS.reset();
            m_resizeAlphaUp.reset();
            m_resizeAlphaDown.reset();
        }
    }

    setFilterInfo(prm->print());
    m_param = prm;
    return sts;
}

tstring NVEncFilterParamDehalo::print() const {
    return dehalo.print();
}

RGY_ERR NVEncFilterDehalo::run_filter(const RGYFrameInfo *pInputFrame, RGYFrameInfo **ppOutputFrames, int *pOutputFrameNum, cudaStream_t stream) {
    RGY_ERR sts = RGY_ERR_NONE;
    if (pInputFrame->ptr[0] == nullptr) {
        return sts;
    }

    *pOutputFrameNum = 1;
    if (ppOutputFrames[0] == nullptr) {
        auto pOutFrame = m_frameBuf[m_nFrameIdx].get();
        ppOutputFrames[0] = &pOutFrame->frame;
        m_nFrameIdx = (m_nFrameIdx + 1) % m_frameBuf.size();
    }
    ppOutputFrames[0]->picstruct = pInputFrame->picstruct;
    if (interlaced(*pInputFrame)) {
        return filter_as_interlaced_pair(pInputFrame, ppOutputFrames[0], stream);
    }
    const auto memcpyKind = getCudaMemcpyKind(pInputFrame->mem_type, ppOutputFrames[0]->mem_type);
    if (memcpyKind != cudaMemcpyDeviceToDevice) {
        AddMessage(RGY_LOG_ERROR, _T("only supported on device memory.\n"));
        return RGY_ERR_INVALID_PARAM;
    }

    auto prm = std::dynamic_pointer_cast<NVEncFilterParamDehalo>(m_param);
    if (!prm) {
        AddMessage(RGY_LOG_ERROR, _T("Invalid parameter type.\n"));
        return RGY_ERR_INVALID_PARAM;
    }

    const int bitDepth = RGY_CSP_BIT_DEPTH[pInputFrame->csp];
    const int maxVal = (1 << bitDepth) - 1;
    const int loScaled = (int)((long long)prm->dehalo.lowsens  * maxVal / 100);
    const int hiScaled = (int)((long long)prm->dehalo.highsens * maxVal / 100);
    const int loScaledAlpha = (int)((long long)prm->dehalo.lowsens * maxVal / 255);

    if (prm->dehalo.mode == VPP_DEHALO_MODE_LEGACY) {
        const RGYFrameInfo *pMorphSrc = pInputFrame;
        if (m_ssActive) {
            int resizeOutNum = 0;
            const auto lumaCsp = (RGY_CSP_BIT_DEPTH[pInputFrame->csp] > 8) ? RGY_CSP_Y16 : RGY_CSP_Y8;
            auto inputLuma = getPlane(pInputFrame, RGY_PLANE_Y);
            auto outputLuma = getPlane(&m_supersampled->frame, RGY_PLANE_Y);
            inputLuma.csp = lumaCsp;
            outputLuma.csp = lumaCsp;
            RGYFrameInfo *resizeOut[1] = { &outputLuma };
            sts = m_resizeUp->filter(&inputLuma, resizeOut, &resizeOutNum, stream);
            if (sts != RGY_ERR_NONE || resizeOutNum != 1) {
                AddMessage(RGY_LOG_ERROR, _T("dehalo resize-up failed: %s.\n"), get_err_mes(sts));
                return sts;
            }
            pMorphSrc = &m_supersampled->frame;
        }

        RGYFrameInfo *pApplyDst = m_ssActive ? &m_corrected->frame : ppOutputFrames[0];
        sts = dehalo_process_y(pApplyDst, pMorphSrc, &m_expanded->frame, &m_inpand->frame, &m_mask->frame,
            prm->dehalo, loScaled, hiScaled, stream);
        if (sts != RGY_ERR_NONE) {
            AddMessage(RGY_LOG_ERROR, _T("dehalo kernel failed: %s.\n"), get_err_mes(sts));
            return sts;
        }

        if (m_ssActive) {
            int resizeOutNum = 0;
            const auto lumaCsp = (RGY_CSP_BIT_DEPTH[pInputFrame->csp] > 8) ? RGY_CSP_Y16 : RGY_CSP_Y8;
            auto correctedLuma = getPlane(&m_corrected->frame, RGY_PLANE_Y);
            auto outputLuma = getPlane(ppOutputFrames[0], RGY_PLANE_Y);
            correctedLuma.csp = lumaCsp;
            outputLuma.csp = lumaCsp;
            RGYFrameInfo *resizeOut[1] = { &outputLuma };
            sts = m_resizeDown->filter(&correctedLuma, resizeOut, &resizeOutNum, stream);
            if (sts != RGY_ERR_NONE || resizeOutNum != 1) {
                AddMessage(RGY_LOG_ERROR, _T("dehalo resize-down failed: %s.\n"), get_err_mes(sts));
                return sts;
            }
        }
    } else {
        const auto lumaCsp = (RGY_CSP_BIT_DEPTH[pInputFrame->csp] > 8) ? RGY_CSP_Y16 : RGY_CSP_Y8;
        auto inputLuma = getPlane(pInputFrame, RGY_PLANE_Y);
        auto outputLuma = getPlane(ppOutputFrames[0], RGY_PLANE_Y);
        inputLuma.csp = lumaCsp;
        outputLuma.csp = lumaCsp;

        auto runResize = [&](NVEncFilterResize *filter, RGYFrameInfo *pIn, RGYFrameInfo *pOut, const TCHAR *label) {
            int resizeOutNum = 0;
            RGYFrameInfo *resizeOut[1] = { pOut };
            const auto resizeSts = filter->filter(pIn, resizeOut, &resizeOutNum, stream);
            if (resizeSts != RGY_ERR_NONE || resizeOutNum != 1) {
                AddMessage(RGY_LOG_ERROR, _T("dehalo %s failed: %s.\n"), label, get_err_mes(resizeSts));
                return (resizeSts != RGY_ERR_NONE) ? resizeSts : RGY_ERR_UNKNOWN;
            }
            return RGY_ERR_NONE;
        };

        sts = runResize(m_resizeAlphaHaloDown.get(), &inputLuma, &m_alphaHalosSmall->frame, _T("alpha halo downscale"));
        if (sts != RGY_ERR_NONE) return sts;
        sts = runResize(m_resizeAlphaHaloUp.get(), &m_alphaHalosSmall->frame, &m_alphaHalos->frame, _T("alpha halo upscale"));
        if (sts != RGY_ERR_NONE) return sts;

        const auto searchRadius = dehalo_alpha_search_radius(prm->dehalo);
        sts = dehalo_alpha_range_y(&m_alphaAre->frame, &inputLuma, searchRadius.first, searchRadius.second, stream);
        if (sts != RGY_ERR_NONE) return sts;
        sts = dehalo_alpha_range_y(&m_alphaUgly->frame, &m_alphaHalos->frame, searchRadius.first, searchRadius.second, stream);
        if (sts != RGY_ERR_NONE) return sts;
        sts = dehalo_alpha_lets_y(&m_alphaLets->frame, &inputLuma, &m_alphaHalos->frame, &m_alphaAre->frame, &m_alphaUgly->frame,
            loScaledAlpha, prm->dehalo.highsens, stream);
        if (sts != RGY_ERR_NONE) return sts;
        sts = dehalo_alpha_morph_y(&m_alphaLimitLow->frame, &m_alphaLets->frame, 1, false, stream);
        if (sts != RGY_ERR_NONE) return sts;
        sts = dehalo_alpha_morph_y(&m_alphaLimitHigh->frame, &m_alphaLets->frame, 1, true, stream);
        if (sts != RGY_ERR_NONE) return sts;

        if (m_ssActive) {
            sts = runResize(m_resizeAlphaUp.get(), &inputLuma, &m_supersampled->frame, _T("alpha upscale"));
            if (sts != RGY_ERR_NONE) return sts;
            sts = runResize(m_resizeAlphaUp.get(), &m_alphaLimitLow->frame, &m_alphaLimitLowSS->frame, _T("alpha limit-low upscale"));
            if (sts != RGY_ERR_NONE) return sts;
            sts = runResize(m_resizeAlphaUp.get(), &m_alphaLimitHigh->frame, &m_alphaLimitHighSS->frame, _T("alpha limit-high upscale"));
            if (sts != RGY_ERR_NONE) return sts;
            sts = dehalo_alpha_clamp_y(&m_corrected->frame, &m_supersampled->frame, &m_alphaLimitLowSS->frame, &m_alphaLimitHighSS->frame, stream);
            if (sts != RGY_ERR_NONE) return sts;
            sts = runResize(m_resizeAlphaDown.get(), &m_corrected->frame, &m_alphaRemoved->frame, _T("alpha downscale"));
            if (sts != RGY_ERR_NONE) return sts;
        } else {
            sts = dehalo_alpha_clamp_y(&m_alphaRemoved->frame, &inputLuma, &m_alphaLimitLow->frame, &m_alphaLimitHigh->frame, stream);
            if (sts != RGY_ERR_NONE) return sts;
        }
        sts = dehalo_alpha_them_y(&outputLuma, &inputLuma, &m_alphaRemoved->frame, prm->dehalo, stream);
        if (sts != RGY_ERR_NONE) return sts;
    }

    const int copyPlanes = std::min<int>(RGY_CSP_PLANES[pInputFrame->csp], RGY_CSP_PLANES[rgy_csp_no_alpha(pInputFrame->csp)]);
    for (int iplane = 1; iplane < copyPlanes; iplane++) {
        const auto planeInput = getPlane(pInputFrame, (RGY_PLANE)iplane);
        auto planeOutput = getPlane(ppOutputFrames[0], (RGY_PLANE)iplane);
        sts = copyPlaneAsync(&planeOutput, &planeInput, stream);
        if (sts != RGY_ERR_NONE) {
            return sts;
        }
    }
    sts = copyPlaneAlphaAsync(ppOutputFrames[0], pInputFrame, stream);
    if (sts != RGY_ERR_NONE) {
        return sts;
    }
    auto cudaerr = cudaGetLastError();
    if (cudaerr != cudaSuccess) {
        return err_to_rgy(cudaerr);
    }
    return RGY_ERR_NONE;
}

void NVEncFilterDehalo::close() {
    m_resizeUp.reset();
    m_resizeDown.reset();
    m_resizeAlphaHaloDown.reset();
    m_resizeAlphaHaloUp.reset();
    m_resizeAlphaUp.reset();
    m_resizeAlphaDown.reset();
    m_supersampled.reset();
    m_expanded.reset();
    m_inpand.reset();
    m_mask.reset();
    m_corrected.reset();
    m_alphaHalosSmall.reset();
    m_alphaHalos.reset();
    m_alphaAre.reset();
    m_alphaUgly.reset();
    m_alphaLets.reset();
    m_alphaLimitLow.reset();
    m_alphaLimitHigh.reset();
    m_alphaLimitLowSS.reset();
    m_alphaLimitHighSS.reset();
    m_alphaRemoved.reset();
    m_alphaHaloW = 0;
    m_alphaHaloH = 0;
    m_ssW = 0;
    m_ssH = 0;
    m_ssActive = false;
    m_frameBuf.clear();
}
