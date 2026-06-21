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
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, ...
// ------------------------------------------------------------------------------------------
//
// Algorithm: Anime4K v3.2 by bloc97 (MIT, 2019). CUDA port of the OpenCL kaizen
// filter; the base sobel/refine/apply chain mirrors rgy_filter_kaizen.cl. No
// shader source copied verbatim.

#include <map>
#include <algorithm>
#define _USE_MATH_DEFINES
#include <cmath>
#include "convert_csp.h"
#include "NVEncFilter.h"
#include "NVEncFilterKaizen.h"
#include "rgy_filesystem.h"
#include "rgy_aspect_ratio.h"

static const int KAIZEN_BLOCK_X = 32;
static const int KAIZEN_BLOCK_Y = 8;

#define KAIZEN_DVAL_THRESHOLD 0.1f
#define RGY_FLT_EPS (1e-6f)
#define KAIZEN_PI_F 3.14159265358979323846f

// Polynomial-fit coefficients (Anime4K_Upscale_Original_x2.glsl v3.2).
__device__ __forceinline__ float kaizen_poly5(float x) {
    const float P5 = 11.68129591f, P4 = -42.46906057f, P3 = 60.28286266f;
    const float P2 = -41.84451327f, P1 = 14.05517353f, P0 = -1.081521930f;
    float x2 = x * x, x3 = x2 * x, x4 = x2 * x2, x5 = x2 * x3;
    return P5 * x5 + P4 * x4 + P3 * x3 + P2 * x2 + P1 * x + P0;
}

// Source texture: normalized coords + bilinear + clamp + readModeNormalizedFloat,
// matching the OpenCL kaizen_src_sampler (the hardware bilinear gives the free 2x).
template<typename Type>
static cudaError_t setKaizenSrcTex(cudaTextureObject_t &tex, const RGYFrameInfo *pPlane) {
    tex = 0;
    cudaResourceDesc r; memset(&r, 0, sizeof(r));
    r.resType = cudaResourceTypePitch2D;
    r.res.pitch2D.desc = cudaCreateChannelDesc<Type>();
    r.res.pitch2D.pitchInBytes = pPlane->pitch[0];
    r.res.pitch2D.width = pPlane->width;
    r.res.pitch2D.height = pPlane->height;
    r.res.pitch2D.devPtr = (uint8_t *)pPlane->ptr[0];
    cudaTextureDesc t; memset(&t, 0, sizeof(t));
    t.addressMode[0] = cudaAddressModeClamp;
    t.addressMode[1] = cudaAddressModeClamp;
    t.filterMode = cudaFilterModeLinear;
    t.readMode = cudaReadModeNormalizedFloat;
    t.normalizedCoords = 1;
    return cudaCreateTextureObject(&tex, &r, &t, nullptr);
}

// ---- base Anime4K chain (float4 scratch, pitchFloats == outW) ----

__global__ void kaizen_sobel_x(float4 *__restrict__ pDstA, const int dstPitchFloats,
    cudaTextureObject_t tex, const int outW, const int outH) {
    const int ix = blockIdx.x * blockDim.x + threadIdx.x;
    const int iy = blockIdx.y * blockDim.y + threadIdx.y;
    if (ix >= outW || iy >= outH) return;
    const float dx = 1.0f / (float)outW;
    const float px = ((float)ix + 0.5f) / (float)outW;
    const float py = ((float)iy + 0.5f) / (float)outH;
    const float l = tex2D<float>(tex, px - dx, py);
    const float c = tex2D<float>(tex, px, py);
    const float r = tex2D<float>(tex, px + dx, py);
    pDstA[iy * dstPitchFloats + ix] = make_float4(-l + r, l + c + c + r, 0.0f, 0.0f);
}

__global__ void kaizen_sobel_y(float4 *__restrict__ pDstB, const int dstPitchFloats,
    const float4 *__restrict__ pSrcA, const int srcPitchFloats, const int outW, const int outH, const float strength) {
    const int ix = blockIdx.x * blockDim.x + threadIdx.x;
    const int iy = blockIdx.y * blockDim.y + threadIdx.y;
    if (ix >= outW || iy >= outH) return;
    const int iy_t = max(iy - 1, 0);
    const int iy_b = min(iy + 1, outH - 1);
    const float4 t = pSrcA[iy_t * srcPitchFloats + ix];
    const float4 c = pSrcA[iy   * srcPitchFloats + ix];
    const float4 b = pSrcA[iy_b * srcPitchFloats + ix];
    const float xgrad = t.x + c.x + c.x + b.x;
    const float ygrad = -t.y + b.y;
    const float sobel_norm = clamp(sqrtf(xgrad * xgrad + ygrad * ygrad), 0.0f, 1.0f);
    const float dval = clamp(kaizen_poly5(sobel_norm) * strength, 0.0f, 1.0f);
    pDstB[iy * dstPitchFloats + ix] = make_float4(sobel_norm, dval, 0.0f, 0.0f);
}

__global__ void kaizen_refine_x(float4 *__restrict__ pDstA, const int dstPitchFloats,
    const float4 *__restrict__ pSrcB, const int srcPitchFloats, const int outW, const int outH) {
    const int ix = blockIdx.x * blockDim.x + threadIdx.x;
    const int iy = blockIdx.y * blockDim.y + threadIdx.y;
    if (ix >= outW || iy >= outH) return;
    const float4 cval = pSrcB[iy * srcPitchFloats + ix];
    const float dval = cval.y;
    if (dval < KAIZEN_DVAL_THRESHOLD) {
        pDstA[iy * dstPitchFloats + ix] = make_float4(0.0f, 0.0f, dval, 0.0f);
        return;
    }
    const int ix_l = max(ix - 1, 0);
    const int ix_r = min(ix + 1, outW - 1);
    const float l = pSrcB[iy * srcPitchFloats + ix_l].x;
    const float c = cval.x;
    const float r = pSrcB[iy * srcPitchFloats + ix_r].x;
    pDstA[iy * dstPitchFloats + ix] = make_float4(-l + r, l + c + c + r, dval, 0.0f);
}

__global__ void kaizen_refine_y(float4 *__restrict__ pDstB, const int dstPitchFloats,
    const float4 *__restrict__ pSrcA, const int srcPitchFloats, const int outW, const int outH) {
    const int ix = blockIdx.x * blockDim.x + threadIdx.x;
    const int iy = blockIdx.y * blockDim.y + threadIdx.y;
    if (ix >= outW || iy >= outH) return;
    const float4 cval = pSrcA[iy * srcPitchFloats + ix];
    const float dval = cval.z;
    if (dval < KAIZEN_DVAL_THRESHOLD) {
        pDstB[iy * dstPitchFloats + ix] = make_float4(0.0f, 0.0f, dval, 0.0f);
        return;
    }
    const int iy_t = max(iy - 1, 0);
    const int iy_b = min(iy + 1, outH - 1);
    const float4 t = pSrcA[iy_t * srcPitchFloats + ix];
    const float4 b = pSrcA[iy_b * srcPitchFloats + ix];
    const float xgrad = t.x + cval.x + cval.x + b.x;
    const float ygrad = -t.y + b.y;
    const float norm = sqrtf(xgrad * xgrad + ygrad * ygrad);
    float ndx = 0.0f, ndy = 0.0f;
    if (norm > 0.001f) { ndx = xgrad / norm; ndy = ygrad / norm; }
    pDstB[iy * dstPitchFloats + ix] = make_float4(ndx, ndy, dval, 0.0f);
}

template<typename Type, int bit_depth>
__global__ void kaizen_apply(uint8_t *__restrict__ pDstY, const int dstPitch,
    cudaTextureObject_t tex, const float4 *__restrict__ pSrcB, const int srcPitchFloats, const int outW, const int outH) {
    const int ix = blockIdx.x * blockDim.x + threadIdx.x;
    const int iy = blockIdx.y * blockDim.y + threadIdx.y;
    if (ix >= outW || iy >= outH) return;
    const float dx = 1.0f / (float)outW;
    const float dy = 1.0f / (float)outH;
    const float px = ((float)ix + 0.5f) / (float)outW;
    const float py = ((float)iy + 0.5f) / (float)outH;
    const float center = tex2D<float>(tex, px, py);
    const float4 dc = pSrcB[iy * srcPitchFloats + ix];
    const float dval = dc.z;
    float result;
    if (dval < KAIZEN_DVAL_THRESHOLD || fabsf(dc.x + dc.y) <= 0.0001f) {
        result = center;
    } else {
        const float sx = (dc.x > 0.0f) - (dc.x < 0.0f);
        const float sy = (dc.y > 0.0f) - (dc.y < 0.0f);
        const float xval = tex2D<float>(tex, px - sx * dx, py);
        const float yval = tex2D<float>(tex, px, py - sy * dy);
        const float adx = fabsf(dc.x);
        const float ady = fabsf(dc.y);
        const float xyratio = adx / (adx + ady + RGY_FLT_EPS);
        const float avg = xyratio * xval + (1.0f - xyratio) * yval;
        result = avg * dval + center * (1.0f - dval);
    }
    result = clamp(result, 0.0f, 1.0f);
    Type *ptr = (Type *)(pDstY + iy * dstPitch + ix * sizeof(Type));
    ptr[0] = (Type)(result * (float)((1 << bit_depth) - 1) + 0.5f);
}

// ---- geometric chroma resize ----

__device__ __forceinline__ float kaizen_sinc(float x) {
    if (fabsf(x) < 1e-9f) return 1.0f;
    const float px = KAIZEN_PI_F * x;
    return sinf(px) / px;
}
__device__ __forceinline__ float kaizen_chroma_weight(int chromaMode, float x) {
    const float ax = fabsf(x);
    if (chromaMode == 1) {                       // bilinear
        return (ax < 1.0f) ? (1.0f - ax) : 0.0f;
    } else if (chromaMode == 2) {                // bicubic Mitchell-Netravali B=C=1/3
        const float B = 1.0f / 3.0f, C = 1.0f / 3.0f;
        const float x2 = ax * ax, x3 = x2 * ax;
        if (ax < 1.0f) return ((12.0f - 9.0f * B - 6.0f * C) * x3 + (-18.0f + 12.0f * B + 6.0f * C) * x2 + (6.0f - 2.0f * B)) / 6.0f;
        else if (ax < 2.0f) return ((-B - 6.0f * C) * x3 + (6.0f * B + 30.0f * C) * x2 + (-12.0f * B - 48.0f * C) * ax + (8.0f * B + 24.0f * C)) / 6.0f;
        return 0.0f;
    } else if (chromaMode == 3) {                // lanczos3
        return (ax < 3.0f) ? (kaizen_sinc(x) * kaizen_sinc(x / 3.0f)) : 0.0f;
    } else {                                     // spline36
        if (ax < 1.0f) return ((13.0f / 11.0f) * ax - (453.0f / 209.0f)) * ax * ax + (1.0f - 3.0f / 209.0f);
        else if (ax < 2.0f) return ((-(6.0f / 11.0f) * ax + (612.0f / 209.0f)) * ax + (-(1038.0f / 209.0f))) * ax + (540.0f / 209.0f);
        else if (ax < 3.0f) return (((1.0f / 11.0f) * ax + (-(159.0f / 209.0f))) * ax + (434.0f / 209.0f)) * ax + (-(384.0f / 209.0f));
        return 0.0f;
    }
}

// One chroma plane (elemStride = 1 planar / 2 nv12-interleaved).
template<typename Type, int bit_depth>
__global__ void kaizen_chroma_resize(uint8_t *__restrict__ pDstC, const int dstPitch, const int dstStride, const int dstW, const int dstH,
    const uint8_t *__restrict__ pSrcC, const int srcPitch, const int srcStride, const int srcW, const int srcH, const int chromaMode) {
    const int ix = blockIdx.x * blockDim.x + threadIdx.x;
    const int iy = blockIdx.y * blockDim.y + threadIdx.y;
    if (ix >= dstW || iy >= dstH) return;
    const float maxv = (float)((1 << bit_depth) - 1);
    const float sx = ((float)ix + 0.5f) * (float)srcW / (float)dstW - 0.5f;
    const float sy = ((float)iy + 0.5f) * (float)srcH / (float)dstH - 0.5f;
    const int sxi = (int)floorf(sx);
    const int syi = (int)floorf(sy);
    const int taps = (chromaMode == 1) ? 2 : ((chromaMode == 2) ? 4 : 6);
    const int kernel_half = taps / 2;
    const int t0 = 1 - kernel_half, t1 = kernel_half;
    float acc = 0.0f, wsum = 0.0f;
    for (int dy = t0; dy <= t1; ++dy) {
        const int sy_c = min(max(syi + dy, 0), srcH - 1);
        const float wy = kaizen_chroma_weight(chromaMode, (float)dy - (sy - (float)syi));
        for (int dx = t0; dx <= t1; ++dx) {
            const int sx_c = min(max(sxi + dx, 0), srcW - 1);
            const float wx = kaizen_chroma_weight(chromaMode, (float)dx - (sx - (float)sxi));
            const float w = wx * wy;
            const Type s = *(const Type *)(pSrcC + sy_c * srcPitch + sx_c * srcStride * (int)sizeof(Type));
            acc += ((float)s / maxv) * w;
            wsum += w;
        }
    }
    float out = (wsum > 1e-6f) ? (acc / wsum) : 0.0f;
    out = clamp(out, 0.0f, 1.0f);
    Type *ptr = (Type *)(pDstC + iy * dstPitch + ix * dstStride * (int)sizeof(Type));
    ptr[0] = (Type)(out * maxv + 0.5f);
}

// ---- joint-bilateral chroma (FastBilateral, MIT): luma-guided chroma upscale ----

template<typename Type, int bit_depth>
__device__ __forceinline__ float kaizen_rdnorm(const uint8_t *p, int pitch, int x, int y) {
    return (float)(*(const Type *)(p + y * pitch + x * (int)sizeof(Type))) / (float)((1 << bit_depth) - 1);
}
template<typename Type, int bit_depth>
__global__ void kaizen_chroma_luma_lowres(uint8_t *__restrict__ pLow, const int lowPitch, const int lowW, const int lowH,
    const uint8_t *__restrict__ pLuma, const int lumaPitch, const int lumaW, const int lumaH) {
    const int ix = blockIdx.x * blockDim.x + threadIdx.x;
    const int iy = blockIdx.y * blockDim.y + threadIdx.y;
    if (ix >= lowW || iy >= lowH) return;
    const int lx = min(ix * 2, lumaW - 1), ly = min(iy * 2, lumaH - 1);
    const int lx1 = min(lx + 1, lumaW - 1), ly1 = min(ly + 1, lumaH - 1);
    const float avg = 0.25f * (kaizen_rdnorm<Type, bit_depth>(pLuma, lumaPitch, lx, ly) + kaizen_rdnorm<Type, bit_depth>(pLuma, lumaPitch, lx1, ly)
        + kaizen_rdnorm<Type, bit_depth>(pLuma, lumaPitch, lx, ly1) + kaizen_rdnorm<Type, bit_depth>(pLuma, lumaPitch, lx1, ly1));
    Type *ptr = (Type *)(pLow + iy * lowPitch + ix * sizeof(Type));
    ptr[0] = (Type)(avg * (float)((1 << bit_depth) - 1) + 0.5f);
}
template<typename Type, int bit_depth>
__global__ void kaizen_chroma_joint_bilateral(uint8_t *__restrict__ pDstC, const int dstPitch, const int dstW, const int dstH,
    const uint8_t *__restrict__ pSrcC, const int srcPitch, const int srcW, const int srcH,
    const uint8_t *__restrict__ pLuma, const int lumaPitch, const uint8_t *__restrict__ pLow, const int lowPitch,
    const float dist_coeff, const float int_coeff) {
    const int ix = blockIdx.x * blockDim.x + threadIdx.x;
    const int iy = blockIdx.y * blockDim.y + threadIdx.y;
    if (ix >= dstW || iy >= dstH) return;
    const float luma_zero = kaizen_rdnorm<Type, bit_depth>(pLuma, lumaPitch, ix, iy);
    const float px = ((float)ix + 0.5f) * (float)srcW / (float)dstW - 0.5f;
    const float py = ((float)iy + 0.5f) * (float)srcH / (float)dstH - 0.5f;
    const int fx = (int)floorf(px), fy = (int)floorf(py);
    const float frx = px - (float)fx, fry = py - (float)fy;
    float wt = 0.0f, ct = 0.0f;
    for (int j = 0; j < 2; ++j) {
        for (int i = 0; i < 2; ++i) {
            const int cx = min(max(fx + i, 0), srcW - 1), cy = min(max(fy + j, 0), srcH - 1);
            const float chroma = kaizen_rdnorm<Type, bit_depth>(pSrcC, srcPitch, cx, cy);
            const float lowl = kaizen_rdnorm<Type, bit_depth>(pLow, lowPitch, cx, cy);
            const float sdx = (float)i - frx, sdy = (float)j - fry, idiff = luma_zero - lowl;
            const float w = fmaxf(100.0f * __expf(-dist_coeff * (sdx * sdx + sdy * sdy) - int_coeff * (idiff * idiff)), 1e-32f);
            wt += w; ct += w * chroma;
        }
    }
    const float outv = clamp(ct / wt, 0.0f, 1.0f);
    Type *ptr = (Type *)(pDstC + iy * dstPitch + ix * sizeof(Type));
    ptr[0] = (Type)(outv * (float)((1 << bit_depth) - 1) + 0.5f);
}

// ---- Y read helper + copy-to-scratch ----

template<typename Type, int bit_depth>
__device__ __forceinline__ float kaizen_read_y_norm(const uint8_t *pY, int pitch, int x, int y) {
    const Type s = *(const Type *)(pY + y * pitch + x * (int)sizeof(Type));
    return (float)s / (float)((1 << bit_depth) - 1);
}

template<typename Type, int bit_depth>
__global__ void kaizen_copy_y_to_scratch(float4 *__restrict__ pDst, const int dstPitchFloats,
    const uint8_t *__restrict__ pY, const int srcPitch, const int W, const int H) {
    const int ix = blockIdx.x * blockDim.x + threadIdx.x;
    const int iy = blockIdx.y * blockDim.y + threadIdx.y;
    if (ix >= W || iy >= H) return;
    pDst[iy * dstPitchFloats + ix] = make_float4(kaizen_read_y_norm<Type, bit_depth>(pY, srcPitch, ix, iy), 0.0f, 0.0f, 0.0f);
}

// ---- bilateral denoise (mean / median / mode) ----

__device__ __forceinline__ float kaizen_bilateral_weight(int dx, int dy, float vc, float vn, float sigma_s, float sigma_i) {
    const float ds = (float)(dx * dx + dy * dy);
    const float dr = (vn - vc) / sigma_i;
    return __expf(-0.5f * ds / (sigma_s * sigma_s)) * __expf(-0.5f * dr * dr);
}
__device__ __forceinline__ float kaizen_denoise_sigma_i(float vc, float curve, float isigma) {
    return fmaxf(powf(vc + 0.0001f, curve) * isigma, 1e-6f);
}

template<typename Type, int bit_depth>
__global__ void kaizen_denoise_mean(uint8_t *__restrict__ pDstY, const int dstPitch,
    const float4 *__restrict__ pSrcRef, const int srcPitchFloats, const int outW, const int outH,
    const float sigma_s, const float isigma, const float curve) {
    const int ix = blockIdx.x * blockDim.x + threadIdx.x;
    const int iy = blockIdx.y * blockDim.y + threadIdx.y;
    if (ix >= outW || iy >= outH) return;
    const float vc = pSrcRef[iy * srcPitchFloats + ix].x;
    const float sigma_i = kaizen_denoise_sigma_i(vc, curve, isigma);
    float sum = 0.0f, n = 0.0f;
    for (int dy = -2; dy <= 2; ++dy) {
        const int yy = min(max(iy + dy, 0), outH - 1);
        for (int dx = -2; dx <= 2; ++dx) {
            const int xx = min(max(ix + dx, 0), outW - 1);
            const float v = pSrcRef[yy * srcPitchFloats + xx].x;
            const float w = kaizen_bilateral_weight(dx, dy, vc, v, sigma_s, sigma_i);
            sum += w * v; n += w;
        }
    }
    float result = clamp((n > 1e-9f) ? (sum / n) : vc, 0.0f, 1.0f);
    Type *ptr = (Type *)(pDstY + iy * dstPitch + ix * sizeof(Type));
    ptr[0] = (Type)(result * (float)((1 << bit_depth) - 1) + 0.5f);
}

template<typename Type, int bit_depth>
__global__ void kaizen_denoise_median(uint8_t *__restrict__ pDstY, const int dstPitch,
    const float4 *__restrict__ pSrcRef, const int srcPitchFloats, const int outW, const int outH,
    const float sigma_s, const float isigma, const float curve, const float reg) {
    const int ix = blockIdx.x * blockDim.x + threadIdx.x;
    const int iy = blockIdx.y * blockDim.y + threadIdx.y;
    if (ix >= outW || iy >= outH) return;
    const float vc = pSrcRef[iy * srcPitchFloats + ix].x;
    const float sigma_i = kaizen_denoise_sigma_i(vc, curve, isigma);
    float vs[9], ws[9]; float total_w = 0.0f; int idx = 0;
    for (int dy = -1; dy <= 1; ++dy) {
        const int yy = min(max(iy + dy, 0), outH - 1);
        for (int dx = -1; dx <= 1; ++dx) {
            const int xx = min(max(ix + dx, 0), outW - 1);
            const float v = pSrcRef[yy * srcPitchFloats + xx].x;
            const float w = kaizen_bilateral_weight(dx, dy, vc, v, sigma_s, sigma_i);
            vs[idx] = v; ws[idx] = w; total_w += w; idx++;
        }
    }
    if (reg > 0.0f) {
        float ws_reg[9]; for (int i = 0; i < 9; ++i) ws_reg[i] = 0.0f;
        total_w = 0.0f; const float inv_reg = 1.0f / reg;
        for (int i = 0; i < 9; ++i) {
            ws_reg[i] += ws[i];
            for (int j = i + 1; j < 9; ++j) {
                const float d = (vs[j] - vs[i]) * inv_reg;
                const float g = __expf(-0.5f * d * d);
                ws_reg[j] += g * ws[i]; ws_reg[i] += g * ws[j];
            }
            total_w += ws_reg[i];
        }
        for (int i = 0; i < 9; ++i) ws[i] = ws_reg[i];
    }
    float median = vc;
    if (total_w > 1e-9f) {
        const float inv_tot = 1.0f / total_w;
        for (int i = 0; i < 9; ++i) {
            float w_above = 0.0f, w_below = 0.0f;
            for (int j = 0; j < 9; ++j) {
                if (vs[j] > vs[i]) w_above += ws[j];
                else if (vs[j] < vs[i]) w_below += ws[j];
            }
            if ((total_w - w_above) * inv_tot >= 0.5f && w_below * inv_tot <= 0.5f) { median = vs[i]; break; }
        }
    }
    float result = clamp(median, 0.0f, 1.0f);
    Type *ptr = (Type *)(pDstY + iy * dstPitch + ix * sizeof(Type));
    ptr[0] = (Type)(result * (float)((1 << bit_depth) - 1) + 0.5f);
}

template<typename Type, int bit_depth>
__global__ void kaizen_denoise_mode(uint8_t *__restrict__ pDstY, const int dstPitch,
    const float4 *__restrict__ pSrcRef, const int srcPitchFloats, const int outW, const int outH,
    const float sigma_s, const float isigma, const float curve, const float reg_in) {
    const int ix = blockIdx.x * blockDim.x + threadIdx.x;
    const int iy = blockIdx.y * blockDim.y + threadIdx.y;
    if (ix >= outW || iy >= outH) return;
    const float vc = pSrcRef[iy * srcPitchFloats + ix].x;
    const float sigma_i = kaizen_denoise_sigma_i(vc, curve, isigma);
    const float reg = fmaxf(reg_in, 1e-6f);
    const float inv_reg = 1.0f / reg;
    float vs[9], ws[9], ws_reg[9]; int idx = 0;
    for (int dy = -1; dy <= 1; ++dy) {
        const int yy = min(max(iy + dy, 0), outH - 1);
        for (int dx = -1; dx <= 1; ++dx) {
            const int xx = min(max(ix + dx, 0), outW - 1);
            const float v = pSrcRef[yy * srcPitchFloats + xx].x;
            vs[idx] = v; ws[idx] = kaizen_bilateral_weight(dx, dy, vc, v, sigma_s, sigma_i); ws_reg[idx] = 0.0f; idx++;
        }
    }
    for (int i = 0; i < 9; ++i) {
        ws_reg[i] += ws[i];
        for (int j = i + 1; j < 9; ++j) {
            const float d = (vs[j] - vs[i]) * inv_reg;
            const float g = __expf(-0.5f * d * d);
            ws_reg[j] += g * ws[i]; ws_reg[i] += g * ws[j];
        }
    }
    float best_v = vc, best_w = 0.0f;
    for (int i = 0; i < 9; ++i) { if (ws_reg[i] > best_w) { best_w = ws_reg[i]; best_v = vs[i]; } }
    float result = clamp(best_v, 0.0f, 1.0f);
    Type *ptr = (Type *)(pDstY + iy * dstPitch + ix * sizeof(Type));
    ptr[0] = (Type)(result * (float)((1 << bit_depth) - 1) + 0.5f);
}

// ---- clamp_highlights (separable 5x5 source-luma max dilation, then clamp high side) ----

template<typename Type, int bit_depth>
__global__ void kaizen_clamp_h_max_y(float *__restrict__ pDst, const int dstStride,
    const uint8_t *__restrict__ pSrcY, const int srcPitch, const int W, const int H) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= W || y >= H) return;
    float m = -1.0e30f;
    for (int i = -2; i <= 2; ++i) m = fmaxf(m, kaizen_read_y_norm<Type, bit_depth>(pSrcY, srcPitch, min(max(x + i, 0), W - 1), y));
    pDst[y * dstStride + x] = m;
}
__global__ void kaizen_clamp_v_max(float *__restrict__ pDst, const int dstStride,
    const float *__restrict__ pSrc, const int srcStride, const int W, const int H) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= W || y >= H) return;
    float m = -1.0e30f;
    for (int i = -2; i <= 2; ++i) m = fmaxf(m, pSrc[min(max(y + i, 0), H - 1) * srcStride + x]);
    pDst[y * dstStride + x] = m;
}
template<typename Type, int bit_depth>
__global__ void kaizen_clamp_apply_y(uint8_t *__restrict__ pYOut, const int dstPitch,
    const float *__restrict__ pStatsMax, const int statsStride, const int dstW, const int dstH, const int srcW, const int srcH) {
    const int dx = blockIdx.x * blockDim.x + threadIdx.x;
    const int dy = blockIdx.y * blockDim.y + threadIdx.y;
    if (dx >= dstW || dy >= dstH) return;
    const float sx_f = ((float)dx + 0.5f) * (float)srcW / (float)dstW - 0.5f;
    const float sy_f = ((float)dy + 0.5f) * (float)srcH / (float)dstH - 0.5f;
    const int x0 = min(max((int)floorf(sx_f), 0), srcW - 1);
    const int y0 = min(max((int)floorf(sy_f), 0), srcH - 1);
    const int x1 = min(x0 + 1, srcW - 1), y1 = min(y0 + 1, srcH - 1);
    const float fx = sx_f - (float)x0, fy = sy_f - (float)y0;
    const float statsMax = (1.0f - fx) * (1.0f - fy) * pStatsMax[y0 * statsStride + x0]
        + fx * (1.0f - fy) * pStatsMax[y0 * statsStride + x1]
        + (1.0f - fx) * fy * pStatsMax[y1 * statsStride + x0]
        + fx * fy * pStatsMax[y1 * statsStride + x1];
    const float curY = kaizen_read_y_norm<Type, bit_depth>(pYOut, dstPitch, dx, dy);
    if (curY > statsMax) {
        Type *ptr = (Type *)(pYOut + dy * dstPitch + dx * sizeof(Type));
        ptr[0] = (Type)(statsMax * (float)((1 << bit_depth) - 1) + 0.5f);
    }
}

// ---- antiring (PixelClipper: clamp output luma to 2x2 source min/max, mix by strength) ----

template<typename Type, int bit_depth>
__global__ void kaizen_antiring_y(uint8_t *__restrict__ pYOut, const int dstPitch,
    const uint8_t *__restrict__ pSrcY, const int srcPitch, const int dstW, const int dstH, const int srcW, const int srcH, const float strength) {
    const int dx = blockIdx.x * blockDim.x + threadIdx.x;
    const int dy = blockIdx.y * blockDim.y + threadIdx.y;
    if (dx >= dstW || dy >= dstH) return;
    const float sx_f = ((float)dx + 0.5f) * (float)srcW / (float)dstW - 0.5f;
    const float sy_f = ((float)dy + 0.5f) * (float)srcH / (float)dstH - 0.5f;
    const int fx = (int)floorf(sx_f), fy = (int)floorf(sy_f);
    const int x0 = min(max(fx, 0), srcW - 1), x1 = min(max(fx + 1, 0), srcW - 1);
    const int y0 = min(max(fy, 0), srcH - 1), y1 = min(max(fy + 1, 0), srcH - 1);
    const float a = kaizen_read_y_norm<Type, bit_depth>(pSrcY, srcPitch, x0, y0);
    const float b = kaizen_read_y_norm<Type, bit_depth>(pSrcY, srcPitch, x1, y0);
    const float c = kaizen_read_y_norm<Type, bit_depth>(pSrcY, srcPitch, x0, y1);
    const float d = kaizen_read_y_norm<Type, bit_depth>(pSrcY, srcPitch, x1, y1);
    const float lo = fminf(fminf(a, b), fminf(c, d));
    const float hi = fmaxf(fmaxf(a, b), fmaxf(c, d));
    const float cur = kaizen_read_y_norm<Type, bit_depth>(pYOut, dstPitch, dx, dy);
    const float outv = cur + (clamp(cur, lo, hi) - cur) * strength;
    Type *ptr = (Type *)(pYOut + dy * dstPitch + dx * sizeof(Type));
    ptr[0] = (Type)(clamp(outv, 0.0f, 1.0f) * (float)((1 << bit_depth) - 1) + 0.5f);
}

// ---- darken_hq (DoG dark-half line darkening; STRENGTH=1.5) ----

#define KAIZEN_DARKEN_STRENGTH 1.5f

__device__ __forceinline__ float kaizen_gauss_w(float d, float sigma) {
    const float s = d / sigma;
    return __expf(-0.5f * s * s);
}

template<typename Type, int bit_depth>
__global__ void kaizen_darken_gauss1_x(float4 *__restrict__ pDstA, const int dstPitchFloats,
    const uint8_t *__restrict__ pSrcY, const int srcPitch, const int outW, const int outH, const float sigma, const int r) {
    const int ix = blockIdx.x * blockDim.x + threadIdx.x;
    const int iy = blockIdx.y * blockDim.y + threadIdx.y;
    if (ix >= outW || iy >= outH) return;
    float acc = 0.0f, wsum = 0.0f;
    for (int dx = -r; dx <= r; ++dx) {
        const int xx = min(max(ix + dx, 0), outW - 1);
        const float w = kaizen_gauss_w((float)dx, sigma);
        acc += kaizen_read_y_norm<Type, bit_depth>(pSrcY, srcPitch, xx, iy) * w; wsum += w;
    }
    pDstA[iy * dstPitchFloats + ix] = make_float4(acc / wsum, 0.0f, 0.0f, 0.0f);
}
template<typename Type, int bit_depth>
__global__ void kaizen_darken_dog_y(float4 *__restrict__ pDstB, const int dstPitchFloats,
    const float4 *__restrict__ pSrcA, const int srcPitchFloats, const uint8_t *__restrict__ pSrcY, const int srcPitch,
    const int outW, const int outH, const float sigma, const int r) {
    const int ix = blockIdx.x * blockDim.x + threadIdx.x;
    const int iy = blockIdx.y * blockDim.y + threadIdx.y;
    if (ix >= outW || iy >= outH) return;
    float acc = 0.0f, wsum = 0.0f;
    for (int dy = -r; dy <= r; ++dy) {
        const int yy = min(max(iy + dy, 0), outH - 1);
        const float w = kaizen_gauss_w((float)dy, sigma);
        acc += pSrcA[yy * srcPitchFloats + ix].x * w; wsum += w;
    }
    const float blur = acc / wsum;
    const float luma = kaizen_read_y_norm<Type, bit_depth>(pSrcY, srcPitch, ix, iy);
    pDstB[iy * dstPitchFloats + ix] = make_float4(fminf(luma - blur, 0.0f), 0.0f, 0.0f, 0.0f);
}
__global__ void kaizen_darken_gauss2_x(float4 *__restrict__ pDstA, const int dstPitchFloats,
    const float4 *__restrict__ pSrcB, const int srcPitchFloats, const int outW, const int outH, const float sigma, const int r) {
    const int ix = blockIdx.x * blockDim.x + threadIdx.x;
    const int iy = blockIdx.y * blockDim.y + threadIdx.y;
    if (ix >= outW || iy >= outH) return;
    float acc = 0.0f, wsum = 0.0f;
    for (int dx = -r; dx <= r; ++dx) {
        const int xx = min(max(ix + dx, 0), outW - 1);
        const float w = kaizen_gauss_w((float)dx, sigma);
        acc += pSrcB[iy * srcPitchFloats + xx].x * w; wsum += w;
    }
    pDstA[iy * dstPitchFloats + ix] = make_float4(acc / wsum, 0.0f, 0.0f, 0.0f);
}
template<typename Type, int bit_depth>
__global__ void kaizen_darken_apply_y(uint8_t *__restrict__ pDstY, const int dstPitch,
    const float4 *__restrict__ pSrcA, const int srcPitchFloats, const int outW, const int outH, const float sigma, const int r, const float strength) {
    const int ix = blockIdx.x * blockDim.x + threadIdx.x;
    const int iy = blockIdx.y * blockDim.y + threadIdx.y;
    if (ix >= outW || iy >= outH) return;
    float acc = 0.0f, wsum = 0.0f;
    for (int dy = -r; dy <= r; ++dy) {
        const int yy = min(max(iy + dy, 0), outH - 1);
        const float w = kaizen_gauss_w((float)dy, sigma);
        acc += pSrcA[yy * srcPitchFloats + ix].x * w; wsum += w;
    }
    const float smoothed = acc / wsum;
    const float luma = kaizen_read_y_norm<Type, bit_depth>(pDstY, dstPitch, ix, iy);
    float result = clamp(luma + smoothed * strength, 0.0f, 1.0f);
    Type *ptr = (Type *)(pDstY + iy * dstPitch + ix * sizeof(Type));
    ptr[0] = (Type)(result * (float)((1 << bit_depth) - 1) + 0.5f);
}

// ---- thin_hq (Sobel flow-field warp toward edges; STRENGTH=0.6) ----

#define KAIZEN_THIN_STRENGTH 0.6f

__device__ __forceinline__ float kaizen_bilinear_x(const float4 *buf, int pitchFloats, int w, int h, float fx, float fy) {
    const int x0 = min(max((int)floorf(fx), 0), w - 1);
    const int y0 = min(max((int)floorf(fy), 0), h - 1);
    const int x1 = min(x0 + 1, w - 1), y1 = min(y0 + 1, h - 1);
    const float dx = fx - (float)x0, dy = fy - (float)y0;
    const float v00 = buf[y0 * pitchFloats + x0].x, v01 = buf[y0 * pitchFloats + x1].x;
    const float v10 = buf[y1 * pitchFloats + x0].x, v11 = buf[y1 * pitchFloats + x1].x;
    return (1.0f - dx) * (1.0f - dy) * v00 + dx * (1.0f - dy) * v01 + (1.0f - dx) * dy * v10 + dx * dy * v11;
}
__device__ __forceinline__ float2 kaizen_bilinear_xy(const float4 *buf, int pitchFloats, int w, int h, float fx, float fy) {
    const int x0 = min(max((int)floorf(fx), 0), w - 1);
    const int y0 = min(max((int)floorf(fy), 0), h - 1);
    const int x1 = min(x0 + 1, w - 1), y1 = min(y0 + 1, h - 1);
    const float dx = fx - (float)x0, dy = fy - (float)y0;
    const float4 v00 = buf[y0 * pitchFloats + x0], v01 = buf[y0 * pitchFloats + x1];
    const float4 v10 = buf[y1 * pitchFloats + x0], v11 = buf[y1 * pitchFloats + x1];
    const float w00 = (1.0f - dx) * (1.0f - dy), w01 = dx * (1.0f - dy), w10 = (1.0f - dx) * dy, w11 = dx * dy;
    return make_float2(w00 * v00.x + w01 * v01.x + w10 * v10.x + w11 * v11.x,
                       w00 * v00.y + w01 * v01.y + w10 * v10.y + w11 * v11.y);
}

template<typename Type, int bit_depth>
__global__ void kaizen_thin_sobel_xy(float4 *__restrict__ pDstB, const int dstPitchFloats,
    const uint8_t *__restrict__ pSrcY, const int srcPitch, const int outW, const int outH) {
    const int ix = blockIdx.x * blockDim.x + threadIdx.x;
    const int iy = blockIdx.y * blockDim.y + threadIdx.y;
    if (ix >= outW || iy >= outH) return;
    const int xl = max(ix - 1, 0), xr = min(ix + 1, outW - 1), yt = max(iy - 1, 0), yb = min(iy + 1, outH - 1);
    #define KZRY(xx, yy) kaizen_read_y_norm<Type, bit_depth>(pSrcY, srcPitch, xx, yy)
    const float l_t = KZRY(xl, yt), c_t = KZRY(ix, yt), r_t = KZRY(xr, yt);
    const float l_c = KZRY(xl, iy), c_c = KZRY(ix, iy), r_c = KZRY(xr, iy);
    const float l_b = KZRY(xl, yb), c_b = KZRY(ix, yb), r_b = KZRY(xr, yb);
    #undef KZRY
    const float xg_t = -l_t + r_t, yg_t = l_t + c_t + c_t + r_t;
    const float xg_c = -l_c + r_c;
    const float xg_b = -l_b + r_b, yg_b = l_b + c_b + c_b + r_b;
    const float xgrad = (xg_t + xg_c + xg_c + xg_b) * (1.0f / 8.0f);
    const float ygrad = (-yg_t + yg_b) * (1.0f / 8.0f);
    const float resp = powf(sqrtf(xgrad * xgrad + ygrad * ygrad), 0.7f);
    pDstB[iy * dstPitchFloats + ix] = make_float4(resp, 0.0f, 0.0f, 0.0f);
}
__global__ void kaizen_thin_gauss_x(float4 *__restrict__ pDstA, const int dstPitchFloats,
    const float4 *__restrict__ pSrcB, const int srcPitchFloats, const int outW, const int outH, const float sigma, const int r) {
    const int ix = blockIdx.x * blockDim.x + threadIdx.x;
    const int iy = blockIdx.y * blockDim.y + threadIdx.y;
    if (ix >= outW || iy >= outH) return;
    float acc = 0.0f, wsum = 0.0f;
    for (int dx = -r; dx <= r; ++dx) {
        const int xx = min(max(ix + dx, 0), outW - 1);
        const float w = kaizen_gauss_w((float)dx, sigma);
        acc += pSrcB[iy * srcPitchFloats + xx].x * w; wsum += w;
    }
    pDstA[iy * dstPitchFloats + ix] = make_float4(acc / wsum, 0.0f, 0.0f, 0.0f);
}
__global__ void kaizen_thin_gauss_y(float4 *__restrict__ pDstB, const int dstPitchFloats,
    const float4 *__restrict__ pSrcA, const int srcPitchFloats, const int outW, const int outH, const float sigma, const int r) {
    const int ix = blockIdx.x * blockDim.x + threadIdx.x;
    const int iy = blockIdx.y * blockDim.y + threadIdx.y;
    if (ix >= outW || iy >= outH) return;
    float acc = 0.0f, wsum = 0.0f;
    for (int dy = -r; dy <= r; ++dy) {
        const int yy = min(max(iy + dy, 0), outH - 1);
        const float w = kaizen_gauss_w((float)dy, sigma);
        acc += pSrcA[yy * srcPitchFloats + ix].x * w; wsum += w;
    }
    pDstB[iy * dstPitchFloats + ix] = make_float4(acc / wsum, 0.0f, 0.0f, 0.0f);
}
__global__ void kaizen_thin_kernel_xy(float4 *__restrict__ pDstA, const int dstPitchFloats,
    const float4 *__restrict__ pSrcB, const int srcPitchFloats, const int outW, const int outH) {
    const int ix = blockIdx.x * blockDim.x + threadIdx.x;
    const int iy = blockIdx.y * blockDim.y + threadIdx.y;
    if (ix >= outW || iy >= outH) return;
    const int xl = max(ix - 1, 0), xr = min(ix + 1, outW - 1), yt = max(iy - 1, 0), yb = min(iy + 1, outH - 1);
    #define KZS(xx, yy) pSrcB[(yy) * srcPitchFloats + (xx)].x
    const float l_t = KZS(xl, yt), c_t = KZS(ix, yt), r_t = KZS(xr, yt);
    const float l_c = KZS(xl, iy), c_c = KZS(ix, iy), r_c = KZS(xr, iy);
    const float l_b = KZS(xl, yb), c_b = KZS(ix, yb), r_b = KZS(xr, yb);
    #undef KZS
    const float xg_t = -l_t + r_t, yg_t = l_t + c_t + c_t + r_t;
    const float xg_c = -l_c + r_c;
    const float xg_b = -l_b + r_b, yg_b = l_b + c_b + c_b + r_b;
    const float xgrad = (xg_t + xg_c + xg_c + xg_b) * (1.0f / 8.0f);
    const float ygrad = (-yg_t + yg_b) * (1.0f / 8.0f);
    pDstA[iy * dstPitchFloats + ix] = make_float4(xgrad, ygrad, 0.0f, 0.0f);
}
template<typename Type, int bit_depth>
__global__ void kaizen_thin_copy_y_to_ref(float4 *__restrict__ pDst, const int dstPitchFloats,
    const uint8_t *__restrict__ pSrcY, const int srcPitch, const int outW, const int outH) {
    const int ix = blockIdx.x * blockDim.x + threadIdx.x;
    const int iy = blockIdx.y * blockDim.y + threadIdx.y;
    if (ix >= outW || iy >= outH) return;
    pDst[iy * dstPitchFloats + ix] = make_float4(kaizen_read_y_norm<Type, bit_depth>(pSrcY, srcPitch, ix, iy), 0.0f, 0.0f, 0.0f);
}
template<typename Type, int bit_depth>
__global__ void kaizen_thin_warp(uint8_t *__restrict__ pDstY, const int dstPitch,
    const float4 *__restrict__ pSrcA, const int srcAPitchFloats, const float4 *__restrict__ pSrcB, const int srcBPitchFloats,
    const int outW, const int outH, const float relstr) {
    const int ix = blockIdx.x * blockDim.x + threadIdx.x;
    const int iy = blockIdx.y * blockDim.y + threadIdx.y;
    if (ix >= outW || iy >= outH) return;
    const float2 flow = kaizen_bilinear_xy(pSrcB, srcBPitchFloats, outW, outH, (float)ix, (float)iy);
    const float invlen = 1.0f / (sqrtf(flow.x * flow.x + flow.y * flow.y) + 0.01f);
    const float ddx = flow.x * invlen * relstr;
    const float ddy = flow.y * invlen * relstr;
    float result = clamp(kaizen_bilinear_x(pSrcA, srcAPitchFloats, outW, outH, (float)ix - ddx, (float)iy - ddy), 0.0f, 1.0f);
    Type *ptr = (Type *)(pDstY + iy * dstPitch + ix * sizeof(Type));
    ptr[0] = (Type)(result * (float)((1 << bit_depth) - 1) + 0.5f);
}

// ---- DoG modes (dog_sharpen 1x, dog 2x): 7-tap binomial Gauss + 3-tap minmax ----

#define KAIZEN_DOG_STRENGTH        0.6f
#define KAIZEN_DOG_BLUR_CURVE      0.6f
#define KAIZEN_DOG_BLUR_THRESHOLD  0.1f
#define KAIZEN_DOG_NOISE_THRESHOLD 0.001f

__device__ __forceinline__ float kaizen_signf(float v) { return (float)((v > 0.0f) - (v < 0.0f)); }

template<typename Type, int bit_depth>
__global__ void kaizen_dog_kernel_x(float4 *__restrict__ pDstA, const int dstPitchFloats,
    const uint8_t *__restrict__ pSrcY, const int srcPitch, const int W, const int H) {
    const int ix = blockIdx.x * blockDim.x + threadIdx.x;
    const int iy = blockIdx.y * blockDim.y + threadIdx.y;
    if (ix >= W || iy >= H) return;
    const int xm2 = min(max(ix - 2, 0), W - 1), xm1 = min(max(ix - 1, 0), W - 1);
    const int xp1 = min(max(ix + 1, 0), W - 1), xp2 = min(max(ix + 2, 0), W - 1);
    const float vm2 = kaizen_read_y_norm<Type, bit_depth>(pSrcY, srcPitch, xm2, iy);
    const float vm1 = kaizen_read_y_norm<Type, bit_depth>(pSrcY, srcPitch, xm1, iy);
    const float vc  = kaizen_read_y_norm<Type, bit_depth>(pSrcY, srcPitch, ix,  iy);
    const float vp1 = kaizen_read_y_norm<Type, bit_depth>(pSrcY, srcPitch, xp1, iy);
    const float vp2 = kaizen_read_y_norm<Type, bit_depth>(pSrcY, srcPitch, xp2, iy);
    const float g = (vm2 + vp2) * 0.06136f + (vm1 + vp1) * 0.24477f + vc * 0.38774f;
    const float lo = fminf(fminf(vm1, vc), vp1), hi = fmaxf(fmaxf(vm1, vc), vp1);
    pDstA[iy * dstPitchFloats + ix] = make_float4(g, lo, hi, 0.0f);
}
__global__ void kaizen_dog_kernel_y(float4 *__restrict__ pDstB, const int dstPitchFloats,
    const float4 *__restrict__ pSrcA, const int srcPitchFloats, const int W, const int H) {
    const int ix = blockIdx.x * blockDim.x + threadIdx.x;
    const int iy = blockIdx.y * blockDim.y + threadIdx.y;
    if (ix >= W || iy >= H) return;
    const int ym2 = min(max(iy - 2, 0), H - 1), ym1 = min(max(iy - 1, 0), H - 1);
    const int yp1 = min(max(iy + 1, 0), H - 1), yp2 = min(max(iy + 2, 0), H - 1);
    const float4 vm2 = pSrcA[ym2 * srcPitchFloats + ix], vm1 = pSrcA[ym1 * srcPitchFloats + ix];
    const float4 vc = pSrcA[iy * srcPitchFloats + ix], vp1 = pSrcA[yp1 * srcPitchFloats + ix], vp2 = pSrcA[yp2 * srcPitchFloats + ix];
    const float g = (vm2.x + vp2.x) * 0.06136f + (vm1.x + vp1.x) * 0.24477f + vc.x * 0.38774f;
    const float lo = fminf(fminf(vm1.y, vc.y), vp1.y), hi = fmaxf(fmaxf(vm1.z, vc.z), vp1.z);
    pDstB[iy * dstPitchFloats + ix] = make_float4(g, lo, hi, 0.0f);
}
template<typename Type, int bit_depth>
__global__ void kaizen_dog_apply_soft(uint8_t *__restrict__ pDstY, const int dstPitch,
    const uint8_t *__restrict__ pSrcLuma, const int srcLumaPitch, const float4 *__restrict__ pSrcMM, const int srcMMPitchFloats,
    const int W, const int H, const float strength) {
    const int ix = blockIdx.x * blockDim.x + threadIdx.x;
    const int iy = blockIdx.y * blockDim.y + threadIdx.y;
    if (ix >= W || iy >= H) return;
    const float luma = kaizen_read_y_norm<Type, bit_depth>(pSrcLuma, srcLumaPitch, ix, iy);
    const float4 mm = pSrcMM[iy * srcMMPitchFloats + ix];
    const float c = (luma - mm.x) * strength;
    const float t_range = KAIZEN_DOG_BLUR_THRESHOLD - KAIZEN_DOG_NOISE_THRESHOLD;
    float c_t; const float c_abs = fabsf(c);
    if (c_abs > KAIZEN_DOG_NOISE_THRESHOLD) {
        float t = (c_abs - KAIZEN_DOG_NOISE_THRESHOLD) / t_range;
        t = powf(t, KAIZEN_DOG_BLUR_CURVE);
        t = t * t_range + KAIZEN_DOG_NOISE_THRESHOLD;
        c_t = t * kaizen_signf(c);
    } else c_t = c;
    float result = clamp(clamp(luma + c_t, mm.y, mm.z), 0.0f, 1.0f);
    Type *ptr = (Type *)(pDstY + iy * dstPitch + ix * sizeof(Type));
    ptr[0] = (Type)(result * (float)((1 << bit_depth) - 1) + 0.5f);
}
template<typename Type, int bit_depth>
__global__ void kaizen_dog_apply_upscale(uint8_t *__restrict__ pDstY, const int dstPitch,
    cudaTextureObject_t tex, const float4 *__restrict__ pSrcMM, const int srcMMPitchFloats, const int srcW, const int srcH, const int outW, const int outH) {
    const int ix = blockIdx.x * blockDim.x + threadIdx.x;
    const int iy = blockIdx.y * blockDim.y + threadIdx.y;
    if (ix >= outW || iy >= outH) return;
    const float luma = tex2D<float>(tex, ((float)ix + 0.5f) / (float)outW, ((float)iy + 0.5f) / (float)outH);
    const float fx = ((float)ix + 0.5f) * (float)srcW / (float)outW - 0.5f;
    const float fy = ((float)iy + 0.5f) * (float)srcH / (float)outH - 0.5f;
    const int x0 = min(max((int)floorf(fx), 0), srcW - 1), y0 = min(max((int)floorf(fy), 0), srcH - 1);
    const int x1 = min(x0 + 1, srcW - 1), y1 = min(y0 + 1, srcH - 1);
    const float dx = fx - (float)x0, dy = fy - (float)y0;
    const float w00 = (1.0f - dx) * (1.0f - dy), w01 = dx * (1.0f - dy), w10 = (1.0f - dx) * dy, w11 = dx * dy;
    const float4 m00 = pSrcMM[y0 * srcMMPitchFloats + x0], m01 = pSrcMM[y0 * srcMMPitchFloats + x1];
    const float4 m10 = pSrcMM[y1 * srcMMPitchFloats + x0], m11 = pSrcMM[y1 * srcMMPitchFloats + x1];
    const float blur = w00 * m00.x + w01 * m01.x + w10 * m10.x + w11 * m11.x;
    const float lo = w00 * m00.y + w01 * m01.y + w10 * m10.y + w11 * m11.y;
    const float hi = w00 * m00.z + w01 * m01.z + w10 * m10.z + w11 * m11.z;
    const float c = (luma - blur) * KAIZEN_DOG_STRENGTH;
    float result = clamp(clamp(luma + c, lo, hi), 0.0f, 1.0f);
    Type *ptr = (Type *)(pDstY + iy * dstPitch + ix * sizeof(Type));
    ptr[0] = (Type)(result * (float)((1 << bit_depth) - 1) + 0.5f);
}

// ---- dtd (fused darken 1.8 -> thin 0.4 -> deblur 0.5, 2x composite) ----

template<typename Type, int bit_depth>
__global__ void kaizen_copy_y_to_y(uint8_t *__restrict__ pDst, const int dstPitch, const uint8_t *__restrict__ pSrc, const int srcPitch, const int W, const int H) {
    const int ix = blockIdx.x * blockDim.x + threadIdx.x;
    const int iy = blockIdx.y * blockDim.y + threadIdx.y;
    if (ix >= W || iy >= H) return;
    *(Type *)(pDst + iy * dstPitch + ix * sizeof(Type)) = *(const Type *)(pSrc + iy * srcPitch + ix * sizeof(Type));
}

template<typename Type, int bit_depth>
__global__ void kaizen_dtd_warp(uint8_t *__restrict__ pDstY, const int dstPitch,
    const uint8_t *__restrict__ pSrcLuma, const int srcLumaPitch, const float4 *__restrict__ pSrcFlow, const int srcFlowPitchFloats,
    const int srcW, const int srcH, const int outW, const int outH, const float relstr) {
    const int ix = blockIdx.x * blockDim.x + threadIdx.x;
    const int iy = blockIdx.y * blockDim.y + threadIdx.y;
    if (ix >= outW || iy >= outH) return;
    const float fx_1x = ((float)ix + 0.5f) * (float)srcW / (float)outW - 0.5f;
    const float fy_1x = ((float)iy + 0.5f) * (float)srcH / (float)outH - 0.5f;
    const float2 flow = kaizen_bilinear_xy(pSrcFlow, srcFlowPitchFloats, srcW, srcH, fx_1x, fy_1x);
    const float invlen = 1.0f / (sqrtf(flow.x * flow.x + flow.y * flow.y) + 0.01f);
    const float fx = fx_1x - flow.x * invlen * relstr;
    const float fy = fy_1x - flow.y * invlen * relstr;
    const int x0 = min(max((int)floorf(fx), 0), srcW - 1), y0 = min(max((int)floorf(fy), 0), srcH - 1);
    const int x1 = min(x0 + 1, srcW - 1), y1 = min(y0 + 1, srcH - 1);
    const float dx = fx - (float)x0, dy = fy - (float)y0;
    const float v00 = kaizen_read_y_norm<Type, bit_depth>(pSrcLuma, srcLumaPitch, x0, y0);
    const float v01 = kaizen_read_y_norm<Type, bit_depth>(pSrcLuma, srcLumaPitch, x1, y0);
    const float v10 = kaizen_read_y_norm<Type, bit_depth>(pSrcLuma, srcLumaPitch, x0, y1);
    const float v11 = kaizen_read_y_norm<Type, bit_depth>(pSrcLuma, srcLumaPitch, x1, y1);
    float result = clamp((1.0f - dx) * (1.0f - dy) * v00 + dx * (1.0f - dy) * v01 + (1.0f - dx) * dy * v10 + dx * dy * v11, 0.0f, 1.0f);
    Type *ptr = (Type *)(pDstY + iy * dstPitch + ix * sizeof(Type));
    ptr[0] = (Type)(result * (float)((1 << bit_depth) - 1) + 0.5f);
}

// =================== filter class ===================

tstring NVEncFilterParamKaizen::print() const {
    return kaizen.print();
}

NVEncFilterKaizen::NVEncFilterKaizen() :
    m_mode(VppKaizenMode::Original), m_isDogMode(false),
    m_inW(0), m_inH(0), m_outW(0), m_outH(0), m_scale(2), m_strength(0.5f), m_chromaMode(0), m_chromaJoint(false), m_doChroma(true),
    m_doPrefilter(false), m_prefilterDenoise(VppKaizenDenoise::Off),
    m_doDarken(false), m_darkenSigma(1.0f), m_darkenRadius(2),
    m_doThin(false), m_thinSigma(2.0f), m_thinRelstr(0.6f), m_thinRadius(4),
    m_denoise(VppKaizenDenoise::Off), m_denoiseIntensity(0.1f), m_denoiseSpatial(1.0f), m_denoiseCurve(1.0f), m_denoiseHistReg(-1.0f),
    m_clampHighlights(false), m_antiring(0.0f),
    m_chromaLowW(0), m_chromaLowH(0),
    m_scratchA(), m_scratchB(), m_clampStatsH(), m_clampStats(), m_chromaLumaLowres(), m_prefilterPlane(), m_prefilterRef(), m_postResize() {
    m_name = _T("kaizen");
}
NVEncFilterKaizen::~NVEncFilterKaizen() { close(); }

RGY_ERR NVEncFilterKaizen::runBaseChainY(RGYFrameInfo *pOutY, const RGYFrameInfo *pInY, cudaStream_t stream) {
    cudaTextureObject_t tex = 0;
    auto cerr = setKaizenSrcTex<uint8_t>(tex, pInY);
    if (cerr != cudaSuccess) return err_to_rgy(cerr);

    float4 *A = (float4 *)m_scratchA->ptr;
    float4 *B = (float4 *)m_scratchB->ptr;
    const int pf = m_outW; // pitch in float4 elements (tightly packed)
    dim3 block(KAIZEN_BLOCK_X, KAIZEN_BLOCK_Y);
    dim3 grid(divCeil(m_outW, block.x), divCeil(m_outH, block.y));

    kaizen_sobel_x<<<grid, block, 0, stream>>>(A, pf, tex, m_outW, m_outH);
    kaizen_sobel_y<<<grid, block, 0, stream>>>(B, pf, A, pf, m_outW, m_outH, m_strength);
    kaizen_refine_x<<<grid, block, 0, stream>>>(A, pf, B, pf, m_outW, m_outH);
    kaizen_refine_y<<<grid, block, 0, stream>>>(B, pf, A, pf, m_outW, m_outH);
    kaizen_apply<uint8_t, 8><<<grid, block, 0, stream>>>((uint8_t *)pOutY->ptr[0], pOutY->pitch[0], tex, B, pf, m_outW, m_outH);

    cerr = cudaGetLastError();
    cudaDestroyTextureObject(tex);
    return err_to_rgy(cerr);
}

RGY_ERR NVEncFilterKaizen::runDogChain(RGYFrameInfo *pOutY, const RGYFrameInfo *pInY, cudaStream_t stream) {
    float4 *A = (float4 *)m_scratchA->ptr;
    float4 *B = (float4 *)m_scratchB->ptr;
    const int sW = pInY->width, sH = pInY->height;
    const int pf = sW; // 1x scratch (pitch in float4 = source width)
    dim3 block(KAIZEN_BLOCK_X, KAIZEN_BLOCK_Y);
    dim3 gridS(divCeil(sW, block.x), divCeil(sH, block.y));
    kaizen_dog_kernel_x<uint8_t, 8><<<gridS, block, 0, stream>>>(A, pf, (const uint8_t *)pInY->ptr[0], pInY->pitch[0], sW, sH);
    kaizen_dog_kernel_y<<<gridS, block, 0, stream>>>(B, pf, A, pf, sW, sH);
    if (m_mode == VppKaizenMode::DogSharpen) { // 1x in-place sharpen
        kaizen_dog_apply_soft<uint8_t, 8><<<gridS, block, 0, stream>>>((uint8_t *)pOutY->ptr[0], pOutY->pitch[0],
            (const uint8_t *)pInY->ptr[0], pInY->pitch[0], B, pf, sW, sH, KAIZEN_DOG_STRENGTH);
        return err_to_rgy(cudaGetLastError());
    }
    // Dog: 2x upscale via bilinear source + minmax-clamped unsharp from the 1x scratch.
    cudaTextureObject_t tex = 0;
    auto cerr = setKaizenSrcTex<uint8_t>(tex, pInY);
    if (cerr != cudaSuccess) return err_to_rgy(cerr);
    dim3 gridD(divCeil(m_outW, block.x), divCeil(m_outH, block.y));
    kaizen_dog_apply_upscale<uint8_t, 8><<<gridD, block, 0, stream>>>((uint8_t *)pOutY->ptr[0], pOutY->pitch[0], tex, B, pf, sW, sH, m_outW, m_outH);
    cerr = cudaGetLastError();
    cudaDestroyTextureObject(tex);
    return err_to_rgy(cerr);
}

RGY_ERR NVEncFilterKaizen::runDtdChain(RGYFrameInfo *pOutY, const RGYFrameInfo *pInY, cudaStream_t stream) {
    // dtd = darken_to_deblur: a 2x composite that darkens (1.8) and thins (0.4) the
    // 1x luma, warps+upscales it to 2x in one pass, then deblurs (DoG soft, 0.5) at 2x.
    float4 *A = (float4 *)m_scratchA->ptr;
    float4 *B = (float4 *)m_scratchB->ptr;
    const int sW = pInY->width, sH = pInY->height;
    const int pf1 = sW;        // 1x scratch pitch (float4 elements = source width)
    const int pf2 = m_outW;    // 2x scratch pitch
    uint8_t *src1x = (uint8_t *)m_dtdSrcLuma->ptr;
    const int src1xPitch = sW;
    dim3 block(KAIZEN_BLOCK_X, KAIZEN_BLOCK_Y);
    dim3 gridS(divCeil(sW, block.x), divCeil(sH, block.y));
    dim3 gridD(divCeil(m_outW, block.x), divCeil(m_outH, block.y));

    // copy input 1x luma into the writable dtd scratch.
    kaizen_copy_y_to_y<uint8_t, 8><<<gridS, block, 0, stream>>>(src1x, src1xPitch, (const uint8_t *)pInY->ptr[0], pInY->pitch[0], sW, sH);

    // Stage A: darken (strength 1.8) in place on the 1x luma.
    {
        const float sigma = 1.0f * (float)sH / 1080.0f;
        const int r = std::min(std::max((int)ceilf(sigma * 2.0f), 1), 12);
        kaizen_darken_gauss1_x<uint8_t, 8><<<gridS, block, 0, stream>>>(A, pf1, src1x, src1xPitch, sW, sH, sigma, r);
        kaizen_darken_dog_y<uint8_t, 8><<<gridS, block, 0, stream>>>(B, pf1, A, pf1, src1x, src1xPitch, sW, sH, sigma, r);
        kaizen_darken_gauss2_x<<<gridS, block, 0, stream>>>(A, pf1, B, pf1, sW, sH, sigma, r);
        kaizen_darken_apply_y<uint8_t, 8><<<gridS, block, 0, stream>>>(src1x, src1xPitch, A, pf1, sW, sH, sigma, r, 1.8f);
    }

    // Stage B: thin flow field (1x) then warp+upscale the darkened luma to 2x (relstr 0.4).
    {
        const float sigma = 2.0f * (float)sH / 1080.0f;
        const int r = std::min(std::max((int)ceilf(sigma * 2.0f), 1), 12);
        const float relstr = (float)sH / 1080.0f * 0.4f;
        kaizen_thin_sobel_xy<uint8_t, 8><<<gridS, block, 0, stream>>>(B, pf1, src1x, src1xPitch, sW, sH);
        kaizen_thin_gauss_x<<<gridS, block, 0, stream>>>(A, pf1, B, pf1, sW, sH, sigma, r);
        kaizen_thin_gauss_y<<<gridS, block, 0, stream>>>(B, pf1, A, pf1, sW, sH, sigma, r);
        kaizen_thin_kernel_xy<<<gridS, block, 0, stream>>>(A, pf1, B, pf1, sW, sH); // flow -> A
        kaizen_dtd_warp<uint8_t, 8><<<gridD, block, 0, stream>>>((uint8_t *)pOutY->ptr[0], pOutY->pitch[0],
            src1x, src1xPitch, A, pf1, sW, sH, m_outW, m_outH, relstr);
    }

    // Stage C: deblur (DoG soft, strength 0.5) in place on the 2x output.
    {
        kaizen_dog_kernel_x<uint8_t, 8><<<gridD, block, 0, stream>>>(A, pf2, (const uint8_t *)pOutY->ptr[0], pOutY->pitch[0], m_outW, m_outH);
        kaizen_dog_kernel_y<<<gridD, block, 0, stream>>>(B, pf2, A, pf2, m_outW, m_outH);
        kaizen_dog_apply_soft<uint8_t, 8><<<gridD, block, 0, stream>>>((uint8_t *)pOutY->ptr[0], pOutY->pitch[0],
            (const uint8_t *)pOutY->ptr[0], pOutY->pitch[0], B, pf2, m_outW, m_outH, 0.5f);
    }
    return err_to_rgy(cudaGetLastError());
}

RGY_ERR NVEncFilterKaizen::runChromaResize(RGYFrameInfo *pOutC, const RGYFrameInfo *pInC, cudaStream_t stream) {
    // pInC / pOutC are single chroma planes (from getPlane); stride 1 (planar).
    dim3 block(KAIZEN_BLOCK_X, KAIZEN_BLOCK_Y);
    dim3 grid(divCeil(pOutC->width, block.x), divCeil(pOutC->height, block.y));
    kaizen_chroma_resize<uint8_t, 8><<<grid, block, 0, stream>>>(
        (uint8_t *)pOutC->ptr[0], pOutC->pitch[0], 1, pOutC->width, pOutC->height,
        (const uint8_t *)pInC->ptr[0], pInC->pitch[0], 1, pInC->width, pInC->height, m_chromaMode);
    return err_to_rgy(cudaGetLastError());
}

RGY_ERR NVEncFilterKaizen::runChromaJoint(RGYFrameInfo *pOutC, const RGYFrameInfo *pInC, const RGYFrameInfo *pSrcLumaY, cudaStream_t stream) {
    dim3 block(KAIZEN_BLOCK_X, KAIZEN_BLOCK_Y);
    dim3 grid(divCeil(pOutC->width, block.x), divCeil(pOutC->height, block.y));
    kaizen_chroma_joint_bilateral<uint8_t, 8><<<grid, block, 0, stream>>>(
        (uint8_t *)pOutC->ptr[0], pOutC->pitch[0], pOutC->width, pOutC->height,
        (const uint8_t *)pInC->ptr[0], pInC->pitch[0], pInC->width, pInC->height,
        (const uint8_t *)pSrcLumaY->ptr[0], pSrcLumaY->pitch[0],
        (const uint8_t *)m_chromaLumaLowres->ptr, m_chromaLowW, 2.0f, 128.0f);
    return err_to_rgy(cudaGetLastError());
}

RGY_ERR NVEncFilterKaizen::runPrefilterDenoise(const RGYFrameInfo *pInY, cudaStream_t stream) {
    float4 *ref = (float4 *)m_prefilterRef->ptr;
    uint8_t *plane = (uint8_t *)m_prefilterPlane->ptr;
    const int W = pInY->width, H = pInY->height, pf = W, pp = W;
    dim3 block(KAIZEN_BLOCK_X, KAIZEN_BLOCK_Y);
    dim3 grid(divCeil(W, block.x), divCeil(H, block.y));
    kaizen_copy_y_to_scratch<uint8_t, 8><<<grid, block, 0, stream>>>(ref, pf, (const uint8_t *)pInY->ptr[0], pInY->pitch[0], W, H);
    const float sigma_s = m_denoiseSpatial, isigma = m_denoiseIntensity, curve = m_denoiseCurve;
    float reg = m_denoiseHistReg;
    if (reg < 0.0f) reg = (m_prefilterDenoise == VppKaizenDenoise::Mode) ? 0.2f : 0.0f;
    switch (m_prefilterDenoise) {
    case VppKaizenDenoise::Mean:   kaizen_denoise_mean<uint8_t, 8><<<grid, block, 0, stream>>>(plane, pp, ref, pf, W, H, sigma_s, isigma, curve); break;
    case VppKaizenDenoise::Median: kaizen_denoise_median<uint8_t, 8><<<grid, block, 0, stream>>>(plane, pp, ref, pf, W, H, sigma_s, isigma, curve, reg); break;
    case VppKaizenDenoise::Mode:   kaizen_denoise_mode<uint8_t, 8><<<grid, block, 0, stream>>>(plane, pp, ref, pf, W, H, sigma_s, isigma, curve, reg); break;
    default: break;
    }
    return err_to_rgy(cudaGetLastError());
}

RGY_ERR NVEncFilterKaizen::runDarkenY(RGYFrameInfo *pOutY, cudaStream_t stream) {
    float4 *A = (float4 *)m_scratchA->ptr;
    float4 *B = (float4 *)m_scratchB->ptr;
    const int pf = m_outW;
    const float sigma = m_darkenSigma;
    const int r = m_darkenRadius;
    dim3 block(KAIZEN_BLOCK_X, KAIZEN_BLOCK_Y);
    dim3 grid(divCeil(m_outW, block.x), divCeil(m_outH, block.y));
    kaizen_darken_gauss1_x<uint8_t, 8><<<grid, block, 0, stream>>>(A, pf, (const uint8_t *)pOutY->ptr[0], pOutY->pitch[0], m_outW, m_outH, sigma, r);
    kaizen_darken_dog_y<uint8_t, 8><<<grid, block, 0, stream>>>(B, pf, A, pf, (const uint8_t *)pOutY->ptr[0], pOutY->pitch[0], m_outW, m_outH, sigma, r);
    kaizen_darken_gauss2_x<<<grid, block, 0, stream>>>(A, pf, B, pf, m_outW, m_outH, sigma, r);
    kaizen_darken_apply_y<uint8_t, 8><<<grid, block, 0, stream>>>((uint8_t *)pOutY->ptr[0], pOutY->pitch[0], A, pf, m_outW, m_outH, sigma, r, KAIZEN_DARKEN_STRENGTH);
    return err_to_rgy(cudaGetLastError());
}

RGY_ERR NVEncFilterKaizen::runThinY(RGYFrameInfo *pOutY, cudaStream_t stream) {
    float4 *A = (float4 *)m_scratchA->ptr;
    float4 *B = (float4 *)m_scratchB->ptr;
    const int pf = m_outW;
    const float sigma = m_thinSigma;
    const int r = m_thinRadius;
    uint8_t *Y = (uint8_t *)pOutY->ptr[0];
    const int yp = pOutY->pitch[0];
    dim3 block(KAIZEN_BLOCK_X, KAIZEN_BLOCK_Y);
    dim3 grid(divCeil(m_outW, block.x), divCeil(m_outH, block.y));
    kaizen_thin_sobel_xy<uint8_t, 8><<<grid, block, 0, stream>>>(B, pf, Y, yp, m_outW, m_outH);        // Y -> B (shaped Sobel)
    kaizen_thin_gauss_x<<<grid, block, 0, stream>>>(A, pf, B, pf, m_outW, m_outH, sigma, r);            // B -> A
    kaizen_thin_gauss_y<<<grid, block, 0, stream>>>(B, pf, A, pf, m_outW, m_outH, sigma, r);            // A -> B
    kaizen_thin_kernel_xy<<<grid, block, 0, stream>>>(A, pf, B, pf, m_outW, m_outH);                    // B -> A (flow)
    kaizen_thin_copy_y_to_ref<uint8_t, 8><<<grid, block, 0, stream>>>(B, pf, Y, yp, m_outW, m_outH);    // Y -> B (Yref)
    kaizen_thin_warp<uint8_t, 8><<<grid, block, 0, stream>>>(Y, yp, B, pf, A, pf, m_outW, m_outH, m_thinRelstr); // Yref=B, flow=A
    return err_to_rgy(cudaGetLastError());
}

RGY_ERR NVEncFilterKaizen::runDenoiseY(RGYFrameInfo *pOutY, cudaStream_t stream) {
    float4 *ref = (float4 *)m_scratchA->ptr;
    const int pf = m_outW;
    dim3 block(KAIZEN_BLOCK_X, KAIZEN_BLOCK_Y);
    dim3 grid(divCeil(m_outW, block.x), divCeil(m_outH, block.y));
    // stash the current output Y into the reference scratch, then denoise back into Y.
    kaizen_copy_y_to_scratch<uint8_t, 8><<<grid, block, 0, stream>>>(ref, pf, (const uint8_t *)pOutY->ptr[0], pOutY->pitch[0], m_outW, m_outH);
    const float sigma_s = m_denoiseSpatial;
    const float isigma = m_denoiseIntensity;
    const float curve = m_denoiseCurve;
    float reg = m_denoiseHistReg;
    if (reg < 0.0f) reg = (m_denoise == VppKaizenDenoise::Mode) ? 0.2f : 0.0f; // sentinel -> tier default
    switch (m_denoise) {
    case VppKaizenDenoise::Mean:
        kaizen_denoise_mean<uint8_t, 8><<<grid, block, 0, stream>>>((uint8_t *)pOutY->ptr[0], pOutY->pitch[0], ref, pf, m_outW, m_outH, sigma_s, isigma, curve);
        break;
    case VppKaizenDenoise::Median:
        kaizen_denoise_median<uint8_t, 8><<<grid, block, 0, stream>>>((uint8_t *)pOutY->ptr[0], pOutY->pitch[0], ref, pf, m_outW, m_outH, sigma_s, isigma, curve, reg);
        break;
    case VppKaizenDenoise::Mode:
        kaizen_denoise_mode<uint8_t, 8><<<grid, block, 0, stream>>>((uint8_t *)pOutY->ptr[0], pOutY->pitch[0], ref, pf, m_outW, m_outH, sigma_s, isigma, curve, reg);
        break;
    default: break;
    }
    return err_to_rgy(cudaGetLastError());
}

RGY_ERR NVEncFilterKaizen::runClampHighlightsY(RGYFrameInfo *pOutY, const RGYFrameInfo *pSrcY, cudaStream_t stream) {
    const int sW = pSrcY->width, sH = pSrcY->height, sStride = sW;
    float *statsH = (float *)m_clampStatsH->ptr;
    float *stats = (float *)m_clampStats->ptr;
    dim3 block(16, 16);
    dim3 gridS(divCeil(sW, block.x), divCeil(sH, block.y));
    kaizen_clamp_h_max_y<uint8_t, 8><<<gridS, block, 0, stream>>>(statsH, sStride, (const uint8_t *)pSrcY->ptr[0], pSrcY->pitch[0], sW, sH);
    kaizen_clamp_v_max<<<gridS, block, 0, stream>>>(stats, sStride, statsH, sStride, sW, sH);
    dim3 gridD(divCeil(m_outW, block.x), divCeil(m_outH, block.y));
    kaizen_clamp_apply_y<uint8_t, 8><<<gridD, block, 0, stream>>>((uint8_t *)pOutY->ptr[0], pOutY->pitch[0], stats, sStride, m_outW, m_outH, sW, sH);
    return err_to_rgy(cudaGetLastError());
}

RGY_ERR NVEncFilterKaizen::runAntiringY(RGYFrameInfo *pOutY, const RGYFrameInfo *pSrcY, cudaStream_t stream) {
    dim3 block(KAIZEN_BLOCK_X, KAIZEN_BLOCK_Y);
    dim3 grid(divCeil(m_outW, block.x), divCeil(m_outH, block.y));
    kaizen_antiring_y<uint8_t, 8><<<grid, block, 0, stream>>>((uint8_t *)pOutY->ptr[0], pOutY->pitch[0],
        (const uint8_t *)pSrcY->ptr[0], pSrcY->pitch[0], m_outW, m_outH, pSrcY->width, pSrcY->height, m_antiring);
    return err_to_rgy(cudaGetLastError());
}

RGY_ERR NVEncFilterKaizen::init(shared_ptr<NVEncFilterParam> pParam, shared_ptr<RGYLog> pPrintMes) {
    m_pLog = pPrintMes;
    auto prm = std::dynamic_pointer_cast<NVEncFilterParamKaizen>(pParam);
    if (!prm) {
        AddMessage(RGY_LOG_ERROR, _T("Invalid parameter type.\n"));
        return RGY_ERR_INVALID_PARAM;
    }
    const auto inCsp = prm->frameIn.csp;
    if ((inCsp != RGY_CSP_YV12 && inCsp != RGY_CSP_NV12) || prm->frameIn.bitdepth != 8) {
        AddMessage(RGY_LOG_ERROR, _T("kaizen: supports 8-bit yuv420 (yv12/nv12) only; got %s %dbit.\n"),
            RGY_CSP_NAMES[inCsp], prm->frameIn.bitdepth);
        return RGY_ERR_UNSUPPORTED;
    }
    if (inCsp == RGY_CSP_NV12) {
        AddMessage(RGY_LOG_ERROR, _T("kaizen: nv12 input not yet supported by this build; provide yv12.\n"));
        return RGY_ERR_UNSUPPORTED;
    }
    const auto &kz = prm->kaizen;
    // Increment 1: base modes only.
    m_mode = kz.mode;
    m_isDogMode = (kz.mode == VppKaizenMode::DogSharpen || kz.mode == VppKaizenMode::Dog);
    m_doDarken = (kz.darken != VppKaizenDarken::Off) || (kz.mode == VppKaizenMode::DarkenHQ);
    m_doThin   = (kz.thin   != VppKaizenThin::Off)   || (kz.mode == VppKaizenMode::ThinHQ);
    if (kz.darken == VppKaizenDarken::Fast || kz.darken == VppKaizenDarken::VeryFast
        || kz.thin == VppKaizenThin::Fast || kz.thin == VppKaizenThin::VeryFast) {
        AddMessage(RGY_LOG_WARN, _T("kaizen: darken/thin fast/veryfast tiers not yet implemented; using hq.\n"));
    }
    m_denoise = kz.denoise;
    m_denoiseIntensity = kz.denoiseIntensity;
    m_denoiseSpatial = kz.denoiseSpatial;
    m_denoiseCurve = kz.denoiseCurve;
    m_denoiseHistReg = kz.denoiseHistReg;
    m_clampHighlights = kz.clampHighlights;
    m_antiring = std::max(0.0f, std::min(1.0f, kz.antiring));

    m_scale = kz.scale;
    if (kz.mode == VppKaizenMode::DogSharpen) m_scale = 1; // dog_sharpen is 1x
    else if (kz.mode == VppKaizenMode::Dog)   m_scale = 2; // dog is 2x
    else if (kz.mode == VppKaizenMode::Dtd)   m_scale = 2; // dtd composite is 2x
    if (m_scale != 1 && m_scale != 2) m_scale = 2;
    m_strength = (kz.mode == VppKaizenMode::Deblur) ? 1.0f : kz.strength;
    m_chromaJoint = (kz.chromaResize == VppKaizenChromaResize::Joint);
    m_chromaMode = m_chromaJoint ? 0 : (int)kz.chromaResize; // joint handled separately; else geometric mode
    m_doChroma = kz.chroma;
    m_doPrefilter = (kz.prefilterDenoise != VppKaizenDenoise::Off);
    m_prefilterDenoise = kz.prefilterDenoise;

    const int inW = prm->frameIn.width;
    const int inH = prm->frameIn.height;
    m_inW = inW;
    m_inH = inH;
    m_outW = inW * m_scale;
    m_outH = inH * m_scale;
    // darken Gaussian sigma scales with output height (reference at 1080p, sigma_ref=1.0).
    m_darkenSigma = 1.0f * (float)m_outH / 1080.0f;
    m_darkenRadius = std::min(std::max((int)ceilf(m_darkenSigma * 2.0f), 1), 12);
    m_thinSigma = 2.0f * (float)m_outH / 1080.0f;
    m_thinRadius = std::min(std::max((int)ceilf(m_thinSigma * 2.0f), 1), 12);
    m_thinRelstr = (float)m_outH / 1080.0f * 0.6f;

    auto frameOut = prm->frameOut;
    frameOut.csp = inCsp;
    frameOut.width = m_outW;
    frameOut.height = m_outH;
    prm->frameOut = frameOut;
    auto err = AllocFrameBuf(prm->frameOut, 1);
    if (err != RGY_ERR_NONE) {
        AddMessage(RGY_LOG_ERROR, _T("kaizen: failed to allocate output frame buffer: %s.\n"), get_err_mes(err));
        return err;
    }
    for (int i = 0; i < RGY_CSP_PLANES[m_frameBuf[0]->frame.csp]; i++) {
        prm->frameOut.pitch[i] = m_frameBuf[0]->frame.pitch[i];
    }

    // float4 ping-pong scratch at output dims.
    const size_t scratchBytes = (size_t)m_outW * m_outH * sizeof(float4);
    m_scratchA = std::unique_ptr<CUMemBuf>(new CUMemBuf(scratchBytes));
    m_scratchB = std::unique_ptr<CUMemBuf>(new CUMemBuf(scratchBytes));
    if (RGY_ERR_NONE != (err = m_scratchA->alloc()) || RGY_ERR_NONE != (err = m_scratchB->alloc())) {
        AddMessage(RGY_LOG_ERROR, _T("kaizen: failed to allocate scratch: %s.\n"), get_err_mes(err));
        return err;
    }
    if (m_clampHighlights) {
        const size_t statsBytes = (size_t)inW * inH * sizeof(float);
        m_clampStatsH = std::unique_ptr<CUMemBuf>(new CUMemBuf(statsBytes));
        m_clampStats  = std::unique_ptr<CUMemBuf>(new CUMemBuf(statsBytes));
        if (RGY_ERR_NONE != (err = m_clampStatsH->alloc()) || RGY_ERR_NONE != (err = m_clampStats->alloc())) {
            AddMessage(RGY_LOG_ERROR, _T("kaizen: failed to allocate clamp stats: %s.\n"), get_err_mes(err));
            return err;
        }
    }
    if (m_chromaJoint && m_scale == 2 && m_doChroma) {
        m_chromaLowW = inW / 2;
        m_chromaLowH = inH / 2;
        m_chromaLumaLowres = std::unique_ptr<CUMemBuf>(new CUMemBuf((size_t)m_chromaLowW * m_chromaLowH));
        if (RGY_ERR_NONE != (err = m_chromaLumaLowres->alloc())) {
            AddMessage(RGY_LOG_ERROR, _T("kaizen: failed to allocate joint-chroma lowres: %s.\n"), get_err_mes(err));
            return err;
        }
    } else {
        m_chromaJoint = false; // joint needs scale=2 + chroma
    }
    if (m_doPrefilter) {
        m_prefilterPlane = std::unique_ptr<CUMemBuf>(new CUMemBuf((size_t)inW * inH));
        m_prefilterRef   = std::unique_ptr<CUMemBuf>(new CUMemBuf((size_t)inW * inH * sizeof(float4)));
        if (RGY_ERR_NONE != (err = m_prefilterPlane->alloc()) || RGY_ERR_NONE != (err = m_prefilterRef->alloc())) {
            AddMessage(RGY_LOG_ERROR, _T("kaizen: failed to allocate prefilter buffers: %s.\n"), get_err_mes(err));
            return err;
        }
    }
    if (m_mode == VppKaizenMode::Dtd) {
        m_dtdSrcLuma = std::unique_ptr<CUMemBuf>(new CUMemBuf((size_t)inW * inH));
        if (RGY_ERR_NONE != (err = m_dtdSrcLuma->alloc())) {
            AddMessage(RGY_LOG_ERROR, _T("kaizen: failed to allocate dtd luma scratch: %s.\n"), get_err_mes(err));
            return err;
        }
    }

    // out_res end-of-chain resize (reuse NVEncFilterResize), like --vpp-onnx.
    m_postResize.reset();
    if (prm->kaizen.postResizeW != 0 && prm->kaizen.postResizeH != 0) {
        int tgtW = prm->kaizen.postResizeW;
        int tgtH = prm->kaizen.postResizeH;
        if (tgtW < 0 || tgtH < 0) {
            sInputCrop nocrop; memset(&nocrop, 0, sizeof(nocrop));
            set_auto_resolution(tgtW, tgtH, 1, 1, m_outW, m_outH, prm->sar[0], prm->sar[1], 2, 2, RGYResizeResMode::Normal, false, nocrop);
        }
        if (tgtW > 0 && tgtH > 0 && (tgtW != m_outW || tgtH != m_outH)) {
            auto rp = std::make_shared<NVEncFilterParamResize>();
            rp->interp = (prm->kaizen.postResizeAlgo == RGY_VPP_RESIZE_AUTO) ? RGY_VPP_RESIZE_LANCZOS4 : prm->kaizen.postResizeAlgo;
            rp->frameIn = prm->frameOut;
            rp->frameOut = prm->frameOut;
            rp->frameOut.width = tgtW;
            rp->frameOut.height = tgtH;
            rp->baseFps = prm->baseFps;
            rp->bOutOverwrite = false;
            m_postResize = std::make_unique<NVEncFilterResize>();
            auto rsts = m_postResize->init(rp, m_pLog);
            if (rsts != RGY_ERR_NONE) {
                AddMessage(RGY_LOG_ERROR, _T("kaizen: failed to init end-of-chain resize: %s.\n"), get_err_mes(rsts));
                return rsts;
            }
            prm->frameOut = rp->frameOut;
        }
    }

    static const TCHAR *chromaName[] = { _T("spline36"), _T("bilinear"), _T("bicubic"), _T("lanczos3") };
    const int chromaIdx = (m_chromaMode < 0) ? 0 : ((m_chromaMode > 3) ? 3 : m_chromaMode);
    tstring info = strsprintf(_T("kaizen: %s  %dx%d -> %dx%d (x%d)  strength=%.2f chroma=%s%s"),
        get_cx_desc(list_vpp_kaizen_mode, (int)kz.mode), inW, inH, m_outW, m_outH, m_scale, m_strength,
        m_chromaJoint ? _T("joint") : chromaName[chromaIdx],
        (m_doPrefilter ? _T(" prefilter") : _T("")));
    if (m_postResize) {
        info += strsprintf(_T(" -> out_res %dx%d (%s)"), prm->frameOut.width, prm->frameOut.height,
            get_cx_desc(list_vpp_resize, (prm->kaizen.postResizeAlgo == RGY_VPP_RESIZE_AUTO) ? RGY_VPP_RESIZE_LANCZOS4 : prm->kaizen.postResizeAlgo));
    }
    setFilterInfo(info);
    m_param = prm;
    return RGY_ERR_NONE;
}

RGY_ERR NVEncFilterKaizen::run_filter(const RGYFrameInfo *pInputFrame, RGYFrameInfo **ppOutputFrames, int *pOutputFrameNum, cudaStream_t stream) {
    if (pInputFrame->ptr[0] == nullptr) {
        *pOutputFrameNum = 0;
        return RGY_ERR_NONE;
    }
    auto pOutFrame = m_frameBuf[0].get();
    RGYFrameInfo *coreFrame = &pOutFrame->frame;
    copyFramePropWithoutRes(coreFrame, pInputFrame);

    // luma chain (DoG modes use a different upscaler; all others use the base Anime4K chain)
    auto planeInY = getPlane(pInputFrame, RGY_PLANE_Y);
    auto planeOutY = getPlane(coreFrame, RGY_PLANE_Y);
    RGYFrameInfo baseInY = planeInY;
    if (m_doPrefilter) {
        auto perr = runPrefilterDenoise(&planeInY, stream);
        if (perr != RGY_ERR_NONE) { AddMessage(RGY_LOG_ERROR, _T("kaizen: prefilter denoise failed: %s.\n"), get_err_mes(perr)); return perr; }
        baseInY.ptr[0] = (uint8_t *)m_prefilterPlane->ptr;
        baseInY.pitch[0] = m_inW; // denoised input luma, tight pitch
    }
    RGY_ERR err;
    if (m_mode == VppKaizenMode::Dtd)   err = runDtdChain(&planeOutY, &baseInY, stream);
    else if (m_isDogMode)               err = runDogChain(&planeOutY, &baseInY, stream);
    else                                err = runBaseChainY(&planeOutY, &baseInY, stream);
    if (err != RGY_ERR_NONE) { AddMessage(RGY_LOG_ERROR, _T("kaizen: luma chain failed: %s.\n"), get_err_mes(err)); return err; }

    // chroma (joint-bilateral or geometric resize); planar yv12 U, V
    if (m_chromaJoint) {
        // box-downscale the (original) input luma to the chroma-res guide.
        dim3 block(KAIZEN_BLOCK_X, KAIZEN_BLOCK_Y);
        dim3 gridLow(divCeil(m_chromaLowW, block.x), divCeil(m_chromaLowH, block.y));
        kaizen_chroma_luma_lowres<uint8_t, 8><<<gridLow, block, 0, stream>>>(
            (uint8_t *)m_chromaLumaLowres->ptr, m_chromaLowW, m_chromaLowW, m_chromaLowH,
            (const uint8_t *)planeInY.ptr[0], planeInY.pitch[0], m_inW, m_inH);
    }
    for (int p = 1; p < RGY_CSP_PLANES[coreFrame->csp]; p++) {
        auto inC = getPlane(pInputFrame, (RGY_PLANE)p);
        auto outC = getPlane(coreFrame, (RGY_PLANE)p);
        err = m_chromaJoint ? runChromaJoint(&outC, &inC, &planeInY, stream) : runChromaResize(&outC, &inC, stream);
        if (err != RGY_ERR_NONE) { AddMessage(RGY_LOG_ERROR, _T("kaizen: chroma failed: %s.\n"), get_err_mes(err)); return err; }
    }

    // post-process Y passes (order matches the reference: darken -> denoise -> clamp -> antiring)
    if (m_doDarken) {
        err = runDarkenY(&planeOutY, stream);
        if (err != RGY_ERR_NONE) { AddMessage(RGY_LOG_ERROR, _T("kaizen: darken failed: %s.\n"), get_err_mes(err)); return err; }
    }
    if (m_doThin) {
        err = runThinY(&planeOutY, stream);
        if (err != RGY_ERR_NONE) { AddMessage(RGY_LOG_ERROR, _T("kaizen: thin failed: %s.\n"), get_err_mes(err)); return err; }
    }
    if (m_denoise != VppKaizenDenoise::Off) {
        err = runDenoiseY(&planeOutY, stream);
        if (err != RGY_ERR_NONE) { AddMessage(RGY_LOG_ERROR, _T("kaizen: denoise failed: %s.\n"), get_err_mes(err)); return err; }
    }
    if (m_clampHighlights) {
        err = runClampHighlightsY(&planeOutY, &planeInY, stream);
        if (err != RGY_ERR_NONE) { AddMessage(RGY_LOG_ERROR, _T("kaizen: clamp_highlights failed: %s.\n"), get_err_mes(err)); return err; }
    }
    if (m_antiring > 0.0f) {
        err = runAntiringY(&planeOutY, &planeInY, stream);
        if (err != RGY_ERR_NONE) { AddMessage(RGY_LOG_ERROR, _T("kaizen: antiring failed: %s.\n"), get_err_mes(err)); return err; }
    }

    if (!m_postResize) {
        ppOutputFrames[0] = coreFrame;
        *pOutputFrameNum = 1;
        return RGY_ERR_NONE;
    }
    RGYFrameInfo *resizeOut[1] = { nullptr };
    int resizeNum = 0;
    auto rerr = m_postResize->filter(coreFrame, resizeOut, &resizeNum, stream);
    if (rerr != RGY_ERR_NONE) { AddMessage(RGY_LOG_ERROR, _T("kaizen: end-of-chain resize failed: %s.\n"), get_err_mes(rerr)); return rerr; }
    ppOutputFrames[0] = resizeOut[0];
    *pOutputFrameNum = 1;
    return RGY_ERR_NONE;
}

void NVEncFilterKaizen::close() {
    m_postResize.reset();
    m_scratchA.reset();
    m_scratchB.reset();
    m_clampStatsH.reset();
    m_clampStats.reset();
    m_chromaLumaLowres.reset();
    m_prefilterPlane.reset();
    m_prefilterRef.reset();
    m_dtdSrcLuma.reset();
    m_frameBuf.clear();
}
