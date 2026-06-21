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

#include <map>
#include <array>
#include <vector>
#define _USE_MATH_DEFINES
#include <cmath>
#include "convert_csp.h"
#include "nis_coef_tables.h"
#include "NVEncFilter.h"
#include "NVEncFilterNvvfx.h"
#include "NVEncFilterNGX.h"
#include "NVEncFilterLibplacebo.h"
#include "rgy_prm.h"
#pragma warning (push)
#pragma warning (disable: 4819)
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#pragma warning (pop)

#if defined(_WIN32) || defined(_WIN64)
const TCHAR *NPPI_DLL_NAME_TSTR = NVENC_NPPI_DLL_NAME_TSTR;
const TCHAR *NVRTC_DLL_NAME_TSTR = NVENC_NVRTC_DLL_NAME_TSTR;
const TCHAR *NVRTC_BUILTIN_DLL_NAME_TSTR = NVENC_NVRTC_BUILTIN_DLL_NAME_TSTR;
#else //#if defined(_WIN32) || defined(_WIN64)
const TCHAR *NPPI_DLL_NAME_TSTR = _T("libnppc.so");
const TCHAR *NVRTC_DLL_NAME_TSTR = _T("libnvrtc.so");
const TCHAR *NVRTC_BUILTIN_DLL_NAME_TSTR = _T("");
#endif //#if defined(_WIN32) || defined(_WIN64)

static const int RESIZE_BLOCK_X = 32;
static const int RESIZE_BLOCK_Y = 8;
static_assert(RESIZE_BLOCK_Y <= RESIZE_BLOCK_X, "RESIZE_BLOCK_Y <= RESIZE_BLOCK_X");

#if (!defined(_M_IX86))
static const auto RGY_VPP_RESIZE_ALGO_TO_NPPI = make_array<std::pair<RGY_VPP_RESIZE_ALGO, NppiInterpolationMode>>(
    std::make_pair(RGY_VPP_RESIZE_NPPI_INTER_NN,                 NPPI_INTER_NN),
    std::make_pair(RGY_VPP_RESIZE_NPPI_INTER_LINEAR,             NPPI_INTER_LINEAR),
    std::make_pair(RGY_VPP_RESIZE_NPPI_INTER_CUBIC,              NPPI_INTER_CUBIC),
    std::make_pair(RGY_VPP_RESIZE_NPPI_INTER_CUBIC2P_BSPLINE,    NPPI_INTER_CUBIC2P_BSPLINE),
    std::make_pair(RGY_VPP_RESIZE_NPPI_INTER_CUBIC2P_CATMULLROM, NPPI_INTER_CUBIC2P_CATMULLROM),
    std::make_pair(RGY_VPP_RESIZE_NPPI_INTER_CUBIC2P_B05C03,     NPPI_INTER_CUBIC2P_B05C03),
    std::make_pair(RGY_VPP_RESIZE_NPPI_INTER_SUPER,              NPPI_INTER_SUPER),
    std::make_pair(RGY_VPP_RESIZE_NPPI_INTER_LANCZOS,            NPPI_INTER_LANCZOS),
    std::make_pair(RGY_VPP_RESIZE_NPPI_INTER_LANCZOS3_ADVANCED,  NPPI_INTER_LANCZOS3_ADVANCED),
    std::make_pair(RGY_VPP_RESIZE_NPPI_SMOOTH_EDGE,              NPPI_SMOOTH_EDGE)
    );

MAP_PAIR_0_1(vpp_resize_algo, rgy, RGY_VPP_RESIZE_ALGO, enc, NppiInterpolationMode, RGY_VPP_RESIZE_ALGO_TO_NPPI, RGY_VPP_RESIZE_UNKNOWN, NPPI_INTER_UNDEFINED);
#endif

enum RESIZE_WEIGHT_TYPE {
    WEIGHT_UNKNOWN,
    WEIGHT_LANCZOS,
    WEIGHT_SPLINE,
    WEIGHT_BICUBIC,
    WEIGHT_BICUBIC_MITCHELL,    // tunable bicubic preset B=1/3, C=1/3
    WEIGHT_BICUBIC_CATMULL_ROM, // tunable bicubic preset B=0,   C=1/2
    WEIGHT_BICUBIC_HERMITE,     // tunable bicubic preset B=0,   C=0
    WEIGHT_BILINEAR
};

// User-tunable B/C for algo=bicubic (set per-resize via cudaMemcpyToSymbolAsync on
// the stream before launch). Defaults match the previous hardcoded bicubic.
__constant__ float g_bicubicBC[2] = { 0.0f, 0.6f }; // defaults match FILTER_DEFAULT_RESIZE_BICUBIC_B/C

template<typename TypePixel>
cudaError_t setTexFieldResize(cudaTextureObject_t& texSrc, const RGYFrameInfo* pFrame, cudaTextureFilterMode filterMode, cudaTextureReadMode readMode, int normalizedCord) {
    texSrc = 0;

    cudaResourceDesc resDescSrc;
    memset(&resDescSrc, 0, sizeof(resDescSrc));
    resDescSrc.resType = cudaResourceTypePitch2D;
    resDescSrc.res.pitch2D.desc = cudaCreateChannelDesc<TypePixel>();
    resDescSrc.res.pitch2D.pitchInBytes = pFrame->pitch[0];
    resDescSrc.res.pitch2D.width = pFrame->width;
    resDescSrc.res.pitch2D.height = pFrame->height;
    resDescSrc.res.pitch2D.devPtr = (uint8_t*)pFrame->ptr[0];

    cudaTextureDesc texDescSrc;
    memset(&texDescSrc, 0, sizeof(texDescSrc));
    texDescSrc.addressMode[0]   = cudaAddressModeClamp;
    texDescSrc.addressMode[1]   = cudaAddressModeClamp;
    texDescSrc.filterMode       = filterMode;
    texDescSrc.readMode         = readMode;
    texDescSrc.normalizedCoords = normalizedCord;

    return cudaCreateTextureObject(&texSrc, &resDescSrc, &texDescSrc, nullptr);
}

template<typename Type, int bit_depth>
__global__ void kernel_resize_texture(uint8_t *__restrict__ pDst, const int dstPitch, const int dstWidth, const int dstHeight,
    cudaTextureObject_t texObj,
    const float ratioX, const float ratioY) {
    const int ix = blockIdx.x * blockDim.x + threadIdx.x;
    const int iy = blockIdx.y * blockDim.y + threadIdx.y;
    if (ix < dstWidth && iy < dstHeight) {
        const float x = (float)ix + 0.5f;
        const float y = (float)iy + 0.5f;

        Type *ptr = (Type *)(pDst + iy * dstPitch + ix * sizeof(Type));
        ptr[0] = (Type)(tex2D<float>(texObj, x * ratioX, y * ratioY) * (float)((1<<bit_depth)-1));
    }
}

template<typename Type, int bit_depth>
void resize_texture(uint8_t *pDst, const int dstPitch, const int dstWidth, const int dstHeight, cudaTextureObject_t texObj, const float ratioX, const float ratioY, cudaStream_t stream) {
    dim3 blockSize(32, 8);
    dim3 gridSize(divCeil(dstWidth, blockSize.x), divCeil(dstHeight, blockSize.y));
    kernel_resize_texture<Type, bit_depth><<<gridSize, blockSize, 0, stream>>>(pDst, dstPitch, dstWidth, dstHeight, texObj, ratioX, ratioY);
}

template<typename Type, int bit_depth>
RGY_ERR resize_texture_plane(RGYFrameInfo *pOutputFrame, const RGYFrameInfo *pInputFrame, RGY_VPP_RESIZE_ALGO interp, cudaStream_t stream) {
    const float ratioX = 1.0f / (float)(pOutputFrame->width);
    const float ratioY = 1.0f / (float)(pOutputFrame->height);

    cudaTextureObject_t texSrc = 0;
    auto cudaerr = cudaSuccess;
    if ((cudaerr = setTexFieldResize<Type>(texSrc, pInputFrame, (interp == RGY_VPP_RESIZE_BILINEAR) ? cudaFilterModeLinear : cudaFilterModePoint, cudaReadModeNormalizedFloat, 1)) != cudaSuccess) {
        return err_to_rgy(cudaerr);
    }
    resize_texture<Type, bit_depth>((uint8_t *)pOutputFrame->ptr[0],
        pOutputFrame->pitch[0], pOutputFrame->width, pOutputFrame->height,
        texSrc, ratioX, ratioY, stream);
    cudaerr = cudaGetLastError();
    if (cudaerr != cudaSuccess) {
        return err_to_rgy(cudaerr);
    }
    cudaerr = cudaDestroyTextureObject(texSrc);
    if (cudaerr != cudaSuccess) {
        return err_to_rgy(cudaerr);
    }
    return RGY_ERR_NONE;
}

template<typename Type, int bit_depth>
static RGY_ERR resize_texture_frame(RGYFrameInfo* pOutputFrame, const RGYFrameInfo* pInputFrame, RGY_VPP_RESIZE_ALGO interp, cudaStream_t stream) {
    const auto planeSrcY = getPlane(pInputFrame, RGY_PLANE_Y);
    const auto planeSrcU = getPlane(pInputFrame, RGY_PLANE_U);
    const auto planeSrcV = getPlane(pInputFrame, RGY_PLANE_V);
    const auto planeSrcA = getPlane(pInputFrame, RGY_PLANE_A);
    auto planeOutputY = getPlane(pOutputFrame, RGY_PLANE_Y);
    auto planeOutputU = getPlane(pOutputFrame, RGY_PLANE_U);
    auto planeOutputV = getPlane(pOutputFrame, RGY_PLANE_V);
    auto planeOutputA = getPlane(pOutputFrame, RGY_PLANE_A);

    auto sts = resize_texture_plane<Type, bit_depth>(&planeOutputY, &planeSrcY, interp, stream);
    if (sts != RGY_ERR_NONE) {
        return sts;
    }
    sts = resize_texture_plane<Type, bit_depth>(&planeOutputU, &planeSrcU, interp, stream);
    if (sts != RGY_ERR_NONE) {
        return sts;
    }
    sts = resize_texture_plane<Type, bit_depth>(&planeOutputV, &planeSrcV, interp, stream);
    if (sts != RGY_ERR_NONE) {
        return sts;
    }
    if (RGY_CSP_PLANES[pOutputFrame->csp] == 4) {
        sts = resize_texture_plane<Type, bit_depth>(&planeOutputA, &planeSrcA, interp, stream);
        if (sts != RGY_ERR_NONE) {
            return sts;
        }
    }
    return sts;
}


template<int radius>
__inline__ __device__
float factor_bilinear(const float x) {
    if (fabs(x) >= (float)radius) return 0.0f;
    return 1.0f - fabs(x) * (1.0f / radius);
}

template<int radius>
__inline__ __device__
float factor_bicubic(float x, float B, float C) {
    x = fabs(x);
    if (x >= (float)radius) return 0.0f;
    const float x2 = x * x;
    const float x3 = x2 * x;
    if (x <= 1.0f) {
        return (2.0f - 1.5f * B - 1.0f * C) * x3 +
            (-3.0f + 2.0f * B + 1.0f * C) * x2 +
            (1.0f - (2.0f / 6.0f) * B);
    } else {
        return (-(1.0f / 6.0f) * B - 1.0f * C) * x3 +
            (1.0f * B + 5.0f * C) * x2 +
            (-2.0f * B - 8.0f * C) * x +
            ((8.0f / 6.0f) * B + 4.0f * C);
    }
}

__inline__ __device__
float sinc(float x) {
    const float pi = (float)M_PI;
    const float pi_x = pi * x;
    return __sinf(pi_x) / pi_x;
}

template<int radius>
__inline__ __device__
float factor_lanczos(const float x) {
    if (fabs(x) >= (float)radius) return 0.0f;
    if (x == 0.0f) return 1.0f;
    return sinc(x) * sinc(x * (1.0f / radius));
}

template<int radius>
__inline__ __device__
float factor_spline(const float x_raw, const float *psCopyFactor) {
    const float x = fabs(x_raw);
    if (x >= (float)radius) return 0.0f;
    const float4 weight = ((const float4 *)psCopyFactor)[min((int)x, radius - 1)];
    //重みを計算
    float w = weight.w;
    w += x * weight.z;
    const float x2 = x * x;
    w += x2 * weight.y;
    w += x2 * x * weight.x;
    return w;
}

template<int radius, RESIZE_WEIGHT_TYPE algo>
void __inline__ __device__ calc_weight(
    float *pWeight, const float srcPos, const int srcFirst, const int srcEnd, const float ratioClamped, const float *psCopyFactor) {
    float *pW = pWeight;
    for (int i = srcFirst; i <= srcEnd; i++, pW++) {
        const float delta = ((i + 0.5f) - srcPos) * ratioClamped;
        float weight = 0.0f;
        switch (algo) {
        case WEIGHT_LANCZOS:  weight = factor_lanczos<radius>(delta); break;
        case WEIGHT_SPLINE:   weight = factor_spline<radius>(delta, psCopyFactor); break;
        case WEIGHT_BICUBIC:  weight = factor_bicubic<radius>(delta, g_bicubicBC[0], g_bicubicBC[1]); break;
        case WEIGHT_BICUBIC_MITCHELL:    weight = factor_bicubic<radius>(delta, 1.0f / 3.0f, 1.0f / 3.0f); break;
        case WEIGHT_BICUBIC_CATMULL_ROM: weight = factor_bicubic<radius>(delta, 0.0f, 0.5f); break;
        case WEIGHT_BICUBIC_HERMITE:     weight = factor_bicubic<radius>(delta, 0.0f, 0.0f); break;
        case WEIGHT_BILINEAR: weight = factor_bilinear<radius>(delta); break;
        default:
            break;
        }
        pW[0] = weight;
    }
}

// 参考:  rgba-image/lanczos
template<typename Type, int bit_depth, RESIZE_WEIGHT_TYPE algo, int radius, int block_x, int block_y>
__global__ void kernel_resize(uint8_t *__restrict__ pDst, const int dstPitch, const int dstWidth, const int dstHeight,
    const uint8_t *__restrict__ pSrc, const int srcPitch, const int srcWidth, const int srcHeight,
    const float ratioX, const float ratioY, const float *__restrict__ pgFactor,
    const int shared_weightXdim, const int shared_weightYdim) {
    const float ratioInvX = 1.0f / ratioX;
    const float ratioClampedX = min(ratioX, 1.0f);
    const float srcWindowX = radius / ratioClampedX;

    const float ratioInvY = 1.0f / ratioY;
    const float ratioClampedY = min(ratioY, 1.0f);
    const float srcWindowY = radius / ratioClampedY;

    //重みをsharedメモリにコピー
    alignas(128) extern __shared__ float shared[];
    float *weightXshared = shared;
    float *weightYshared = weightXshared + shared_weightXdim * block_x;
    float *psCopyFactor  = weightYshared + shared_weightYdim * block_y;
    // 事前計算した srcFirst/srcEnd を shared に保持して使用側で再計算を避ける
    int *metaBase = (int*)(psCopyFactor + ((algo == WEIGHT_SPLINE) ? radius * 4 : 0));
    int *srcFirstXArr = metaBase;
    int *srcEndXArr   = srcFirstXArr + block_x;
    int *srcFirstYArr = srcEndXArr + block_x;
    int *srcEndYArr   = srcFirstYArr + block_y;
    static_assert(block_x % 4 == 0 && block_y % 4 == 0, "block_x & block_y must be able to be divided by 4 to ensure psCopyFactor can be accessed by float4.");

    if (algo == WEIGHT_SPLINE) {
        if (threadIdx.y == 0) {
            static_assert(algo != WEIGHT_SPLINE || radius * 4 < block_x, "radius * 4 < block_x");
            if (threadIdx.x < radius * 4) {
                psCopyFactor[threadIdx.x] = pgFactor[threadIdx.x];
            }
        }
        __syncthreads();
    }

    if (threadIdx.y == 0) {
        // threadIdx.y==0のスレッドが、x方向の重みをそれぞれ計算してsharedメモリに書き込み
        const int dstX = blockIdx.x * block_x + threadIdx.x;
        const float srcX = ((float)(dstX + 0.5f)) * ratioInvX;
        const int srcFirstX = max(0, (int)floorf(srcX - srcWindowX));
        const int srcEndX = min(srcWidth - 1, (int)ceilf(srcX + srcWindowX));
        srcFirstXArr[threadIdx.x] = srcFirstX;
        srcEndXArr[threadIdx.x]   = srcEndX;
        calc_weight<radius, algo>(weightXshared + threadIdx.x * shared_weightXdim, srcX, srcFirstX, srcEndX, ratioClampedX, psCopyFactor);

        if (threadIdx.x < block_y) {
            // threadIdx.y==0のスレッドが、y方向の重みをそれぞれ計算してsharedメモリに書き込み
            const int thready = threadIdx.x;
            const int dstY = blockIdx.y * block_y + thready;
            const float srcY = ((float)(dstY + 0.5f)) * ratioInvY;
            const int srcFirstY = max(0, (int)floorf(srcY - srcWindowY));
            const int srcEndY = min(srcHeight - 1, (int)ceilf(srcY + srcWindowY));
            srcFirstYArr[thready] = srcFirstY;
            srcEndYArr[thready]   = srcEndY;
            calc_weight<radius, algo>(weightYshared + thready * shared_weightYdim, srcY, srcFirstY, srcEndY, ratioClampedY, psCopyFactor);
        }
    }

    __syncthreads();

    const int ix = blockIdx.x * block_x + threadIdx.x;
    const int iy = blockIdx.y * block_y + threadIdx.y;
    if (ix < dstWidth && iy < dstHeight) {
        //ピクセルの中心を算出してからスケール
        //const float x = ((float)ix + 0.5f) * ratioX;
        //const float y = ((float)iy + 0.5f) * ratioY;

        const int srcFirstX = srcFirstXArr[threadIdx.x];
        const int srcEndX = srcEndXArr[threadIdx.x];
        const float *weightX = weightXshared + threadIdx.x * shared_weightXdim;

        const int srcFirstY = srcFirstYArr[threadIdx.y];
        const int srcEndY = srcEndYArr[threadIdx.y];
        const float *weightY = weightYshared + threadIdx.y * shared_weightYdim;

        const uint8_t *srcLine = pSrc + srcFirstY * srcPitch + srcFirstX * sizeof(Type);
        float clr = 0.0f;
        float weightSum = 0.0f;
        for (int j = srcFirstY; j <= srcEndY; j++, weightY++, srcLine += srcPitch) {
            const float wy = weightY[0];
            const float *pwx = weightX;
            const Type *srcPtr = (Type*)srcLine;
            for (int i = srcFirstX; i <= srcEndX; i++, pwx++, srcPtr++) {
                const float weight = pwx[0] * wy;
                clr += srcPtr[0] * weight;
                weightSum += weight;
            }
        }

		Type* ptr = (Type*)(pDst + iy * dstPitch + ix * sizeof(Type));
		ptr[0] = (Type)clamp(clr / weightSum, 0.0f, (1 << bit_depth) - 0.1f);
    }
}

template<typename Type, int bit_depth, RESIZE_WEIGHT_TYPE algo, int radius>
void resize_plane(uint8_t *pDst, const int dstPitch, const int dstWidth, const int dstHeight,
    const uint8_t *pSrc, const int srcPitch, const int srcWidth, const int srcHeight,
    const float *pgFactor, cudaStream_t stream) {
    const float ratioX = (float)(dstWidth) / srcWidth;
    const float ratioClampedX = min(ratioX, 1.0f);
    const float srcWindowX = radius / ratioClampedX;

    // 最大寄与数は n_max = 2 * ceil(srcWindow) + 2。
    // さらに float4 アクセス時の整列を保証するため、4 の倍数に切り上げる。
    const int shared_weightXdim_raw = (((int)ceilf(srcWindowX) + 1) * 2);
    const int shared_weightXdim = (shared_weightXdim_raw + 3) & ~3;
    const int shared_weightX = RESIZE_BLOCK_X * shared_weightXdim;

    const float ratioY = (float)(dstHeight) / srcHeight;
    const float ratioClampedY = min(ratioY, 1.0f);
    const float srcWindowY = radius / ratioClampedY;

    // Y 方向も同様に 4 の倍数に切り上げ
    const int shared_weightYdim_raw = (((int)ceilf(srcWindowY) + 1) * 2);
    const int shared_weightYdim = (shared_weightYdim_raw + 3) & ~3;
    const int shared_weightY = RESIZE_BLOCK_Y * shared_weightYdim;

    const int meta_ints = (RESIZE_BLOCK_X * 2) + (RESIZE_BLOCK_Y * 2);
    const int shared_size_byte = (shared_weightX + shared_weightY + ((algo == WEIGHT_SPLINE) ? radius * 4 : 0)) * sizeof(float)
        + meta_ints * (int)sizeof(int);

    dim3 blockSize(RESIZE_BLOCK_X, RESIZE_BLOCK_Y);
    dim3 gridSize(divCeil(dstWidth, blockSize.x), divCeil(dstHeight, blockSize.y));
    kernel_resize<Type, bit_depth, algo, radius, RESIZE_BLOCK_X, RESIZE_BLOCK_Y><<<gridSize, blockSize, shared_size_byte, stream>>>(
        pDst, dstPitch, dstWidth, dstHeight,
        pSrc, srcPitch, srcWidth, srcHeight,
        ratioX, ratioY, pgFactor, shared_weightXdim, shared_weightYdim);
}

template<typename Type, int bit_depth, RESIZE_WEIGHT_TYPE algo, int radius>
static RGY_ERR resize_plane(RGYFrameInfo *pOutputPlane, const RGYFrameInfo *pInputPlane, const float *pgFactor, cudaStream_t stream) {
    resize_plane<Type, bit_depth, algo, radius>(
        (uint8_t*)pOutputPlane->ptr[0], pOutputPlane->pitch[0], pOutputPlane->width, pOutputPlane->height,
        (uint8_t*)pInputPlane->ptr[0], pInputPlane->pitch[0], pInputPlane->width, pInputPlane->height,
        pgFactor, stream);
    auto cudaerr = cudaGetLastError();
    if (cudaerr != cudaSuccess) {
        return err_to_rgy(cudaerr);
    }
    return RGY_ERR_NONE;
}

template<typename Type, int bit_depth, RESIZE_WEIGHT_TYPE algo, int radius>
static RGY_ERR resize_frame(RGYFrameInfo *pOutputFrame, const RGYFrameInfo *pInputFrame, const float *pgFactor, cudaStream_t stream) {
    for (int iplane = 0; iplane < RGY_CSP_PLANES[pInputFrame->csp]; iplane++) {
        const auto plane = (RGY_PLANE)iplane;
        const auto planeSrc = getPlane(pInputFrame, plane);
        auto planeOutput = getPlane(pOutputFrame, plane);
        auto sts = resize_plane<Type, bit_depth, algo, radius>(&planeOutput, &planeSrc, pgFactor, stream);
        if (sts != RGY_ERR_NONE) {
            return sts;
        }
    }
    return RGY_ERR_NONE;
}

static constexpr float FSR_RCAS_LIMIT = 0.25f - (1.0f / 16.0f);

__inline__ __device__
float fsr_clamp(const float x, const float low, const float high) {
    return fminf(fmaxf(x, low), high);
}

template<typename Type, int bit_depth>
__inline__ __device__
float fsr_load_norm(const uint8_t *pSrc, const int srcPitch, int x, int y, const int w, const int h) {
    x = min(max(x, 0), w - 1);
    y = min(max(y, 0), h - 1);
    const auto val = *(const Type *)(pSrc + y * srcPitch + x * sizeof(Type));
    return (float)val * (1.0f / (float)((1 << bit_depth) - 1));
}

__inline__ __device__
void fsr_easu_set(
    float *dirX, float *dirY, float *lenAcc,
    const float2 pp,
    const int biS, const int biT, const int biU, const int biV,
    const float lA, const float lB, const float lC, const float lD, const float lE) {
    float w = 0.0f;
    if (biS) w = (1.0f - pp.x) * (1.0f - pp.y);
    if (biT) w =          pp.x * (1.0f - pp.y);
    if (biU) w = (1.0f - pp.x) *          pp.y;
    if (biV) w =          pp.x *          pp.y;

    const float dc = lD - lC;
    const float cb = lC - lB;
    const float lenX_inv = fmaxf(fabsf(dc), fabsf(cb));
    const float rcpX = (lenX_inv > 0.0f) ? (1.0f / lenX_inv) : 0.0f;
    const float dirXv = lD - lB;
    *dirX += dirXv * w;
    float lenX = fsr_clamp(fabsf(dirXv) * rcpX, 0.0f, 1.0f);
    lenX *= lenX;
    *lenAcc += lenX * w;

    const float ec = lE - lC;
    const float ca = lC - lA;
    const float lenY_inv = fmaxf(fabsf(ec), fabsf(ca));
    const float rcpY = (lenY_inv > 0.0f) ? (1.0f / lenY_inv) : 0.0f;
    const float dirYv = lE - lA;
    *dirY += dirYv * w;
    float lenY = fsr_clamp(fabsf(dirYv) * rcpY, 0.0f, 1.0f);
    lenY *= lenY;
    *lenAcc += lenY * w;
}

__inline__ __device__
void fsr_easu_tap(float *aC, float *aW, const float2 off, const float2 dir, const float2 len2, const float lob, const float clp, const float c) {
    float vx = off.x * ( dir.x) + off.y * dir.y;
    float vy = off.x * (-dir.y) + off.y * dir.x;
    vx *= len2.x;
    vy *= len2.y;
    float d2 = vx * vx + vy * vy;
    d2 = fminf(d2, clp);
    float wB = 0.4f * d2 - 1.0f;
    float wA = lob   * d2 - 1.0f;
    wB *= wB;
    wA *= wA;
    wB = (25.0f / 16.0f) * wB - (25.0f / 16.0f - 1.0f);
    const float w = wB * wA;
    *aC += c * w;
    *aW += w;
}

template<typename Type, int bit_depth>
__global__ void kernel_fsr1_easu(uint8_t *__restrict__ pDst, const int dstPitch, const int dstWidth, const int dstHeight,
    const uint8_t *__restrict__ pSrc, const int srcPitch, const int srcWidth, const int srcHeight,
    const float ratioInvX, const float ratioInvY, const float offsetX, const float offsetY) {
    const int ix = blockIdx.x * blockDim.x + threadIdx.x;
    const int iy = blockIdx.y * blockDim.y + threadIdx.y;
    if (ix >= dstWidth || iy >= dstHeight) {
        return;
    }

    const float ppx = (float)ix * ratioInvX + offsetX;
    const float ppy = (float)iy * ratioInvY + offsetY;
    const int fpx = (int)floorf(ppx);
    const int fpy = (int)floorf(ppy);
    const float2 pp = make_float2(ppx - (float)fpx, ppy - (float)fpy);

    const float b = fsr_load_norm<Type, bit_depth>(pSrc, srcPitch, fpx + 0, fpy - 1, srcWidth, srcHeight);
    const float c = fsr_load_norm<Type, bit_depth>(pSrc, srcPitch, fpx + 1, fpy - 1, srcWidth, srcHeight);
    const float e = fsr_load_norm<Type, bit_depth>(pSrc, srcPitch, fpx - 1, fpy + 0, srcWidth, srcHeight);
    const float f = fsr_load_norm<Type, bit_depth>(pSrc, srcPitch, fpx + 0, fpy + 0, srcWidth, srcHeight);
    const float g = fsr_load_norm<Type, bit_depth>(pSrc, srcPitch, fpx + 1, fpy + 0, srcWidth, srcHeight);
    const float h = fsr_load_norm<Type, bit_depth>(pSrc, srcPitch, fpx + 2, fpy + 0, srcWidth, srcHeight);
    const float i = fsr_load_norm<Type, bit_depth>(pSrc, srcPitch, fpx - 1, fpy + 1, srcWidth, srcHeight);
    const float j = fsr_load_norm<Type, bit_depth>(pSrc, srcPitch, fpx + 0, fpy + 1, srcWidth, srcHeight);
    const float k = fsr_load_norm<Type, bit_depth>(pSrc, srcPitch, fpx + 1, fpy + 1, srcWidth, srcHeight);
    const float l = fsr_load_norm<Type, bit_depth>(pSrc, srcPitch, fpx + 2, fpy + 1, srcWidth, srcHeight);
    const float n = fsr_load_norm<Type, bit_depth>(pSrc, srcPitch, fpx + 0, fpy + 2, srcWidth, srcHeight);
    const float o = fsr_load_norm<Type, bit_depth>(pSrc, srcPitch, fpx + 1, fpy + 2, srcWidth, srcHeight);

    float dirX = 0.0f, dirY = 0.0f, lenAcc = 0.0f;
    fsr_easu_set(&dirX, &dirY, &lenAcc, pp, 1, 0, 0, 0, b, e, f, g, j);
    fsr_easu_set(&dirX, &dirY, &lenAcc, pp, 0, 1, 0, 0, c, f, g, h, k);
    fsr_easu_set(&dirX, &dirY, &lenAcc, pp, 0, 0, 1, 0, f, i, j, k, n);
    fsr_easu_set(&dirX, &dirY, &lenAcc, pp, 0, 0, 0, 1, g, j, k, l, o);

    const float dirR = dirX * dirX + dirY * dirY;
    const int zro = (dirR < (1.0f / 32768.0f));
    const float invR = zro ? 1.0f : rsqrtf(dirR);
    const float ndx = zro ? 1.0f : (dirX * invR);
    const float ndy = zro ? 0.0f : (dirY * invR);

    float len = lenAcc * 0.5f;
    len *= len;
    const float stretch_num = ndx * ndx + ndy * ndy;
    const float stretch_den = fmaxf(fabsf(ndx), fabsf(ndy));
    const float stretch = stretch_num * ((stretch_den > 0.0f) ? (1.0f / stretch_den) : 0.0f);
    const float2 len2 = make_float2(1.0f + (stretch - 1.0f) * len, 1.0f - 0.5f * len);
    const float lob = 0.5f + ((1.0f / 4.0f) - 0.04f - 0.5f) * len;
    const float clp = (lob > 0.0f) ? (1.0f / lob) : 0.0f;

    const float mn4 = fminf(fminf(fminf(f, g), j), k);
    const float mx4 = fmaxf(fmaxf(fmaxf(f, g), j), k);

    const float2 dir = make_float2(ndx, ndy);
    float aC = 0.0f, aW = 0.0f;
    fsr_easu_tap(&aC, &aW, make_float2( 0.0f - pp.x, -1.0f - pp.y), dir, len2, lob, clp, b);
    fsr_easu_tap(&aC, &aW, make_float2( 1.0f - pp.x, -1.0f - pp.y), dir, len2, lob, clp, c);
    fsr_easu_tap(&aC, &aW, make_float2(-1.0f - pp.x,  1.0f - pp.y), dir, len2, lob, clp, i);
    fsr_easu_tap(&aC, &aW, make_float2( 0.0f - pp.x,  1.0f - pp.y), dir, len2, lob, clp, j);
    fsr_easu_tap(&aC, &aW, make_float2( 0.0f - pp.x,  0.0f - pp.y), dir, len2, lob, clp, f);
    fsr_easu_tap(&aC, &aW, make_float2(-1.0f - pp.x,  0.0f - pp.y), dir, len2, lob, clp, e);
    fsr_easu_tap(&aC, &aW, make_float2( 1.0f - pp.x,  1.0f - pp.y), dir, len2, lob, clp, k);
    fsr_easu_tap(&aC, &aW, make_float2( 2.0f - pp.x,  1.0f - pp.y), dir, len2, lob, clp, l);
    fsr_easu_tap(&aC, &aW, make_float2( 2.0f - pp.x,  0.0f - pp.y), dir, len2, lob, clp, h);
    fsr_easu_tap(&aC, &aW, make_float2( 1.0f - pp.x,  0.0f - pp.y), dir, len2, lob, clp, g);
    fsr_easu_tap(&aC, &aW, make_float2( 1.0f - pp.x,  2.0f - pp.y), dir, len2, lob, clp, o);
    fsr_easu_tap(&aC, &aW, make_float2( 0.0f - pp.x,  2.0f - pp.y), dir, len2, lob, clp, n);

    const float pix = (aW > 0.0f) ? (aC * (1.0f / aW)) : f;
    const float dered = fminf(mx4, fmaxf(mn4, pix));
    const float clamped = fsr_clamp(dered, 0.0f, 1.0f);
    auto ptr = (Type *)(pDst + iy * dstPitch + ix * sizeof(Type));
    ptr[0] = (Type)(clamped * (float)((1 << bit_depth) - 1));
}

template<typename Type, int bit_depth>
__global__ void kernel_fsr1_rcas(uint8_t *__restrict__ pDst, const int dstPitch, const int dstWidth, const int dstHeight,
    const uint8_t *__restrict__ pSrc, const int srcPitch, const float con0_sharp) {
    const int ix = blockIdx.x * blockDim.x + threadIdx.x;
    const int iy = blockIdx.y * blockDim.y + threadIdx.y;
    if (ix >= dstWidth || iy >= dstHeight) {
        return;
    }

    const float bV = fsr_load_norm<Type, bit_depth>(pSrc, srcPitch, ix,     iy - 1, dstWidth, dstHeight);
    const float dV = fsr_load_norm<Type, bit_depth>(pSrc, srcPitch, ix - 1, iy,     dstWidth, dstHeight);
    const float eV = fsr_load_norm<Type, bit_depth>(pSrc, srcPitch, ix,     iy,     dstWidth, dstHeight);
    const float fV = fsr_load_norm<Type, bit_depth>(pSrc, srcPitch, ix + 1, iy,     dstWidth, dstHeight);
    const float hV = fsr_load_norm<Type, bit_depth>(pSrc, srcPitch, ix,     iy + 1, dstWidth, dstHeight);

    const float mn4 = fminf(fminf(fminf(bV, dV), fV), hV);
    const float mx4 = fmaxf(fmaxf(fmaxf(bV, dV), fV), hV);

    const float rcpMx4 = (mx4 > 0.0f) ? (1.0f / (4.0f * mx4)) : 0.0f;
    const float rcpMn4 = 1.0f / (4.0f * mn4 - 4.0f);
    const float hitMin = fminf(mn4, eV) * rcpMx4;
    const float hitMax = (1.0f - fmaxf(mx4, eV)) * rcpMn4;
    float lobe = fmaxf(-hitMin, hitMax);
    lobe = fmaxf(-FSR_RCAS_LIMIT, fminf(lobe, 0.0f)) * con0_sharp;

    const float rcpL = 1.0f / (4.0f * lobe + 1.0f);
    const float pix = (lobe * (bV + dV + fV + hV) + eV) * rcpL;

    const float result = fsr_clamp(pix, 0.0f, 1.0f);
    auto ptr = (Type *)(pDst + iy * dstPitch + ix * sizeof(Type));
    ptr[0] = (Type)(result * (float)((1 << bit_depth) - 1));
}

template<typename Type, int bit_depth>
static RGY_ERR resize_fsr1_plane(RGYFrameInfo *pOutputPlane, const RGYFrameInfo *pInputPlane, RGYFrameInfo *pMidPlane, const float sharpness, cudaStream_t stream) {
    const float ratioInvX = (float)pInputPlane->width / (float)pOutputPlane->width;
    const float ratioInvY = (float)pInputPlane->height / (float)pOutputPlane->height;
    const float offsetX = 0.5f * ratioInvX - 0.5f;
    const float offsetY = 0.5f * ratioInvY - 0.5f;
    const float sharpness_user = clamp(sharpness, 0.0f, 1.0f);
    const float stops = (1.0f - sharpness_user) * 4.0f;
    const float con0_sharp = exp2f(-stops);

    dim3 blockSize(RESIZE_BLOCK_X, RESIZE_BLOCK_Y);
    dim3 gridSize(divCeil(pOutputPlane->width, blockSize.x), divCeil(pOutputPlane->height, blockSize.y));
    kernel_fsr1_easu<Type, bit_depth><<<gridSize, blockSize, 0, stream>>>(
        (uint8_t *)pMidPlane->ptr[0], pMidPlane->pitch[0], pMidPlane->width, pMidPlane->height,
        (const uint8_t *)pInputPlane->ptr[0], pInputPlane->pitch[0], pInputPlane->width, pInputPlane->height,
        ratioInvX, ratioInvY, offsetX, offsetY);
    auto cudaerr = cudaGetLastError();
    if (cudaerr != cudaSuccess) {
        return err_to_rgy(cudaerr);
    }

    kernel_fsr1_rcas<Type, bit_depth><<<gridSize, blockSize, 0, stream>>>(
        (uint8_t *)pOutputPlane->ptr[0], pOutputPlane->pitch[0], pOutputPlane->width, pOutputPlane->height,
        (const uint8_t *)pMidPlane->ptr[0], pMidPlane->pitch[0],
        con0_sharp);
    cudaerr = cudaGetLastError();
    if (cudaerr != cudaSuccess) {
        return err_to_rgy(cudaerr);
    }
    return RGY_ERR_NONE;
}

template<typename Type, int bit_depth>
static RGY_ERR resize_fsr1_frame(RGYFrameInfo *pOutputFrame, const RGYFrameInfo *pInputFrame, RGYFrameInfo *pMidFrame, const float sharpness, cudaStream_t stream) {
    for (int iplane = 0; iplane < RGY_CSP_PLANES[pInputFrame->csp]; iplane++) {
        const auto plane = (RGY_PLANE)iplane;
        const auto planeSrc = getPlane(pInputFrame, plane);
        auto planeMid = getPlane(pMidFrame, plane);
        auto planeOutput = getPlane(pOutputFrame, plane);
        auto sts = resize_fsr1_plane<Type, bit_depth>(&planeOutput, &planeSrc, &planeMid, sharpness, stream);
        if (sts != RGY_ERR_NONE) {
            return sts;
        }
    }
    return RGY_ERR_NONE;
}

static bool useTextureBilinear(const RGY_VPP_RESIZE_ALGO interp, const RGYFrameInfo *pOutputFrame, const RGYFrameInfo *pInputFrame) {
    return interp == RGY_VPP_RESIZE_NEAREST
       || (interp == RGY_VPP_RESIZE_BILINEAR
            && pOutputFrame->width > pInputFrame->width
            && pOutputFrame->height > pInputFrame->height);
}

template<typename Type, int bit_depth>
static RGY_ERR resize_frame(RGYFrameInfo *pOutputFrame, const RGYFrameInfo *pInputFrame, RGY_VPP_RESIZE_ALGO interp, const float *pgFactor, cudaStream_t stream) {
    if (useTextureBilinear(interp, pOutputFrame, pInputFrame)) {
        return resize_texture_frame<Type, bit_depth>(pOutputFrame, pInputFrame, interp, stream);
    }
    switch (interp) {
    case RGY_VPP_RESIZE_BILINEAR: return resize_frame<Type, bit_depth, WEIGHT_BILINEAR, 1>(pOutputFrame, pInputFrame, pgFactor, stream);
    case RGY_VPP_RESIZE_BICUBIC:  return resize_frame<Type, bit_depth, WEIGHT_BICUBIC,  2>(pOutputFrame, pInputFrame, pgFactor, stream);
    case RGY_VPP_RESIZE_SPLINE16: return resize_frame<Type, bit_depth, WEIGHT_SPLINE,   2>(pOutputFrame, pInputFrame, pgFactor, stream);
    case RGY_VPP_RESIZE_SPLINE36: return resize_frame<Type, bit_depth, WEIGHT_SPLINE,   3>(pOutputFrame, pInputFrame, pgFactor, stream);
    case RGY_VPP_RESIZE_SPLINE64: return resize_frame<Type, bit_depth, WEIGHT_SPLINE,   4>(pOutputFrame, pInputFrame, pgFactor, stream);
    case RGY_VPP_RESIZE_LANCZOS2: return resize_frame<Type, bit_depth, WEIGHT_LANCZOS,  2>(pOutputFrame, pInputFrame, pgFactor, stream);
    case RGY_VPP_RESIZE_LANCZOS3: return resize_frame<Type, bit_depth, WEIGHT_LANCZOS,  3>(pOutputFrame, pInputFrame, pgFactor, stream);
    case RGY_VPP_RESIZE_LANCZOS4: return resize_frame<Type, bit_depth, WEIGHT_LANCZOS,  4>(pOutputFrame, pInputFrame, pgFactor, stream);
    case RGY_VPP_RESIZE_LANCZOS5: return resize_frame<Type, bit_depth, WEIGHT_LANCZOS,  5>(pOutputFrame, pInputFrame, pgFactor, stream);
    case RGY_VPP_RESIZE_LANCZOS6: return resize_frame<Type, bit_depth, WEIGHT_LANCZOS,  6>(pOutputFrame, pInputFrame, pgFactor, stream);
    case RGY_VPP_RESIZE_LANCZOS7: return resize_frame<Type, bit_depth, WEIGHT_LANCZOS,  7>(pOutputFrame, pInputFrame, pgFactor, stream);
    case RGY_VPP_RESIZE_LANCZOS8: return resize_frame<Type, bit_depth, WEIGHT_LANCZOS,  8>(pOutputFrame, pInputFrame, pgFactor, stream);
    case RGY_VPP_RESIZE_MITCHELL:    return resize_frame<Type, bit_depth, WEIGHT_BICUBIC_MITCHELL,    2>(pOutputFrame, pInputFrame, pgFactor, stream);
    case RGY_VPP_RESIZE_CATMULL_ROM: return resize_frame<Type, bit_depth, WEIGHT_BICUBIC_CATMULL_ROM, 2>(pOutputFrame, pInputFrame, pgFactor, stream);
    case RGY_VPP_RESIZE_HERMITE:     return resize_frame<Type, bit_depth, WEIGHT_BICUBIC_HERMITE,     2>(pOutputFrame, pInputFrame, pgFactor, stream);
    default:  return RGY_ERR_NONE;
    }
}

// --------- JINC (windowed-jinc EWA) resampler ---------
// Ports the LUT-based circular EWA scheme from Asd-g/AviSynth-JincResize (MIT).
// jinc(x) = 2 * J1(pi*x) / (pi*x); jinc(0) = 1. The windowed-jinc weight per
// squared-radius is precomputed on the host into a 1024-entry LUT (built once per
// radius, uploaded to device); the kernel does a single non-separable circular
// gather. radius = tap (3/4/6/8 for jinc36/64/144/256).
static const int JINC_LUT_SIZE = 1024;
static const double JINC_ZERO_SQR_D = 1.48759464366204680005356; // (first zero of J1(pi*x)/x)^2

// Bessel J1 (Abramowitz & Stegun 9.4.4 / 9.4.6).
static double rgy_bessel_j1(double x) {
    const double ax = fabs(x);
    if (ax < 8.0) {
        const double y = x * x;
        const double num = x * (72362614232.0 + y * (-7895059235.0 + y * (242396853.1
            + y * (-2972611.439 + y * (15704.48260 + y * (-30.16036606))))));
        const double den = 144725228442.0 + y * (2300535178.0 + y * (18583304.74
            + y * (99447.43394 + y * (376.9991397 + y * 1.0))));
        return num / den;
    } else {
        const double z = 8.0 / ax;
        const double y = z * z;
        const double p1 = 1.0 + y * (0.183105e-2 + y * (-0.3516396496e-4
            + y * (0.2457520174e-5 + y * (-0.240337019e-6))));
        const double p2 = 0.04687499995 + y * (-0.2002690873e-3 + y * (0.8449199096e-5
            + y * (-0.88228987e-6 + y * 0.105787412e-6)));
        const double ans = sqrt(0.636619772 / ax)
            * (cos(ax - 2.356194491) * p1 - z * sin(ax - 2.356194491) * p2);
        return (x < 0.0) ? -ans : ans;
    }
}
static double rgy_jinc(double r) {
    if (r == 0.0) return 1.0;
    const double pix = M_PI * r;
    return 2.0 * rgy_bessel_j1(pix) / pix;
}
std::vector<float> buildJincLut(const int radius) {
    const double tap2 = (double)radius * (double)radius;
    const double win_scale = sqrt(JINC_ZERO_SQR_D / tap2);
    std::vector<float> lut(JINC_LUT_SIZE);
    for (int i = 0; i < JINC_LUT_SIZE; i++) {
        const double r2 = (double)i * tap2 / (double)(JINC_LUT_SIZE - 1);
        const double r  = sqrt(r2);
        const double w  = (r >= (double)radius) ? 0.0 : (rgy_jinc(r) * rgy_jinc(win_scale * r));
        lut[i] = (float)w;
    }
    return lut;
}
int jincRadius(const RGY_VPP_RESIZE_ALGO interp) {
    switch (interp) {
    case RGY_VPP_RESIZE_JINC36:  return 3;
    case RGY_VPP_RESIZE_JINC64:  return 4;
    case RGY_VPP_RESIZE_JINC144: return 6;
    case RGY_VPP_RESIZE_JINC256: return 8;
    default: return 0;
    }
}
bool isJincResize(const RGY_VPP_RESIZE_ALGO interp) {
    return interp == RGY_VPP_RESIZE_JINC36 || interp == RGY_VPP_RESIZE_JINC64
        || interp == RGY_VPP_RESIZE_JINC144 || interp == RGY_VPP_RESIZE_JINC256;
}

#define JINC_BLOCK_X 32
#define JINC_BLOCK_Y 8

template<typename Type, int bit_depth, int radius>
__global__ void kernel_resize_jinc(uint8_t *__restrict__ pDst, const int dstPitch, const int dstWidth, const int dstHeight,
    const uint8_t *__restrict__ pSrc, const int srcPitch, const int srcWidth, const int srcHeight,
    const float ratioX, const float ratioY, const float *__restrict__ lut) {
    const int ix = blockIdx.x * blockDim.x + threadIdx.x;
    const int iy = blockIdx.y * blockDim.y + threadIdx.y;
    if (ix >= dstWidth || iy >= dstHeight) return;

    const float ratioInvX = 1.0f / ratioX;
    const float ratioInvY = 1.0f / ratioY;
    const float fx = ((float)ix + 0.5f) * ratioInvX - 0.5f;
    const float fy = ((float)iy + 0.5f) * ratioInvY - 0.5f;
    const int sx0 = (int)floorf(fx) - radius + 1;
    const int sy0 = (int)floorf(fy) - radius + 1;

    const float tap2 = (float)(radius * radius);
    const float lut_scale = (float)(JINC_LUT_SIZE - 1) / tap2;

    float sum = 0.0f;
    float wsum = 0.0f;
    for (int dy = 0; dy < 2 * radius; dy++) {
        const int sy_raw = sy0 + dy;
        const float dyf = (float)sy_raw - fy;
        const float dy2 = dyf * dyf;
        if (dy2 >= tap2) continue;
        const int sy = min(max(sy_raw, 0), srcHeight - 1);
        const Type *srcRow = (const Type *)(pSrc + sy * srcPitch);
        for (int dx = 0; dx < 2 * radius; dx++) {
            const int sx_raw = sx0 + dx;
            const float dxf = (float)sx_raw - fx;
            const float r2 = dxf * dxf + dy2;
            if (r2 >= tap2) continue;
            const int sx = min(max(sx_raw, 0), srcWidth - 1);
            int li = (int)(r2 * lut_scale + 0.5f);
            if (li >= JINC_LUT_SIZE) li = JINC_LUT_SIZE - 1;
            const float w = lut[li];
            sum  += w * (float)srcRow[sx];
            wsum += w;
        }
    }
    const float v = (wsum > 0.0f) ? (sum / wsum) : 0.0f;
    Type *ptr = (Type *)(pDst + iy * dstPitch + ix * sizeof(Type));
    ptr[0] = (Type)clamp(v, 0.0f, (float)((1 << bit_depth) - 1) - 0.1f);
}

template<typename Type, int bit_depth, int radius>
static RGY_ERR resize_jinc_plane(RGYFrameInfo *pOutputPlane, const RGYFrameInfo *pInputPlane, const float *lut, cudaStream_t stream) {
    dim3 blockSize(JINC_BLOCK_X, JINC_BLOCK_Y);
    dim3 gridSize(divCeil(pOutputPlane->width, blockSize.x), divCeil(pOutputPlane->height, blockSize.y));
    const float ratioX = (float)pOutputPlane->width / pInputPlane->width;
    const float ratioY = (float)pOutputPlane->height / pInputPlane->height;
    kernel_resize_jinc<Type, bit_depth, radius><<<gridSize, blockSize, 0, stream>>>(
        (uint8_t *)pOutputPlane->ptr[0], pOutputPlane->pitch[0], pOutputPlane->width, pOutputPlane->height,
        (const uint8_t *)pInputPlane->ptr[0], pInputPlane->pitch[0], pInputPlane->width, pInputPlane->height,
        ratioX, ratioY, lut);
    return err_to_rgy(cudaGetLastError());
}

template<typename Type, int bit_depth>
static RGY_ERR resize_jinc_frame(RGYFrameInfo *pOutputFrame, const RGYFrameInfo *pInputFrame, RGY_VPP_RESIZE_ALGO interp, const float *lut, cudaStream_t stream) {
    for (int iplane = 0; iplane < RGY_CSP_PLANES[pInputFrame->csp]; iplane++) {
        const auto planeInput = getPlane(pInputFrame, (RGY_PLANE)iplane);
        auto planeOutput = getPlane(pOutputFrame, (RGY_PLANE)iplane);
        RGY_ERR sts = RGY_ERR_NONE;
        switch (interp) {
        case RGY_VPP_RESIZE_JINC36:  sts = resize_jinc_plane<Type, bit_depth, 3>(&planeOutput, &planeInput, lut, stream); break;
        case RGY_VPP_RESIZE_JINC64:  sts = resize_jinc_plane<Type, bit_depth, 4>(&planeOutput, &planeInput, lut, stream); break;
        case RGY_VPP_RESIZE_JINC144: sts = resize_jinc_plane<Type, bit_depth, 6>(&planeOutput, &planeInput, lut, stream); break;
        case RGY_VPP_RESIZE_JINC256: sts = resize_jinc_plane<Type, bit_depth, 8>(&planeOutput, &planeInput, lut, stream); break;
        default: return RGY_ERR_UNSUPPORTED;
        }
        if (sts != RGY_ERR_NONE) return sts;
    }
    return RGY_ERR_NONE;
}

// --------- NIS (NVIDIA Image Scaling) resampler ---------
// Ports the simplified separable NIS scaler: a 6-tap polyphase base resample
// (coef_scale LUT) plus a band-gated unsharp-mask high-pass (coef_usm LUT),
// from NVIDIA NIS v1.0.3 (MIT, tables in nis_coef_tables.h). Single-stage,
// covers upscale ratios up to 2x (kScale in [0.5, 1.0]). sharpness in [0,1]
// (default 0.5) controls the USM strength/limit; hdrMode 0=SDR (1/2 = HDR
// bands) follows NVScalerUpdateConfig.
#define NIS_BLOCK_X 32
#define NIS_BLOCK_Y 8

struct NISParams {
    float kScaleX, kScaleY;
    float kSharpStartY, kSharpScaleY;
    float kSharpStrengthMin, kSharpStrengthScale;
    float kSharpLimitMin, kSharpLimitScale;
};

// Port of NIS_Config.h NVScalerUpdateConfig (the 8 fields the scaler uses).
static NISParams nisBuildParams(float sharpness, int hdrMode, int inW, int inH, int outW, int outH) {
    sharpness = fminf(fmaxf(sharpness, 0.0f), 1.0f);
    const float slider = sharpness - 0.5f;
    const float MaxScale   = (slider >= 0.0f) ? 1.25f : 1.75f;
    const float MinScale   = (slider >= 0.0f) ? 1.25f : 1.0f;
    const float LimitScale = (slider >= 0.0f) ? 1.25f : 1.0f;
    float startY = 0.45f, endY = 0.9f;
    float strMin = fmaxf(0.0f, 0.4f + slider * MinScale * 1.2f);
    float strMax = 1.6f + slider * MaxScale * 1.8f;
    float limMin = fmaxf(0.1f, 0.14f + slider * LimitScale * 0.32f);
    float limMax = 0.5f + slider * LimitScale * 0.6f;
    if (hdrMode == 1 || hdrMode == 2) {
        strMin = fmaxf(0.0f, 0.4f + slider * MinScale * 1.1f);
        strMax = 2.2f + slider * MaxScale * 1.8f;
        limMin = fmaxf(0.06f, 0.10f + slider * LimitScale * 0.28f);
        limMax = 0.6f + slider * LimitScale * 0.6f;
        if (hdrMode == 2) { startY = 0.35f; endY = 0.55f; } else { startY = 0.3f; endY = 0.5f; }
    }
    NISParams p;
    p.kScaleX = (float)inW / outW;
    p.kScaleY = (float)inH / outH;
    p.kSharpStartY = startY;
    p.kSharpScaleY = 1.0f / (endY - startY);
    p.kSharpStrengthMin = strMin;
    p.kSharpStrengthScale = strMax - strMin;
    p.kSharpLimitMin = limMin;
    p.kSharpLimitScale = limMax - limMin;
    return p;
}

template<typename Type, int bit_depth>
__device__ __forceinline__ float nis_sample(const uint8_t *src, int srcPitch, int srcW, int srcH, int x, int y) {
    x = (x < 0) ? 0 : ((x >= srcW) ? (srcW - 1) : x);
    y = (y < 0) ? 0 : ((y >= srcH) ? (srcH - 1) : y);
    const Type *row = (const Type *)(src + y * srcPitch);
    const float maxv = (float)((1 << bit_depth) - 1);
    return (float)row[x] / maxv;
}

// 6-tap separable polyphase apply. coefTable is the flattened 64x8 LUT (taps
// 0..5 carry weight). Pass coef_scale for the base resample, coef_usm for USM.
template<typename Type, int bit_depth>
__device__ __forceinline__ float nis_polyphase_apply(const uint8_t *src, int srcPitch, int srcW, int srcH,
    float kScaleX, float kScaleY, const float *coefTable, int dx, int dy) {
    const float sx = ((float)dx + 0.5f) * kScaleX - 0.5f;
    const float sy = ((float)dy + 0.5f) * kScaleY - 0.5f;
    const int isx_base = (int)floorf(sx);
    const int isy_base = (int)floorf(sy);
    const float fx = sx - (float)isx_base;
    const float fy = sy - (float)isy_base;
    int phase_x = (int)(fx * 64.0f); phase_x = min(max(phase_x, 0), 63);
    int phase_y = (int)(fy * 64.0f); phase_y = min(max(phase_y, 0), 63);
    const float *wx = coefTable + phase_x * 8;
    const float *wy = coefTable + phase_y * 8;
    float result = 0.0f;
    #pragma unroll
    for (int j = 0; j < 6; j++) {
        const int sy_int = isy_base + (j - 2);
        float rowsum = 0.0f;
        #pragma unroll
        for (int i = 0; i < 6; i++) {
            rowsum += nis_sample<Type, bit_depth>(src, srcPitch, srcW, srcH, isx_base + (i - 2), sy_int) * wx[i];
        }
        result += rowsum * wy[j];
    }
    return result;
}

template<typename Type, int bit_depth>
__global__ void kernel_nis_scaler(uint8_t *__restrict__ pDst, int dstPitch, int dstWidth, int dstHeight,
    const uint8_t *__restrict__ pSrc, int srcPitch, int srcWidth, int srcHeight,
    NISParams prm, const float *__restrict__ coefScale, const float *__restrict__ coefUsm) {
    const int dx = blockIdx.x * blockDim.x + threadIdx.x;
    const int dy = blockIdx.y * blockDim.y + threadIdx.y;
    if (dx >= dstWidth || dy >= dstHeight) return;

    const float y_base = nis_polyphase_apply<Type, bit_depth>(pSrc, srcPitch, srcWidth, srcHeight, prm.kScaleX, prm.kScaleY, coefScale, dx, dy);
    const float y_usm  = nis_polyphase_apply<Type, bit_depth>(pSrc, srcPitch, srcWidth, srcHeight, prm.kScaleX, prm.kScaleY, coefUsm, dx, dy);

    float t = (y_base - prm.kSharpStartY) * prm.kSharpScaleY;
    t = clamp(t, 0.0f, 1.0f);
    const float strength = prm.kSharpStrengthMin + t * prm.kSharpStrengthScale;
    const float limit    = prm.kSharpLimitMin    + t * prm.kSharpLimitScale;
    const float usm_clamped = clamp(y_usm, -limit, limit);
    const float result = y_base + strength * usm_clamped;

    const float maxv = (float)((1 << bit_depth) - 1);
    Type *ptr = (Type *)(pDst + dy * dstPitch + dx * sizeof(Type));
    ptr[0] = (Type)(clamp(result * maxv, 0.0f, maxv) + 0.5f);
}

template<typename Type, int bit_depth>
static RGY_ERR resize_nis_plane(RGYFrameInfo *pOut, const RGYFrameInfo *pIn, NISParams prm, const float *coefScale, const float *coefUsm, cudaStream_t stream) {
    dim3 block(NIS_BLOCK_X, NIS_BLOCK_Y);
    dim3 grid(divCeil(pOut->width, block.x), divCeil(pOut->height, block.y));
    prm.kScaleX = (float)pIn->width / pOut->width;   // per-plane scale (same ratio as luma)
    prm.kScaleY = (float)pIn->height / pOut->height;
    kernel_nis_scaler<Type, bit_depth><<<grid, block, 0, stream>>>(
        (uint8_t *)pOut->ptr[0], pOut->pitch[0], pOut->width, pOut->height,
        (const uint8_t *)pIn->ptr[0], pIn->pitch[0], pIn->width, pIn->height,
        prm, coefScale, coefUsm);
    return err_to_rgy(cudaGetLastError());
}

template<typename Type, int bit_depth>
static RGY_ERR resize_nis_frame(RGYFrameInfo *pOut, const RGYFrameInfo *pIn, NISParams prm, const float *coefScale, const float *coefUsm, cudaStream_t stream) {
    for (int i = 0; i < RGY_CSP_PLANES[pIn->csp]; i++) {
        const auto in = getPlane(pIn, (RGY_PLANE)i);
        auto out = getPlane(pOut, (RGY_PLANE)i);
        auto sts = resize_nis_plane<Type, bit_depth>(&out, &in, prm, coefScale, coefUsm, stream);
        if (sts != RGY_ERR_NONE) return sts;
    }
    return RGY_ERR_NONE;
}

static bool isNisResize(const RGY_VPP_RESIZE_ALGO interp) { return interp == RGY_VPP_RESIZE_NIS; }

template<typename T, typename Tfunc>
static RGY_ERR resize_nppi_plane_call_func(RGYFrameInfo *pOutputPlane, const RGYFrameInfo *pInputPlane, const Tfunc funcResize, const NppiInterpolationMode interpMode, cudaStream_t stream) {
    const double factorX = pOutputPlane->width / (double)pInputPlane->width;
    const double factorY = pOutputPlane->height / (double)pInputPlane->height;
    auto srcSize = nppisize(pInputPlane);
    auto srcRect = nppiroi(pInputPlane);
    auto dstRect = nppiroi(pOutputPlane);
#if CUDA_VERSION >= 13000
	NppStreamContext nppStreamCtx = {};
	nppStreamCtx.hStream = stream;
	int dev = 0;
	cudaGetDevice(&dev);
	nppStreamCtx.nCudaDeviceId = dev;
	cudaDeviceProp prop = {};
	cudaGetDeviceProperties(&prop, dev);
	nppStreamCtx.nMultiProcessorCount = prop.multiProcessorCount;
	nppStreamCtx.nMaxThreadsPerMultiProcessor = prop.maxThreadsPerMultiProcessor;
	nppStreamCtx.nMaxThreadsPerBlock = prop.maxThreadsPerBlock;
	nppStreamCtx.nSharedMemPerBlock = prop.sharedMemPerBlock;
#else
    (void)stream;
#endif
    NppStatus sts = funcResize(
        (const T *)pInputPlane->ptr[0],
        srcSize, pInputPlane->pitch[0], srcRect,
        (T *)pOutputPlane->ptr[0],
        pOutputPlane->pitch[0], dstRect,
		factorX, factorY, 0.0, 0.0, interpMode
#if CUDA_VERSION >= 13000
		, nppStreamCtx
#endif
    );
    if (sts != NPP_SUCCESS) {
        return err_to_rgy(sts);
    }
    return RGY_ERR_NONE;
}

static RGY_ERR resize_nppi_plane(RGYFrameInfo *pOutputPlane, const RGYFrameInfo *pInputPlane, const NppiInterpolationMode interpMode, cudaStream_t stream) {
    auto sts = RGY_ERR_NONE;
    if (RGY_CSP_DATA_TYPE[pInputPlane->csp] == RGY_DATA_TYPE_U8) {
#if CUDA_VERSION >= 13000
		sts = resize_nppi_plane_call_func<Npp8u>(pOutputPlane, pInputPlane, nppiResizeSqrPixel_8u_C1R_Ctx, interpMode, stream);
#else
		sts = resize_nppi_plane_call_func<Npp8u>(pOutputPlane, pInputPlane, nppiResizeSqrPixel_8u_C1R, interpMode, stream);
#endif
    } else if (RGY_CSP_DATA_TYPE[pInputPlane->csp] == RGY_DATA_TYPE_U16) {
#if CUDA_VERSION >= 13000
		sts = resize_nppi_plane_call_func<Npp16u>(pOutputPlane, pInputPlane, nppiResizeSqrPixel_16u_C1R_Ctx, interpMode, stream);
#else
		sts = resize_nppi_plane_call_func<Npp16u>(pOutputPlane, pInputPlane, nppiResizeSqrPixel_16u_C1R, interpMode, stream);
#endif
    } else {
        sts = RGY_ERR_UNSUPPORTED;
    }
    return sts;
}

static RGY_ERR resize_nppi_frame(RGYFrameInfo *pOutputFrame, const RGYFrameInfo *pInputFrame, const NppiInterpolationMode interpMode, cudaStream_t stream) {
    for (int iplane = 0; iplane < RGY_CSP_PLANES[pInputFrame->csp]; iplane++) {
        const auto plane = (RGY_PLANE)iplane;
        const auto planeInput = getPlane(pInputFrame, plane);
        auto planeOutput = getPlane(pOutputFrame, plane);
		auto sts = resize_nppi_plane(&planeOutput, &planeInput, interpMode, stream);
        if (sts != RGY_ERR_NONE) {
            return sts;
        }
    }
    return RGY_ERR_NONE;
}

RGY_ERR NVEncFilterResize::resizeNppi(RGYFrameInfo *pOutputFrame, const RGYFrameInfo *pInputFrame, cudaStream_t stream) {
#if _M_IX86
    AddMessage(RGY_LOG_ERROR, _T("npp filter not supported on x86.\n"));
    return RGY_ERR_UNSUPPORTED;
#else
    RGY_ERR sts = RGY_ERR_NONE;
    if (m_param->frameOut.csp != m_param->frameIn.csp) {
        AddMessage(RGY_LOG_ERROR, _T("csp does not match.\n"));
        return RGY_ERR_INVALID_PARAM;
    }
    auto pResizeParam = std::dynamic_pointer_cast<NVEncFilterParamResize>(m_param);
    if (!pResizeParam) {
        AddMessage(RGY_LOG_ERROR, _T("Invalid parameter type.\n"));
        return RGY_ERR_INVALID_PARAM;
    }
    const auto interp = vpp_resize_algo_rgy_to_enc(pResizeParam->interp);
    if (interp == NPPI_INTER_UNDEFINED) {
        AddMessage(RGY_LOG_ERROR, _T("Unknown nppi interp mode: %d.\n"), (int)pResizeParam->interp);
        return RGY_ERR_UNSUPPORTED;
    }
	sts = resize_nppi_frame(pOutputFrame, pInputFrame, interp, stream);
    if (sts != RGY_ERR_NONE) {
        AddMessage(RGY_LOG_ERROR, _T("failed to resize: %d, %s.\n"), sts, get_err_mes(sts));
        return sts;
    }
    return sts;
#endif
}

template<typename T, typename Tfunc>
static RGY_ERR resize_nppi_yuv444(RGYFrameInfo *pOutputFrame, const RGYFrameInfo *pInputFrame, Tfunc funcResize, NppiInterpolationMode interpMode, cudaStream_t stream) {
    const double factorX = pOutputFrame->width / (double)pInputFrame->width;
    const double factorY = pOutputFrame->height / (double)pInputFrame->height;
    auto srcSize = nppisize(pInputFrame);
    auto srcRect = nppiroi(pInputFrame);
    auto dstRect = nppiroi(pOutputFrame);
    const auto planeSrcY = getPlane(pInputFrame, RGY_PLANE_Y);
    const auto planeSrcU = getPlane(pInputFrame, RGY_PLANE_U);
    const auto planeSrcV = getPlane(pInputFrame, RGY_PLANE_V);
    auto planeOutputY = getPlane(pOutputFrame, RGY_PLANE_Y);
    auto planeOutputU = getPlane(pOutputFrame, RGY_PLANE_U);
    auto planeOutputV = getPlane(pOutputFrame, RGY_PLANE_V);
    const T *pSrc[3] = {
        (const T *)planeSrcY.ptr[0],
        (const T *)planeSrcU.ptr[0],
        (const T *)planeSrcV.ptr[0]
    };
    T *pDst[3] = {
        (T *)planeOutputY.ptr[0],
        (T *)planeOutputU.ptr[0],
        (T *)planeOutputV.ptr[0]
    };
#if CUDA_VERSION >= 13000
	NppStreamContext nppStreamCtx = {};
	nppStreamCtx.hStream = stream;
	int dev = 0;
	cudaGetDevice(&dev);
	nppStreamCtx.nCudaDeviceId = dev;
	cudaDeviceProp prop = {};
	cudaGetDeviceProperties(&prop, dev);
	nppStreamCtx.nMultiProcessorCount = prop.multiProcessorCount;
	nppStreamCtx.nMaxThreadsPerMultiProcessor = prop.maxThreadsPerMultiProcessor;
	nppStreamCtx.nMaxThreadsPerBlock = prop.maxThreadsPerBlock;
	nppStreamCtx.nSharedMemPerBlock = prop.sharedMemPerBlock;
#else
    (void)stream;
#endif
	NppStatus sts = funcResize(
        pSrc,
        srcSize, planeSrcY.pitch[0], srcRect,
        pDst,
        planeOutputY.pitch[0], dstRect,
		factorX, factorY, 0.0, 0.0, interpMode
#if CUDA_VERSION >= 13000
		, nppStreamCtx
#endif
    );
    if (sts != NPP_SUCCESS) {
        return err_to_rgy(sts);
    }
    return RGY_ERR_NONE;
}

RGY_ERR NVEncFilterResize::resizeNppiYUV444(RGYFrameInfo *pOutputFrame, const RGYFrameInfo *pInputFrame, cudaStream_t stream) {
#if _M_IX86
    AddMessage(RGY_LOG_ERROR, _T("npp filter not supported on x86.\n"));
    return RGY_ERR_UNSUPPORTED;
#else
    RGY_ERR sts = RGY_ERR_NONE;
    if (m_param->frameOut.csp != m_param->frameIn.csp) {
        AddMessage(RGY_LOG_ERROR, _T("csp does not match.\n"));
        return RGY_ERR_INVALID_PARAM;
    }
    if (rgy_csp_has_alpha(m_param->frameIn.csp)) {
        AddMessage(RGY_LOG_ERROR, _T("unsupported, NVEncFilterResize::resizeNppi() should be called.\n"));
        sts = RGY_ERR_UNSUPPORTED;
    }
    auto pResizeParam = std::dynamic_pointer_cast<NVEncFilterParamResize>(m_param);
    if (!pResizeParam) {
        AddMessage(RGY_LOG_ERROR, _T("Invalid parameter type.\n"));
        return RGY_ERR_INVALID_PARAM;
    }
    const auto interp = vpp_resize_algo_rgy_to_enc(pResizeParam->interp);
    if (interp == NPPI_INTER_UNDEFINED) {
        AddMessage(RGY_LOG_ERROR, _T("Unknown nppi interp mode: %d.\n"), (int)pResizeParam->interp);
        return RGY_ERR_UNSUPPORTED;
    }
	if (RGY_CSP_DATA_TYPE[m_param->frameIn.csp] == RGY_DATA_TYPE_U8) {
#if CUDA_VERSION >= 13000
		sts = resize_nppi_yuv444<Npp8u>(pOutputFrame, pInputFrame, nppiResizeSqrPixel_8u_P3R_Ctx, interp, stream);
#else
		sts = resize_nppi_yuv444<Npp8u>(pOutputFrame, pInputFrame, nppiResizeSqrPixel_8u_P3R, interp, stream);
#endif
        if (sts != RGY_ERR_NONE) {
            AddMessage(RGY_LOG_ERROR, _T("failed to resize: %d, %s.\n"), sts, get_err_mes(sts));
            return sts;
        }
	} else if (RGY_CSP_DATA_TYPE[m_param->frameIn.csp] == RGY_DATA_TYPE_U16) {
#if CUDA_VERSION >= 13000
		sts = resize_nppi_yuv444<Npp16u>(pOutputFrame, pInputFrame, nppiResizeSqrPixel_16u_P3R_Ctx, interp, stream);
#else
		sts = resize_nppi_yuv444<Npp16u>(pOutputFrame, pInputFrame, nppiResizeSqrPixel_16u_P3R, interp, stream);
#endif
        if (sts != RGY_ERR_NONE) {
            AddMessage(RGY_LOG_ERROR, _T("failed to resize: %d, %s.\n"), sts, get_err_mes(sts));
            return sts;
        }
    } else {
        AddMessage(RGY_LOG_ERROR, _T("unsupported csp.\n"));
        sts = RGY_ERR_UNSUPPORTED;
    }
    return sts;
#endif
}

NVEncFilterResize::NVEncFilterResize() :
    m_bInterlacedWarn(false),
    m_weightSpline(),
    m_weightSplineAlgo(RGY_VPP_RESIZE_UNKNOWN),
    m_fsr1Easu(),
    m_weightJinc(),
    m_weightJincAlgo(RGY_VPP_RESIZE_UNKNOWN),
    m_nisCoefScale(),
    m_nisCoefUsm(),
    m_nisStages(1),
    m_nisCascadeInter(),
    m_nvvfxSuperRes(),
    m_ngxVSR(),
    m_libplaceboResample() {
    m_name = _T("resize");
}

NVEncFilterResize::~NVEncFilterResize() {
    close();
}

RGY_ERR NVEncFilterResize::initNvvfxFilter(NVEncFilterParamResize *param) {
    const tstring filter_name = get_cx_desc(list_vpp_resize, param->interp);
    const double target_scale_ratio_min = std::min(
        param->frameOut.width / (double)param->frameIn.width,
        param->frameOut.height / (double)param->frameIn.height);
    const double target_scale_ratio_max = std::max(
        param->frameOut.width / (double)param->frameIn.width,
        param->frameOut.height / (double)param->frameIn.height);
    if (target_scale_ratio_max < 1.0) {
        AddMessage(RGY_LOG_ERROR, _T("%s not supported for resize ratio below 1.0.\n"), filter_name.c_str());
        return RGY_ERR_UNSUPPORTED;
    }
    if (m_nvvfxSuperRes
        // モード変更には再初期化が必要
        && std::dynamic_pointer_cast<NVEncFilterParamResize>(m_param)->nvvfxSuperRes->nvvfxSuperRes.mode == param->nvvfxSuperRes->nvvfxSuperRes.mode
        && m_param->frameIn.width == param->frameIn.width
        && m_param->frameIn.height == param->frameIn.height
        && m_param->frameOut.width == param->frameOut.width
        && m_param->frameOut.height == param->frameOut.height) {
        auto newParam = param->nvvfxSuperRes->nvvfxSuperRes;
        auto oldParam = std::dynamic_pointer_cast<NVEncFilterParamResize>(m_param);
        param->nvvfxSuperRes = oldParam->nvvfxSuperRes;
        param->nvvfxSuperRes->nvvfxSuperRes = newParam;
        auto sts = m_nvvfxSuperRes->init(param->nvvfxSuperRes, m_pLog);
        if (sts != RGY_ERR_NONE) {
            return sts;
        }
        return RGY_ERR_NONE;
    }
    static const auto nvvfx_scale_ratio = make_array<rgy_rational<int>>(
        rgy_rational<int>(4, 3),
        rgy_rational<int>(3, 2),
        rgy_rational<int>(2, 1),
        rgy_rational<int>(3, 1),
        rgy_rational<int>(4, 1)
    );
    std::vector<bool> ratio_checked(nvvfx_scale_ratio.size(), false);
    // 入力側からチェックする
    // allow_upscale_after_nvvfx は nvvfxの処理後に拡大リサイズを許可するかどうか?
    for (const bool allow_upscale_after_nvvfx : { false, true }) {
        for (int iratio = 0; iratio < (int)nvvfx_scale_ratio.size(); iratio++) {
            const auto ratio = nvvfx_scale_ratio[iratio];
            const double ratiod = ratio.qdouble();
            if ((param->frameIn.height * ratio.n()) % ratio.d() != 0) {
                continue; // 割り切れない場合は使用しない
            }
            if (!ratio_checked[iratio]
                && (ratiod >= ((allow_upscale_after_nvvfx) ? target_scale_ratio_min : target_scale_ratio_max) * (1.0 - 1e-3)
                 || ratio == nvvfx_scale_ratio.back())) {
                ratio_checked[iratio] = true;
                unique_ptr<NVEncFilterNvvfxSuperRes> filter(new NVEncFilterNvvfxSuperRes());
                param->nvvfxSuperRes->frameIn = param->frameIn;
                param->nvvfxSuperRes->frameOut = param->frameIn;
                param->nvvfxSuperRes->frameOut.width = param->frameIn.width * ratio.n() / ratio.d();
                param->nvvfxSuperRes->frameOut.height = param->frameIn.height * ratio.n() / ratio.d();
                param->nvvfxSuperRes->baseFps = param->baseFps;
                param->nvvfxSuperRes->frameIn.mem_type = RGY_MEM_TYPE_GPU;
                param->nvvfxSuperRes->frameOut.mem_type = RGY_MEM_TYPE_GPU;
                param->nvvfxSuperRes->bOutOverwrite = false;
                auto sts = filter->init(param->nvvfxSuperRes, m_pLog);
                if (sts == RGY_ERR_NONE) {
                    m_nvvfxSuperRes = std::move(filter);
                    AddMessage(RGY_LOG_DEBUG, _T("created %s with ratio %.1f (%dx%d -> %dx%d).\n"), m_nvvfxSuperRes->GetInputMessage().c_str(), ratio.qdouble(),
                        param->frameIn.width, param->frameIn.height, param->nvvfxSuperRes->frameOut.width, param->nvvfxSuperRes->frameOut.height);
                    return RGY_ERR_NONE;
                }
                AddMessage(RGY_LOG_WARN, _T("Failed to init nvvfx-superres with ratio %.1f (%dx%d -> %dx%d), retrying with other ratios...\n"), ratio.qdouble(),
                    param->frameIn.width, param->frameIn.height, param->nvvfxSuperRes->frameOut.width, param->nvvfxSuperRes->frameOut.height);
            }
        }
    }
    // 出力側からチェックする(倍率は逆順にチェック)
    for (int iratio = (int)nvvfx_scale_ratio.size() - 1; iratio >= 0; iratio--) {
        const auto ratio = nvvfx_scale_ratio[iratio];
        if ((param->frameOut.width * ratio.d()) % ratio.n() != 0 || (param->frameOut.height * ratio.d()) % ratio.n() != 0) {
            continue; // 割り切れない場合は使用しない
        }
        const int inWidth = param->frameOut.width * ratio.d() / ratio.n();
        const int inHeight = param->frameOut.height * ratio.d() / ratio.n();
        if (inWidth * inHeight < param->frameIn.width * param->frameIn.height) {
            continue; // 入力サイズが小さい場合は使用しない
        }
        unique_ptr<NVEncFilterNvvfxSuperRes> filter(new NVEncFilterNvvfxSuperRes());
        param->nvvfxSuperRes->frameIn = param->frameOut;
        param->nvvfxSuperRes->frameIn.width = inWidth;
        param->nvvfxSuperRes->frameIn.height = inHeight;
        param->nvvfxSuperRes->frameOut = param->frameOut;
        param->nvvfxSuperRes->baseFps = param->baseFps;
        param->nvvfxSuperRes->frameIn.mem_type = RGY_MEM_TYPE_GPU;
        param->nvvfxSuperRes->frameOut.mem_type = RGY_MEM_TYPE_GPU;
        param->nvvfxSuperRes->bOutOverwrite = false;
        auto sts = filter->init(param->nvvfxSuperRes, m_pLog);
        if (sts == RGY_ERR_NONE) {
            m_nvvfxSuperRes = std::move(filter);
            AddMessage(RGY_LOG_DEBUG, _T("created %s with ratio %.1f (%dx%d -> %dx%d).\n"), m_nvvfxSuperRes->GetInputMessage().c_str(), ratio.qdouble(),
                inWidth, inHeight, param->frameOut.width, param->frameOut.height);
            return RGY_ERR_NONE;
        }
        AddMessage(RGY_LOG_WARN, _T("Failed to init nvvfx-superres with ratio %.1f (%dx%d -> %dx%d), retrying with other ratios...\n"), ratio.qdouble(),
            inWidth, inHeight, param->frameOut.width, param->frameOut.height);
    }

    AddMessage(RGY_LOG_ERROR, _T("Suitable ratio for nvvfx-superres not found.\n"));
    return RGY_ERR_UNSUPPORTED;
}

RGY_ERR NVEncFilterResize::init(shared_ptr<NVEncFilterParam> pParam, shared_ptr<RGYLog> pPrintMes) {
    RGY_ERR sts = RGY_ERR_NONE;
    m_pLog = pPrintMes;
    auto pResizeParam = std::dynamic_pointer_cast<NVEncFilterParamResize>(pParam);
    if (!pResizeParam) {
        AddMessage(RGY_LOG_ERROR, _T("Invalid parameter type.\n"));
        return RGY_ERR_INVALID_PARAM;
    }
#if defined(_WIN32) || defined(_WIN64)
    // linuxではnppは静的リンクにしたので、下記チェックは不要になった
    if (isNppResizeFiter(pResizeParam->interp) && !check_if_nppi_dll_available()) {
        AddMessage(RGY_LOG_WARN, _T("--vpp-resize %s requires \"%s\", not available on your system.\n"), get_chr_from_value(list_vpp_resize, pResizeParam->interp), NPPI_DLL_NAME_TSTR);
        pResizeParam->interp = RGY_VPP_RESIZE_SPLINE36;
        AddMessage(RGY_LOG_WARN, _T("switching to %s."), get_chr_from_value(list_vpp_resize, pResizeParam->interp));
    }
#endif
    //パラメータチェック
    if (pResizeParam->frameOut.height <= 0 || pResizeParam->frameOut.width <= 0) {
        AddMessage(RGY_LOG_ERROR, _T("Invalid parameter.\n"));
        return RGY_ERR_INVALID_PARAM;
    }

    auto resizeInterp = pResizeParam->interp;
    if (isNvvfxResizeFiter(pResizeParam->interp)) {
        if (!pResizeParam->nvvfxSuperRes) {
            AddMessage(RGY_LOG_ERROR, _T("nvvfx parameter unknown.\n"));
            return RGY_ERR_UNKNOWN;
        }
        sts = initNvvfxFilter(pResizeParam.get());
        if (sts != RGY_ERR_NONE) {
            AddMessage(RGY_LOG_ERROR, _T("Failed to init nvvfx filter: %s.\n"), get_err_mes(sts));
            return sts;
        }
        resizeInterp = pResizeParam->nvvfxSubAlgo;
    } else {
        m_nvvfxSuperRes.reset(); // 不要になったら解放
        pResizeParam->nvvfxSuperRes.reset();
    }
    if (isNgxResizeFiter(pResizeParam->interp)) {
        if (!m_ngxVSR) {
            m_ngxVSR = std::make_unique<NVEncFilterNGXVSR>();
        }
        pResizeParam->ngxvsr->frameIn = pResizeParam->frameIn;
        pResizeParam->ngxvsr->frameOut = pResizeParam->frameOut;
        sts = m_ngxVSR->init(pResizeParam->ngxvsr, m_pLog);
        if (sts != RGY_ERR_NONE) {
            AddMessage(RGY_LOG_ERROR, _T("Failed to init ngx vsr filter: %s.\n"), get_err_mes(sts));
            return sts;
        }
        pResizeParam->frameOut = pResizeParam->ngxvsr->frameOut;
    } else {
        m_ngxVSR.reset(); // 不要になったら解放
        pResizeParam->ngxvsr.reset();
    }
    if (isLibplaceboResizeFiter(pResizeParam->interp)) {
        if (!m_libplaceboResample) {
            m_libplaceboResample = std::make_unique<NVEncFilterLibplaceboResample>();
        }
        pResizeParam->libplaceboResample->frameIn = pResizeParam->frameIn;
        pResizeParam->libplaceboResample->frameOut = pResizeParam->frameOut;
        sts = m_libplaceboResample->init(pResizeParam->libplaceboResample, m_pLog);
        if (sts != RGY_ERR_NONE) {
            AddMessage(RGY_LOG_ERROR, _T("Failed to init libplacebo resample filter: %s.\n"), get_err_mes(sts));
            return sts;
        }
    } else {
        m_libplaceboResample.reset(); // 不要になったら解放
        pResizeParam->libplaceboResample.reset();
    }

    sts = AllocFrameBuf(pResizeParam->frameOut, 1);
    if (sts != RGY_ERR_NONE) {
        AddMessage(RGY_LOG_ERROR, _T("failed to allocate memory: %s.\n"), get_err_mes(sts));
        return sts;
    }
    for (int i = 0; i < RGY_CSP_PLANES[pParam->frameOut.csp]; i++) {
        pResizeParam->frameOut.pitch[i] = m_frameBuf[0]->frame.pitch[i];
    }

    if (resizeInterp == RGY_VPP_RESIZE_FSR1) {
        if (!m_fsr1Easu || cmpFrameInfoCspResolution(&m_fsr1Easu->frame, &pResizeParam->frameOut)) {
            m_fsr1Easu.reset(new CUFrameBuf(pResizeParam->frameOut));
            m_fsr1Easu->releasePtr();
            sts = m_fsr1Easu->alloc();
            if (sts != RGY_ERR_NONE) {
                AddMessage(RGY_LOG_ERROR, _T("failed to allocate FSR intermediate buffer: %s.\n"), get_err_mes(sts));
                return sts;
            }
        }
    } else {
        m_fsr1Easu.reset();
    }

    if ((!m_weightSpline || m_weightSplineAlgo != resizeInterp)
        && (resizeInterp == RGY_VPP_RESIZE_SPLINE16 || resizeInterp == RGY_VPP_RESIZE_SPLINE36 || resizeInterp == RGY_VPP_RESIZE_SPLINE64)) {
        static const auto SPLINE16_WEIGHT = std::vector<float>{
            1.0f,       -9.0f/5.0f,  -1.0f/5.0f, 1.0f,
            -1.0f/3.0f,  9.0f/5.0f, -46.0f/15.0f, 8.0f/5.0f
        };
        static const auto SPLINE36_WEIGHT = std::vector<float>{
            13.0f/11.0f, -453.0f/209.0f,    -3.0f/209.0f,  1.0f,
            -6.0f/11.0f,  612.0f/209.0f, -1038.0f/209.0f,  540.0f/209.0f,
             1.0f/11.0f, -159.0f/209.0f,   434.0f/209.0f, -384.0f/209.0f
        };
        static const auto SPLINE64_WEIGHT = std::vector<float>{
             49.0f/41.0f, -6387.0f/2911.0f,     -3.0f/2911.0f,  1.0f,
            -24.0f/41.0f,  9144.0f/2911.0f, -15504.0f/2911.0f,  8064.0f/2911.0f,
              6.0f/41.0f, -3564.0f/2911.0f,   9726.0f/2911.0f, -8604.0f/2911.0f,
             -1.0f/41.0f,   807.0f/2911.0f,  -3022.0f/2911.0f,  3720.0f/2911.0f
        };
        const std::vector<float>* weight = nullptr;
        switch (resizeInterp) {
        case RGY_VPP_RESIZE_SPLINE16: weight = &SPLINE16_WEIGHT; break;
        case RGY_VPP_RESIZE_SPLINE36: weight = &SPLINE36_WEIGHT; break;
        case RGY_VPP_RESIZE_SPLINE64: weight = &SPLINE64_WEIGHT; break;
        default: {
            AddMessage(RGY_LOG_ERROR, _T("unknown interpolation type: %d.\n"), resizeInterp);
            return RGY_ERR_INVALID_PARAM;
        }
        }
        m_weightSpline = std::unique_ptr<CUMemBuf>(new CUMemBuf(sizeof((*weight)[0]) * weight->size()));
        if (RGY_ERR_NONE != (sts = m_weightSpline->alloc())) {
            AddMessage(RGY_LOG_ERROR, _T("failed to allocate memory: %s.\n"), get_err_mes(sts));
            return sts;
        }
        sts = err_to_rgy(cudaMemcpy(m_weightSpline->ptr, weight->data(), m_weightSpline->nSize, cudaMemcpyHostToDevice));
        if (sts != RGY_ERR_NONE) {
            AddMessage(RGY_LOG_ERROR, _T("failed to send weight to gpu memory: %s.\n"), get_err_mes(sts));
            return sts;
        }
        m_weightSplineAlgo = resizeInterp;
    }

    if (isJincResize(resizeInterp) && (!m_weightJinc || m_weightJincAlgo != resizeInterp)) {
        const std::vector<float> lut = buildJincLut(jincRadius(resizeInterp));
        m_weightJinc = std::unique_ptr<CUMemBuf>(new CUMemBuf(sizeof(lut[0]) * lut.size()));
        if (RGY_ERR_NONE != (sts = m_weightJinc->alloc())) {
            AddMessage(RGY_LOG_ERROR, _T("failed to allocate jinc LUT memory: %s.\n"), get_err_mes(sts));
            return sts;
        }
        sts = err_to_rgy(cudaMemcpy(m_weightJinc->ptr, lut.data(), m_weightJinc->nSize, cudaMemcpyHostToDevice));
        if (sts != RGY_ERR_NONE) {
            AddMessage(RGY_LOG_ERROR, _T("failed to send jinc LUT to gpu memory: %s.\n"), get_err_mes(sts));
            return sts;
        }
        m_weightJincAlgo = resizeInterp;
    }

    if (isNisResize(resizeInterp)) {
        if (!m_nisCoefScale || !m_nisCoefUsm) {
            const size_t coefBytes = sizeof(float) * nis::kPhaseCount * nis::kFilterSize;
            m_nisCoefScale = std::unique_ptr<CUMemBuf>(new CUMemBuf(coefBytes));
            m_nisCoefUsm   = std::unique_ptr<CUMemBuf>(new CUMemBuf(coefBytes));
            if (RGY_ERR_NONE != (sts = m_nisCoefScale->alloc()) || RGY_ERR_NONE != (sts = m_nisCoefUsm->alloc())) {
                AddMessage(RGY_LOG_ERROR, _T("failed to allocate NIS coef memory: %s.\n"), get_err_mes(sts));
                return sts;
            }
            sts = err_to_rgy(cudaMemcpy(m_nisCoefScale->ptr, &nis::coef_scale[0][0], coefBytes, cudaMemcpyHostToDevice));
            if (sts == RGY_ERR_NONE) sts = err_to_rgy(cudaMemcpy(m_nisCoefUsm->ptr, &nis::coef_usm[0][0], coefBytes, cudaMemcpyHostToDevice));
            if (sts != RGY_ERR_NONE) {
                AddMessage(RGY_LOG_ERROR, _T("failed to send NIS coef to gpu memory: %s.\n"), get_err_mes(sts));
                return sts;
            }
        }
        // NIS cascade: NIS is tuned for <=2x (kScale in [0.5,1]). For larger ratios
        // split into N geometric stages of <=2x each (e.g. 2.5x -> two ~1.58x stages),
        // chaining through intermediate buffers; only the final stage applies USM.
        const float ratioX = (float)pResizeParam->frameOut.width  / pResizeParam->frameIn.width;
        const float ratioY = (float)pResizeParam->frameOut.height / pResizeParam->frameIn.height;
        const float maxRatio = std::max(ratioX, ratioY);
        int stages = 1;
        if (maxRatio > 2.0f) {
            stages = (int)std::ceil(std::log2(maxRatio) - 1e-4f);
            if (stages < 2) stages = 2;
        }
        if (pResizeParam->nis.cascade == RGY_NIS_CASCADE_OFF && stages > 1) {
            AddMessage(RGY_LOG_ERROR, _T("NIS cascade=off cannot handle %.2fx (> 2x). Use cascade=auto or a smaller output size.\n"), maxRatio);
            return RGY_ERR_UNSUPPORTED;
        }
        if (pResizeParam->nis.cascade == RGY_NIS_CASCADE_ON && stages < 2 && maxRatio > 1.0f) {
            stages = 2; // force a 2-stage cascade even at <=2x (test path)
        }
        m_nisStages = stages;
        m_nisCascadeInter.clear();
        if (stages > 1) {
            const float perStageRatio = std::pow(maxRatio, 1.0f / (float)stages);
            for (int k = 0; k < stages - 1; k++) {
                const int nextW = (int)std::round((float)pResizeParam->frameIn.width  * std::pow(perStageRatio, (float)(k + 1)));
                const int nextH = (int)std::round((float)pResizeParam->frameIn.height * std::pow(perStageRatio, (float)(k + 1)));
                auto inter = std::make_unique<CUFrameBuf>();
                sts = inter->alloc(nextW, nextH, pResizeParam->frameOut.csp);
                if (sts != RGY_ERR_NONE) {
                    AddMessage(RGY_LOG_ERROR, _T("failed to allocate NIS cascade intermediate %dx%d: %s.\n"), nextW, nextH, get_err_mes(sts));
                    return sts;
                }
                AddMessage(RGY_LOG_DEBUG, _T("NIS cascade stage %d/%d intermediate: %dx%d.\n"), k + 1, stages, nextW, nextH);
                m_nisCascadeInter.push_back(std::move(inter));
            }
        }
    }

    tstring info;
    if (m_nvvfxSuperRes) {
        info = strsprintf(_T("resize: %s %dx%d -> %dx%d"),
            get_chr_from_value(list_vpp_resize, pResizeParam->interp),
            pParam->frameIn.width, pParam->frameIn.height,
            pResizeParam->nvvfxSuperRes->frameOut.width, pResizeParam->nvvfxSuperRes->frameOut.height);
        const auto indent2 = tstring(_tcslen(_T("resize:")) + 5, _T(' '));
        const auto indent = tstring(INFO_INDENT) + indent2;
        bool firstIndent = true;
        if (   pResizeParam->nvvfxSuperRes->frameIn.width != pParam->frameIn.width
            || pResizeParam->nvvfxSuperRes->frameIn.height != pParam->frameIn.height) {
            info += _T("\n") + tstring(INFO_INDENT) + tstring(_tcslen(_T("resize: ")), _T(' '));
            info += strsprintf(_T("%s %dx%d -> %dx%d"),
                get_chr_from_value(list_vpp_resize, pResizeParam->nvvfxSubAlgo),
                pParam->frameIn.width, pParam->frameIn.height,
                pResizeParam->nvvfxSuperRes->frameIn.width, pResizeParam->nvvfxSuperRes->frameIn.height);
        }
        for (const auto& str : split(m_nvvfxSuperRes->GetInputMessage(), _T("\n"))) {
            info += _T("\n") + ((firstIndent) ? indent : indent2) + str;
            firstIndent = false;
        }
        if (   pResizeParam->nvvfxSuperRes->frameOut.width != pParam->frameOut.width
            || pResizeParam->nvvfxSuperRes->frameOut.height != pParam->frameOut.height) {
            info += _T("\n") + tstring(INFO_INDENT) + tstring(_tcslen(_T("resize: ")), _T(' '));
            info += strsprintf(_T("%s %dx%d -> %dx%d"),
                get_chr_from_value(list_vpp_resize, pResizeParam->nvvfxSubAlgo),
                pResizeParam->nvvfxSuperRes->frameOut.width, pResizeParam->nvvfxSuperRes->frameOut.height,
                pParam->frameOut.width, pParam->frameOut.height);
        }
    } else if (m_ngxVSR) {
        info = strsprintf(_T("resize: %s %dx%d -> %dx%d"),
            get_chr_from_value(list_vpp_resize, pResizeParam->interp),
            pParam->frameIn.width, pParam->frameIn.height,
            pParam->frameOut.width, pParam->frameOut.height);
        const auto indent2 = tstring(_tcslen(_T("resize:")) + 5, _T(' '));
        const auto indent = tstring(INFO_INDENT) + indent2;
        bool firstIndent = true;
        for (const auto& str : split(m_ngxVSR->GetInputMessage(), _T("\n"))) {
            info += _T("\n") + ((firstIndent) ? indent : indent2) + str;
            firstIndent = false;
        }
    } else {
        info = pResizeParam->print();
    }
    setFilterInfo(info);

    //コピーを保存
    m_param = pResizeParam;
    return sts;
}


NVEncFilterParamResize::NVEncFilterParamResize() :
    interp(RGY_VPP_RESIZE_SPLINE36),
    nvvfxSubAlgo(RGY_VPP_RESIZE_SPLINE36),
    fsr1(),
    nvvfxSuperRes(),
    ngxvsr(),
    libplaceboResample() {
}

NVEncFilterParamResize::~NVEncFilterParamResize() {};

tstring NVEncFilterParamResize::print() const {
    if (nvvfxSuperRes) {
        auto str = strsprintf(_T("resize: %s %dx%d -> %dx%d"),
            get_chr_from_value(list_vpp_resize, interp),
            frameIn.width, frameIn.height,
            nvvfxSuperRes->frameOut.width, nvvfxSuperRes->frameOut.height);
        if (   nvvfxSuperRes->frameOut.width != frameOut.width
            || nvvfxSuperRes->frameOut.height != frameOut.height) {
            str += strsprintf(_T("\n                       %s %dx%d -> %dx%d"),
                get_chr_from_value(list_vpp_resize, nvvfxSubAlgo),
                nvvfxSuperRes->frameOut.width, nvvfxSuperRes->frameOut.height,
                frameOut.width, frameOut.height);
        }
        return str;
    }
    auto str = strsprintf(_T("resize(%s): %dx%d -> %dx%d"),
        get_chr_from_value(list_vpp_resize, interp),
        frameIn.width, frameIn.height,
        frameOut.width, frameOut.height);
    if (libplaceboResample) {
        str += _T("\n                 ");
        str += libplaceboResample->print();
    } else if (interp == RGY_VPP_RESIZE_FSR1) {
        str += _T("\n                 ");
        str += fsr1.print();
    }
    return str;
}

RGY_ERR NVEncFilterResize::run_filter(const RGYFrameInfo *pInputFrame, RGYFrameInfo **ppOutputFrames, int *pOutputFrameNum, cudaStream_t stream) {
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
        return RGY_ERR_UNSUPPORTED;
    }

    auto pResizeParam = std::dynamic_pointer_cast<NVEncFilterParamResize>(m_param);
    if (!pResizeParam) {
        AddMessage(RGY_LOG_ERROR, _T("Invalid parameter type.\n"));
        return RGY_ERR_INVALID_PARAM;
    }

    auto resizeInterp = (m_nvvfxSuperRes) ? pResizeParam->nvvfxSubAlgo : pResizeParam->interp;
    if (m_nvvfxSuperRes) {
        // 入力フレームの解像度が一致している場合は、先にnvvfxSuperresを適用する
        if (pResizeParam->nvvfxSuperRes->frameIn.width  == pInputFrame->width
         && pResizeParam->nvvfxSuperRes->frameIn.height == pInputFrame->height) {
            int nvvfxOutputNum = 0;
            RGYFrameInfo *outInfo[1] = { 0 };
            RGYFrameInfo inputFrame = *pInputFrame;
            auto sts_filter = m_nvvfxSuperRes->filter(&inputFrame, (RGYFrameInfo **)&outInfo, &nvvfxOutputNum, stream);
            if (outInfo[0] == nullptr || nvvfxOutputNum != 1) {
                AddMessage(RGY_LOG_ERROR, _T("Unknown behavior \"%s\".\n"), m_nvvfxSuperRes->name().c_str());
                return sts_filter;
            }
            if (sts_filter != RGY_ERR_NONE || nvvfxOutputNum != 1) {
                AddMessage(RGY_LOG_ERROR, _T("Error while running filter \"%s\".\n"), m_nvvfxSuperRes->name().c_str());
                return sts_filter;
            }
            pInputFrame = outInfo[0];
        } else {
            ppOutputFrames[0]->width = pResizeParam->nvvfxSuperRes->frameIn.width;
            ppOutputFrames[0]->height = pResizeParam->nvvfxSuperRes->frameIn.height;
        }
    } else if (m_ngxVSR || m_libplaceboResample) {
        RGYFrameInfo inputFrame = *pInputFrame;
        NVEncFilter *filter = (m_ngxVSR) ? (NVEncFilter *)m_ngxVSR.get() : (NVEncFilter *)m_libplaceboResample.get();
        if (filter == nullptr) {
            AddMessage(RGY_LOG_ERROR, _T("Failed to get filter to be applied.\n"));
            return RGY_ERR_UNKNOWN;
        }
        auto sts_filter = filter->filter(&inputFrame, ppOutputFrames, pOutputFrameNum, stream);
        if (ppOutputFrames[0] == nullptr || *pOutputFrameNum != 1) {
            AddMessage(RGY_LOG_ERROR, _T("Unknown behavior \"%s\".\n"), filter->name().c_str());
            return sts_filter;
        }
        if (sts_filter != RGY_ERR_NONE || *pOutputFrameNum != 1) {
            AddMessage(RGY_LOG_ERROR, _T("Error while running filter \"%s\".\n"), filter->name().c_str());
            return sts_filter;
        }
        return RGY_ERR_NONE;
    }

    if (m_param->frameOut.csp != m_param->frameIn.csp) {
        AddMessage(RGY_LOG_ERROR, _T("csp does not match.\n"));
        return RGY_ERR_UNSUPPORTED;
    }

    if (   pInputFrame->width != ppOutputFrames[0]->width
        || pInputFrame->height != ppOutputFrames[0]->height) {
        if (isJincResize(pResizeParam->interp)) {
            static const std::map<RGY_DATA_TYPE, decltype(resize_jinc_frame<uint8_t, 8>)*> jinc_list = {
                { RGY_DATA_TYPE_U8,  resize_jinc_frame<uint8_t,   8> },
                { RGY_DATA_TYPE_U16, resize_jinc_frame<uint16_t, 16> }
            };
            if (jinc_list.count(RGY_CSP_DATA_TYPE[pInputFrame->csp]) == 0) {
                AddMessage(RGY_LOG_ERROR, _T("unsupported csp %s for jinc.\n"), RGY_CSP_NAMES[pInputFrame->csp]);
                return RGY_ERR_UNSUPPORTED;
            }
            sts = jinc_list.at(RGY_CSP_DATA_TYPE[pInputFrame->csp])(ppOutputFrames[0], pInputFrame, pResizeParam->interp, (float *)m_weightJinc->ptr, stream);
        } else if (isNisResize(pResizeParam->interp)) {
            static const std::map<RGY_DATA_TYPE, decltype(resize_nis_frame<uint8_t, 8>)*> nis_list = {
                { RGY_DATA_TYPE_U8,  resize_nis_frame<uint8_t,   8> },
                { RGY_DATA_TYPE_U16, resize_nis_frame<uint16_t, 16> }
            };
            if (nis_list.count(RGY_CSP_DATA_TYPE[pInputFrame->csp]) == 0) {
                AddMessage(RGY_LOG_ERROR, _T("unsupported csp %s for nis.\n"), RGY_CSP_NAMES[pInputFrame->csp]);
                return RGY_ERR_UNSUPPORTED;
            }
            auto nisFunc = nis_list.at(RGY_CSP_DATA_TYPE[pInputFrame->csp]);
            // sharpen fields (kScale is recomputed per plane/stage in the launcher).
            // hdr=auto/sdr -> SDR band; hdr=pq -> PQ specular-protect band.
            const int nisHdr = (pResizeParam->nis.hdrMode == RGY_NIS_HDR_PQ) ? 2 : 0;
            NISParams nisFull  = nisBuildParams(pResizeParam->nis.sharpness, nisHdr, pInputFrame->width, pInputFrame->height, ppOutputFrames[0]->width, ppOutputFrames[0]->height);
            NISParams nisInter = nisFull; // intermediate stages: base resample only, no USM (avoid chained over-sharpen)
            nisInter.kSharpStrengthMin = nisInter.kSharpStrengthScale = nisInter.kSharpLimitMin = nisInter.kSharpLimitScale = 0.0f;
            const RGYFrameInfo *src = pInputFrame;
            for (int k = 0; k < m_nisStages; k++) {
                const bool finalStage = (k == m_nisStages - 1);
                RGYFrameInfo *dst = finalStage ? ppOutputFrames[0] : &m_nisCascadeInter[k]->frame;
                sts = nisFunc(dst, src, finalStage ? nisFull : nisInter, (float *)m_nisCoefScale->ptr, (float *)m_nisCoefUsm->ptr, stream);
                if (sts != RGY_ERR_NONE) break;
                src = dst;
            }
        } else if (isNppResizeFiter(pResizeParam->interp)) {
            static const auto supportedCspYUV444 = make_array<RGY_CSP>(RGY_CSP_YUV444, RGY_CSP_YUV444_09, RGY_CSP_YUV444_10, RGY_CSP_YUV444_12, RGY_CSP_YUV444_14, RGY_CSP_YUV444_16);
            if (std::find(supportedCspYUV444.begin(), supportedCspYUV444.end(), m_param->frameIn.csp) != supportedCspYUV444.end()) {
                sts = resizeNppiYUV444(ppOutputFrames[0], pInputFrame, stream);
            } else {
                sts = resizeNppi(ppOutputFrames[0], pInputFrame, stream);
            }
        } else {
            static const std::map<RGY_DATA_TYPE, decltype(resize_frame<uint8_t, 8>)*> resize_list = {
                { RGY_DATA_TYPE_U8,  resize_frame<uint8_t,   8> },
                { RGY_DATA_TYPE_U16, resize_frame<uint16_t, 16> }
            };
            if (resize_list.count(RGY_CSP_DATA_TYPE[pInputFrame->csp]) == 0) {
                AddMessage(RGY_LOG_ERROR, _T("unsupported csp %s.\n"), RGY_CSP_NAMES[pInputFrame->csp]);
                return RGY_ERR_UNSUPPORTED;
            }
            if (resizeInterp == RGY_VPP_RESIZE_FSR1) {
            if (resizeInterp == RGY_VPP_RESIZE_FSR1) {
                if (!m_fsr1Easu) {
                    AddMessage(RGY_LOG_ERROR, _T("FSR intermediate buffer is not allocated.\n"));
                    return RGY_ERR_NULL_PTR;
                }
                static const std::map<RGY_DATA_TYPE, decltype(resize_fsr1_frame<uint8_t, 8>)*> resize_fsr1_list = {
                    { RGY_DATA_TYPE_U8,  resize_fsr1_frame<uint8_t,   8> },
                    { RGY_DATA_TYPE_U16, resize_fsr1_frame<uint16_t, 16> }
                };
                sts = resize_fsr1_list.at(RGY_CSP_DATA_TYPE[pInputFrame->csp])(ppOutputFrames[0], pInputFrame, &m_fsr1Easu->frame, pResizeParam->fsr1.sharpness, stream);
            } else {
                if (resizeInterp == RGY_VPP_RESIZE_BICUBIC) {
                    // push the user-tunable bicubic B/C to constant memory (stream-ordered before the kernel).
                    const float bc[2] = { pResizeParam->bicubic.b, pResizeParam->bicubic.c };
                    cudaMemcpyToSymbolAsync(g_bicubicBC, bc, sizeof(bc), 0, cudaMemcpyHostToDevice, stream);
                }
                sts = resize_list.at(RGY_CSP_DATA_TYPE[pInputFrame->csp])(ppOutputFrames[0], pInputFrame, resizeInterp, (m_weightSpline) ? (float *)m_weightSpline->ptr : nullptr, stream);
            }
            if (sts != RGY_ERR_NONE) {
                AddMessage(RGY_LOG_ERROR, _T("error at resize(%s): %s.\n"),
                    RGY_CSP_NAMES[pInputFrame->csp],
                    get_err_mes(sts));
                return sts;
            }
        }
        if (m_nvvfxSuperRes
            && pResizeParam->frameOut.width  != ppOutputFrames[0]->width
            && pResizeParam->frameOut.height != ppOutputFrames[0]->height) {
            int nvvfxOutputNum = 0;
            RGYFrameInfo *outInfo[1] = { 0 };
            RGYFrameInfo inputFrame = *ppOutputFrames[0];
            auto sts_filter = m_nvvfxSuperRes->filter(&inputFrame, (RGYFrameInfo **)&outInfo, &nvvfxOutputNum, stream);
            if (outInfo[0] == nullptr || nvvfxOutputNum != 1) {
                AddMessage(RGY_LOG_ERROR, _T("Unknown behavior \"%s\".\n"), m_nvvfxSuperRes->name().c_str());
                return sts_filter;
            }
            if (sts_filter != RGY_ERR_NONE || nvvfxOutputNum != 1) {
                AddMessage(RGY_LOG_ERROR, _T("Error while running filter \"%s\".\n"), m_nvvfxSuperRes->name().c_str());
                return sts_filter;
            }
            ppOutputFrames[0] = outInfo[0];
        }
    } else {
        sts = copyFrameAsync(ppOutputFrames[0], pInputFrame, stream);
        if (sts != RGY_ERR_NONE) {
            AddMessage(RGY_LOG_ERROR, _T("Failed to copy frame: %s.\n"), get_err_mes(sts));
            return sts;
        }
    }
    return sts;
}

void NVEncFilterResize::close() {
    m_frameBuf.clear();
    m_ngxVSR.reset();
    m_nvvfxSuperRes.reset();
    m_libplaceboResample.reset();
    m_weightSpline.reset();
    m_weightSplineAlgo = RGY_VPP_RESIZE_UNKNOWN;
    m_fsr1Easu.reset();
    m_weightJinc.reset();
    m_weightJincAlgo = RGY_VPP_RESIZE_UNKNOWN;
    m_nisCoefScale.reset();
    m_nisCoefUsm.reset();
    m_nisCascadeInter.clear();
    m_nisStages = 1;
    m_bInterlacedWarn = false;
}
