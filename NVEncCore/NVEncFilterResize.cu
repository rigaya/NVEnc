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
#define _USE_MATH_DEFINES
#include <cmath>
#include "convert_csp.h"
#include "NVEncFilter.h"
#include "NVEncFilterNvvfx.h"
#include "NVEncParam.h"
#pragma warning (push)
#pragma warning (disable: 4819)
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#pragma warning (pop)

#if defined(_WIN32) || defined(_WIN64)
#if __CUDACC_VER_MAJOR__ == 8
const TCHAR *NPPI_DLL_NAME_TSTR = _T("nppc64_80.dll");
#elif __CUDACC_VER_MAJOR__ == 9 && __CUDACC_VER_MINOR__ == 0
const TCHAR *NPPI_DLL_NAME_TSTR = _T("nppc64_90.dll");
#elif __CUDACC_VER_MAJOR__ == 9 && __CUDACC_VER_MINOR__ == 1
const TCHAR *NPPI_DLL_NAME_TSTR = _T("nppc64_91.dll");
#elif __CUDACC_VER_MAJOR__ == 9 && __CUDACC_VER_MINOR__ == 2
const TCHAR *NPPI_DLL_NAME_TSTR = _T("nppc64_92.dll");
#elif __CUDACC_VER_MAJOR__ == 10
const TCHAR *NPPI_DLL_NAME_TSTR = _T("nppc64_10.dll");
#elif __CUDACC_VER_MAJOR__ == 11
const TCHAR *NPPI_DLL_NAME_TSTR = _T("nppc64_11.dll");
#elif __CUDACC_VER_MAJOR__ == 12
const TCHAR *NPPI_DLL_NAME_TSTR = _T("nppc64_12.dll");
#endif

#if __CUDACC_VER_MAJOR__ == 8
const TCHAR *NVRTC_DLL_NAME_TSTR = _T("nvrtc64_80.dll");
const TCHAR *NVRTC_BUILTIN_DLL_NAME_TSTR = _T("nvrtc-builtins64_80.dll");
#elif __CUDACC_VER_MAJOR__ == 9 && __CUDACC_VER_MINOR__ == 0
const TCHAR *NVRTC_DLL_NAME_TSTR = _T("nvrtc64_90.dll");
const TCHAR *NVRTC_BUILTIN_DLL_NAME_TSTR = _T("nvrtc-builtins64_90.dll");
#elif __CUDACC_VER_MAJOR__ == 9 && __CUDACC_VER_MINOR__ == 1
const TCHAR *NVRTC_DLL_NAME_TSTR = _T("nvrtc64_91.dll");
const TCHAR *NVRTC_BUILTIN_DLL_NAME_TSTR = _T("nvrtc-builtins64_91.dll");
#elif __CUDACC_VER_MAJOR__ == 9 && __CUDACC_VER_MINOR__ == 2
const TCHAR *NVRTC_DLL_NAME_TSTR = _T("nvrtc64_92.dll");
const TCHAR *NVRTC_BUILTIN_DLL_NAME_TSTR = _T("nvrtc-builtins64_92.dll");
#elif __CUDACC_VER_MAJOR__ == 10 && __CUDACC_VER_MINOR__ == 0
const TCHAR *NVRTC_DLL_NAME_TSTR = _T("nvrtc64_100_0.dll");
const TCHAR *NVRTC_BUILTIN_DLL_NAME_TSTR = _T("nvrtc-builtins64_100.dll");
#elif __CUDACC_VER_MAJOR__ == 10 && __CUDACC_VER_MINOR__ == 1
const TCHAR *NVRTC_DLL_NAME_TSTR = _T("nvrtc64_101_0.dll");
const TCHAR *NVRTC_BUILTIN_DLL_NAME_TSTR = _T("nvrtc-builtins64_101.dll");
#elif __CUDACC_VER_MAJOR__ == 10 && __CUDACC_VER_MINOR__ == 2
const TCHAR *NVRTC_DLL_NAME_TSTR = _T("nvrtc64_102_0.dll");
const TCHAR *NVRTC_BUILTIN_DLL_NAME_TSTR = _T("nvrtc-builtins64_102.dll");
#elif __CUDACC_VER_MAJOR__ == 11 && __CUDACC_VER_MINOR__ == 0
const TCHAR *NVRTC_DLL_NAME_TSTR = _T("nvrtc64_110_0.dll");
const TCHAR *NVRTC_BUILTIN_DLL_NAME_TSTR = _T("nvrtc-builtins64_110.dll");
#elif __CUDACC_VER_MAJOR__ == 11 && __CUDACC_VER_MINOR__ == 1
const TCHAR *NVRTC_DLL_NAME_TSTR = _T("nvrtc64_111_0.dll");
const TCHAR *NVRTC_BUILTIN_DLL_NAME_TSTR = _T("nvrtc-builtins64_111.dll");
#elif __CUDACC_VER_MAJOR__ == 11 && __CUDACC_VER_MINOR__ == 2
const TCHAR *NVRTC_DLL_NAME_TSTR = _T("nvrtc64_112_0.dll");
const TCHAR *NVRTC_BUILTIN_DLL_NAME_TSTR = _T("nvrtc-builtins64_112.dll");
#elif __CUDACC_VER_MAJOR__ == 11 && __CUDACC_VER_MINOR__ == 3
const TCHAR *NVRTC_DLL_NAME_TSTR = _T("nvrtc64_113_0.dll");
const TCHAR *NVRTC_BUILTIN_DLL_NAME_TSTR = _T("nvrtc-builtins64_113.dll");
#elif __CUDACC_VER_MAJOR__ == 11 && __CUDACC_VER_MINOR__ == 4
const TCHAR *NVRTC_DLL_NAME_TSTR = _T("nvrtc64_114_0.dll");
const TCHAR *NVRTC_BUILTIN_DLL_NAME_TSTR = _T("nvrtc-builtins64_114.dll");
#elif __CUDACC_VER_MAJOR__ == 11 && __CUDACC_VER_MINOR__ == 5
const TCHAR *NVRTC_DLL_NAME_TSTR = _T("nvrtc64_115_0.dll");
const TCHAR *NVRTC_BUILTIN_DLL_NAME_TSTR = _T("nvrtc-builtins64_115.dll");
#elif __CUDACC_VER_MAJOR__ == 11 && __CUDACC_VER_MINOR__ == 6
const TCHAR *NVRTC_DLL_NAME_TSTR = _T("nvrtc64_116_0.dll");
const TCHAR *NVRTC_BUILTIN_DLL_NAME_TSTR = _T("nvrtc-builtins64_116.dll");
#elif __CUDACC_VER_MAJOR__ == 11 && __CUDACC_VER_MINOR__ == 7
const TCHAR *NVRTC_DLL_NAME_TSTR = _T("nvrtc64_117_0.dll");
const TCHAR *NVRTC_BUILTIN_DLL_NAME_TSTR = _T("nvrtc-builtins64_117.dll");
#elif __CUDACC_VER_MAJOR__ == 12 && __CUDACC_VER_MINOR__ == 0
const TCHAR *NVRTC_DLL_NAME_TSTR = _T("nvrtc64_120_0.dll");
const TCHAR *NVRTC_BUILTIN_DLL_NAME_TSTR = _T("nvrtc-builtins64_120.dll");
#elif __CUDACC_VER_MAJOR__ == 12 && __CUDACC_VER_MINOR__ == 1
const TCHAR *NVRTC_DLL_NAME_TSTR = _T("nvrtc64_120_0.dll");
const TCHAR *NVRTC_BUILTIN_DLL_NAME_TSTR = _T("nvrtc-builtins64_121.dll");
#elif __CUDACC_VER_MAJOR__ == 12 && __CUDACC_VER_MINOR__ == 2
const TCHAR *NVRTC_DLL_NAME_TSTR = _T("nvrtc64_120_0.dll");
const TCHAR *NVRTC_BUILTIN_DLL_NAME_TSTR = _T("nvrtc-builtins64_122.dll");
#elif __CUDACC_VER_MAJOR__ == 12 && __CUDACC_VER_MINOR__ == 3
const TCHAR *NVRTC_DLL_NAME_TSTR = _T("nvrtc64_120_0.dll");
const TCHAR *NVRTC_BUILTIN_DLL_NAME_TSTR = _T("nvrtc-builtins64_123.dll");
#endif
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
    WEIGHT_BILINEAR
};

template<typename TypePixel>
cudaError_t setTexFieldResize(cudaTextureObject_t& texSrc, const RGYFrameInfo* pFrame, cudaTextureFilterMode filterMode, cudaTextureReadMode readMode, int normalizedCord) {
    texSrc = 0;

    cudaResourceDesc resDescSrc;
    memset(&resDescSrc, 0, sizeof(resDescSrc));
    resDescSrc.resType = cudaResourceTypePitch2D;
    resDescSrc.res.pitch2D.desc = cudaCreateChannelDesc<TypePixel>();
    resDescSrc.res.pitch2D.pitchInBytes = pFrame->pitch;
    resDescSrc.res.pitch2D.width = pFrame->width;
    resDescSrc.res.pitch2D.height = pFrame->height;
    resDescSrc.res.pitch2D.devPtr = (uint8_t*)pFrame->ptr;

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
    resize_texture<Type, bit_depth>((uint8_t *)pOutputFrame->ptr,
        pOutputFrame->pitch, pOutputFrame->width, pOutputFrame->height,
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
    if (planeOutputA.ptr != nullptr) {
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
    return 1.0f - x * (1.0f / radius);
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
        case WEIGHT_BICUBIC:  weight = factor_bicubic<radius>(delta, 0.0f, 0.6f); break;
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
    static_assert(block_x % 4 == 0 && block_y % 4 == 0, "block_x & block_y must be able to be divided by 4 to ensure psCopyFactor can be accessed by float4.");

    if (algo == WEIGHT_SPLINE) {
        if (threadIdx.y == 0) {
            static_assert(radius * 4 < block_x, "radius * 4 < block_x");
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
        calc_weight<radius, algo>(weightXshared + threadIdx.x * shared_weightXdim, srcX, srcFirstX, srcEndX, ratioClampedX, psCopyFactor);

        if (threadIdx.x < block_y) {
            // threadIdx.y==0のスレッドが、y方向の重みをそれぞれ計算してsharedメモリに書き込み
            const int thready = threadIdx.x;
            const int dstY = blockIdx.y * block_y + thready;
            const float srcY = ((float)(dstY + 0.5f)) * ratioInvY;
            const int srcFirstY = max(0, (int)floorf(srcY - srcWindowY));
            const int srcEndY = min(srcHeight - 1, (int)ceilf(srcY + srcWindowY));
            calc_weight<radius, algo>(weightYshared + thready * shared_weightYdim, srcY, srcFirstY, srcEndY, ratioClampedY, psCopyFactor);
        }
    }

    __syncthreads();

    const int ix = blockIdx.x * block_x + threadIdx.x;
    const int iy = blockIdx.y * block_y + threadIdx.y;
    if (ix < dstWidth && iy < dstHeight) {
        //ピクセルの中心を算出してからスケール
        const float x = ((float)ix + 0.5f) * ratioX;
        const float y = ((float)iy + 0.5f) * ratioY;

        const float srcX = ((float)(ix + 0.5f)) * ratioInvX;
        const int srcFirstX = max(0, (int)floorf(srcX - srcWindowX));
        const int srcEndX = min(srcWidth - 1, (int)ceilf(srcX + srcWindowX));
        const float *weightX = weightXshared + threadIdx.x * shared_weightXdim;

        const float srcY = ((float)(iy + 0.5f)) * ratioInvY;
        const int srcFirstY = max(0, (int)floorf(srcY - srcWindowY));
        const int srcEndY = min(srcHeight - 1, (int)ceilf(srcY + srcWindowY));
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

    const int shared_weightXdim = (((int)ceil(srcWindowX) + 1) * 2);
    const int shared_weightX = RESIZE_BLOCK_X * shared_weightXdim;

    const float ratioY = (float)(dstHeight) / srcHeight;
    const float ratioClampedY = min(ratioY, 1.0f);
    const float srcWindowY = radius / ratioClampedY;

    const int shared_weightYdim = (((int)ceil(srcWindowY) + 1) * 2);
    const int shared_weightY = RESIZE_BLOCK_Y * shared_weightYdim;

    const int shared_size_byte = (shared_weightX + shared_weightY + ((algo == WEIGHT_SPLINE) ? radius * 4 : 0)) * sizeof(float);

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
        (uint8_t*)pOutputPlane->ptr, pOutputPlane->pitch, pOutputPlane->width, pOutputPlane->height,
        (uint8_t*)pInputPlane->ptr, pInputPlane->pitch, pInputPlane->width, pInputPlane->height,
        pgFactor, stream);
    auto cudaerr = cudaGetLastError();
    if (cudaerr != cudaSuccess) {
        return err_to_rgy(cudaerr);
    }
    return RGY_ERR_NONE;
}

template<typename Type, int bit_depth, RESIZE_WEIGHT_TYPE algo, int radius>
static RGY_ERR resize_frame(RGYFrameInfo *pOutputFrame, const RGYFrameInfo *pInputFrame, const float *pgFactor, cudaStream_t stream) {
    const auto planeSrcY = getPlane(pInputFrame, RGY_PLANE_Y);
    const auto planeSrcU = getPlane(pInputFrame, RGY_PLANE_U);
    const auto planeSrcV = getPlane(pInputFrame, RGY_PLANE_V);
    const auto planeSrcA = getPlane(pInputFrame, RGY_PLANE_A);
    auto planeOutputY = getPlane(pOutputFrame, RGY_PLANE_Y);
    auto planeOutputU = getPlane(pOutputFrame, RGY_PLANE_U);
    auto planeOutputV = getPlane(pOutputFrame, RGY_PLANE_V);
    auto planeOutputA = getPlane(pOutputFrame, RGY_PLANE_A);

    auto sts = resize_plane<Type, bit_depth, algo, radius>(&planeOutputY, &planeSrcY, pgFactor, stream);
    if (sts != RGY_ERR_NONE) {
        return sts;
    }
    sts = resize_plane<Type, bit_depth, algo, radius>(&planeOutputU, &planeSrcU, pgFactor, stream);
    if (sts != RGY_ERR_NONE) {
        return sts;
    }
    sts = resize_plane<Type, bit_depth, algo, radius>(&planeOutputV, &planeSrcV, pgFactor, stream);
    if (sts != RGY_ERR_NONE) {
        return sts;
    }
    if (planeOutputA.ptr != nullptr) {
        sts = resize_plane<Type, bit_depth, algo, radius>(&planeOutputA, &planeSrcA, pgFactor, stream);
        if (sts != RGY_ERR_NONE) {
            return sts;
        }
    }
    return sts;
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
    default:  return RGY_ERR_NONE;
    }
}

template<typename T, typename Tfunc>
static NppStatus resize_nppi_yv12(RGYFrameInfo *pOutputFrame, const RGYFrameInfo *pInputFrame, Tfunc funcResize, NppiInterpolationMode interpMode) {
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
    //Y
    NppStatus sts = funcResize(
        (const T *)planeSrcY.ptr,
        srcSize, planeSrcY.pitch, srcRect,
        (T *)planeOutputY.ptr,
        planeOutputY.pitch, dstRect,
        factorX, factorY, 0.0, 0.0, interpMode);
    if (sts != NPP_SUCCESS) {
        return sts;
    }
    //U
    srcSize.width  >>= 1;
    srcSize.height >>= 1;
    srcRect.width  >>= 1;
    srcRect.height >>= 1;
    dstRect.width  >>= 1;
    dstRect.height >>= 1;
    sts = funcResize(
        (const T *)planeSrcU.ptr,
        srcSize, planeSrcU.pitch, srcRect,
        (T *)planeOutputU.ptr,
        planeOutputU.pitch, dstRect,
        factorX, factorY, 0.0, 0.0, interpMode);
    if (sts != NPP_SUCCESS) {
        return sts;
    }
    //V
    sts = funcResize(
        (const T *)planeSrcV.ptr,
        srcSize, planeSrcV.pitch, srcRect,
        (T *)planeOutputV.ptr,
        planeOutputV.pitch, dstRect,
        factorX, factorY, 0.0, 0.0, interpMode);
    return sts;
}

RGY_ERR NVEncFilterResize::resizeNppiYV12(RGYFrameInfo *pOutputFrame, const RGYFrameInfo *pInputFrame) {
#if _M_IX86
    AddMessage(RGY_LOG_ERROR, _T("npp filter not supported on x86.\n"));
    return RGY_ERR_UNSUPPORTED;
#else
    RGY_ERR sts = RGY_ERR_NONE;
    if (m_pParam->frameOut.csp != m_pParam->frameIn.csp) {
        AddMessage(RGY_LOG_ERROR, _T("csp does not match.\n"));
        return RGY_ERR_INVALID_PARAM;
    }
    auto pResizeParam = std::dynamic_pointer_cast<NVEncFilterParamResize>(m_pParam);
    if (!pResizeParam) {
        AddMessage(RGY_LOG_ERROR, _T("Invalid parameter type.\n"));
        return RGY_ERR_INVALID_PARAM;
    }
    const auto interp = vpp_resize_algo_rgy_to_enc(pResizeParam->interp);
    if (interp == NPPI_INTER_UNDEFINED) {
        AddMessage(RGY_LOG_ERROR, _T("Unknown nppi interp mode: %d.\n"), (int)pResizeParam->interp);
        return RGY_ERR_UNSUPPORTED;
    }
    static const auto supportedCspYV12High = make_array<RGY_CSP>(RGY_CSP_YV12_09, RGY_CSP_YV12_10, RGY_CSP_YV12_12, RGY_CSP_YV12_14, RGY_CSP_YV12_16);
    NppStatus nppsts = NPP_SUCCESS;
    if (m_pParam->frameIn.csp == RGY_CSP_YV12) {
        nppsts = resize_nppi_yv12<Npp8u>(pOutputFrame, pInputFrame, nppiResizeSqrPixel_8u_C1R, interp);
        if (nppsts != NPP_SUCCESS) {
            AddMessage(RGY_LOG_ERROR, _T("failed to resize: %d, %s.\n"), nppsts, char_to_tstring(_cudaGetErrorEnum(nppsts)).c_str());
            sts = RGY_ERR_CUDA;
        }
    } else if (std::find(supportedCspYV12High.begin(), supportedCspYV12High.end(), m_pParam->frameIn.csp) != supportedCspYV12High.end()) {
        nppsts = resize_nppi_yv12<Npp16u>(pOutputFrame, pInputFrame, nppiResizeSqrPixel_16u_C1R, interp);
        if (nppsts != NPP_SUCCESS) {
            AddMessage(RGY_LOG_ERROR, _T("failed to resize: %d, %s.\n"), nppsts, char_to_tstring(_cudaGetErrorEnum(nppsts)).c_str());
            sts = RGY_ERR_CUDA;
        }
    } else {
        AddMessage(RGY_LOG_ERROR, _T("unsupported csp.\n"));
        sts = RGY_ERR_UNSUPPORTED;
    }
    return sts;
#endif
}

template<typename T, typename Tfunc>
static NppStatus resize_nppi_yuv444(RGYFrameInfo *pOutputFrame, const RGYFrameInfo *pInputFrame, Tfunc funcResize, NppiInterpolationMode interpMode) {
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
        (const T *)planeSrcY.ptr,
        (const T *)planeSrcU.ptr,
        (const T *)planeSrcV.ptr
    };
    T *pDst[3] = {
        (T *)planeOutputY.ptr,
        (T *)planeOutputU.ptr,
        (T *)planeOutputV.ptr
    };
    NppStatus sts = funcResize(
        pSrc,
        srcSize, planeSrcY.pitch, srcRect,
        pDst,
        planeOutputY.pitch, dstRect,
        factorX, factorY, 0.0, 0.0, interpMode);
    if (sts != NPP_SUCCESS) {
        return sts;
    }
    return sts;
}

RGY_ERR NVEncFilterResize::resizeNppiYUV444(RGYFrameInfo *pOutputFrame, const RGYFrameInfo *pInputFrame) {
#if _M_IX86
    AddMessage(RGY_LOG_ERROR, _T("npp filter not supported on x86.\n"));
    return RGY_ERR_UNSUPPORTED;
#else
    RGY_ERR sts = RGY_ERR_NONE;
    if (m_pParam->frameOut.csp != m_pParam->frameIn.csp) {
        AddMessage(RGY_LOG_ERROR, _T("csp does not match.\n"));
        return RGY_ERR_INVALID_PARAM;
    }
    auto pResizeParam = std::dynamic_pointer_cast<NVEncFilterParamResize>(m_pParam);
    if (!pResizeParam) {
        AddMessage(RGY_LOG_ERROR, _T("Invalid parameter type.\n"));
        return RGY_ERR_INVALID_PARAM;
    }
    const auto interp = vpp_resize_algo_rgy_to_enc(pResizeParam->interp);
    if (interp == NPPI_INTER_UNDEFINED) {
        AddMessage(RGY_LOG_ERROR, _T("Unknown nppi interp mode: %d.\n"), (int)pResizeParam->interp);
        return RGY_ERR_UNSUPPORTED;
    }
    static const auto supportedCspYUV444High = make_array<RGY_CSP>(RGY_CSP_YUV444_09, RGY_CSP_YUV444_10, RGY_CSP_YUV444_12, RGY_CSP_YUV444_14, RGY_CSP_YUV444_16);
    NppStatus nppsts = NPP_SUCCESS;
    if (m_pParam->frameIn.csp == RGY_CSP_YUV444) {
        nppsts = resize_nppi_yuv444<Npp8u>(pOutputFrame, pInputFrame, nppiResizeSqrPixel_8u_P3R, interp);
        if (nppsts != NPP_SUCCESS) {
            AddMessage(RGY_LOG_ERROR, _T("failed to resize: %d, %s.\n"), nppsts, char_to_tstring(_cudaGetErrorEnum(nppsts)).c_str());
            sts = RGY_ERR_CUDA;
        }
    } else if (std::find(supportedCspYUV444High.begin(), supportedCspYUV444High.end(), m_pParam->frameIn.csp) != supportedCspYUV444High.end()) {
        nppsts = resize_nppi_yuv444<Npp16u>(pOutputFrame, pInputFrame, nppiResizeSqrPixel_16u_P3R, interp);
        if (nppsts != NPP_SUCCESS) {
            AddMessage(RGY_LOG_ERROR, _T("failed to resize: %d, %s.\n"), nppsts, char_to_tstring(_cudaGetErrorEnum(nppsts)).c_str());
            sts = RGY_ERR_CUDA;
        }
    } else {
        AddMessage(RGY_LOG_ERROR, _T("unsupported csp.\n"));
        sts = RGY_ERR_UNSUPPORTED;
    }
    return sts;
#endif
}

NVEncFilterResize::NVEncFilterResize() : m_bInterlacedWarn(false), m_weightSpline(), m_nvvfxSuperRes() {
    m_sFilterName = _T("resize");
}

NVEncFilterResize::~NVEncFilterResize() {
    close();
}

RGY_ERR NVEncFilterResize::initNvvfxFilter(NVEncFilterParamResize *param) {
    const tstring filter_name = get_cx_desc(list_vpp_resize, param->interp);
    const double target_scale_ratio = std::max(
        param->frameOut.width / (double)param->frameIn.width,
        param->frameOut.height / (double)param->frameIn.height);
    if (target_scale_ratio < 1.0) {
        AddMessage(RGY_LOG_ERROR, _T("%s not supported for resize ratio below 1.0.\n"), filter_name.c_str());
        return RGY_ERR_UNSUPPORTED;
    }
    static const auto nvvfx_scale_ratio = make_array<rgy_rational<int>>(
        rgy_rational<int>(4, 3),
        rgy_rational<int>(3, 2),
        rgy_rational<int>(2, 1),
        rgy_rational<int>(3, 1),
        rgy_rational<int>(4, 1)
    );

    for (const auto& ratio : nvvfx_scale_ratio) {
        double ratiod = ratio.qdouble();
        if (param->frameIn.height % ratio.d() != 0) {
            continue; // 割り切れない場合は使用しない
        }
        if (ratiod >= target_scale_ratio * (1.0 - 1e-3)
            || ratio == nvvfx_scale_ratio.back()) {
            unique_ptr<NVEncFilterNvvfxSuperRes> filter(new NVEncFilterNvvfxSuperRes());
            param->nvvfxSuperRes->frameIn = param->frameIn;
            param->nvvfxSuperRes->frameOut = param->frameIn;
            param->nvvfxSuperRes->frameOut.width = param->frameIn.width * ratio.n() / ratio.d();
            param->nvvfxSuperRes->frameOut.height = param->frameIn.height * ratio.n() / ratio.d();
            param->nvvfxSuperRes->baseFps = param->baseFps;
            param->nvvfxSuperRes->frameIn.deivce_mem = true;
            param->nvvfxSuperRes->frameOut.deivce_mem = true;
            param->nvvfxSuperRes->bOutOverwrite = false;
            auto sts = filter->init(param->nvvfxSuperRes, m_pPrintMes);
            if (sts == RGY_ERR_NONE) {
                m_nvvfxSuperRes = std::move(filter);
                AddMessage(RGY_LOG_DEBUG, _T("created %s.\n"), m_nvvfxSuperRes->GetInputMessage().c_str());
                return RGY_ERR_NONE;
            }
            AddMessage(RGY_LOG_WARN, _T("Failed to init nvvfx-superres with ratio %.1f, retrying with other ratios...\n"), ratio.qdouble());
        }
    }
    AddMessage(RGY_LOG_ERROR, _T("Suitable ratio for nvvfx-superres not found.\n"));
    return RGY_ERR_UNSUPPORTED;
}

RGY_ERR NVEncFilterResize::init(shared_ptr<NVEncFilterParam> pParam, shared_ptr<RGYLog> pPrintMes) {
    RGY_ERR sts = RGY_ERR_NONE;
    m_pPrintMes = pPrintMes;
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

    auto cudaerr = AllocFrameBuf(pResizeParam->frameOut, 1);
    if (cudaerr != cudaSuccess) {
        AddMessage(RGY_LOG_ERROR, _T("failed to allocate memory: %s.\n"), char_to_tstring(cudaGetErrorName(cudaerr)).c_str());
        return RGY_ERR_MEMORY_ALLOC;
    }
    pResizeParam->frameOut.pitch = m_pFrameBuf[0]->frame.pitch;

    auto resizeInterp = pResizeParam->interp;
    if (isNvvfxResizeFiter(pResizeParam->interp)) {
        if (!pResizeParam->nvvfxSuperRes) {
            AddMessage(RGY_LOG_ERROR, _T("nvvfx parameter unknown.\n"));
            return RGY_ERR_UNKNOWN;
        }
        sts = initNvvfxFilter(pResizeParam.get());
        if (sts != RGY_ERR_NONE) {
            AddMessage(RGY_LOG_ERROR, _T("Failed to init nvvfx filter.\n"));
            return sts;
        }
        resizeInterp = pResizeParam->nvvfxSubAlgo;
    }

    if (m_weightSpline.ptr == nullptr
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

        m_weightSpline = CUMemBuf(sizeof((*weight)[0]) * weight->size());
        if (cudaSuccess != (cudaerr = m_weightSpline.alloc())) {
            AddMessage(RGY_LOG_ERROR, _T("failed to allocate memory: %s.\n"), char_to_tstring(cudaGetErrorName(cudaerr)).c_str());
            return RGY_ERR_MEMORY_ALLOC;
        }
        cudaerr = cudaMemcpy(m_weightSpline.ptr, weight->data(), m_weightSpline.nSize, cudaMemcpyHostToDevice);
        if (cudaerr != cudaSuccess) {
            AddMessage(RGY_LOG_ERROR, _T("failed to send weight to gpu memory: %s.\n"), char_to_tstring(cudaGetErrorName(cudaerr)).c_str());
            return RGY_ERR_CUDA;
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
    } else {
        info = pResizeParam->print();
    }
    setFilterInfo(info);

    //コピーを保存
    m_pParam = pResizeParam;
    return sts;
}


NVEncFilterParamResize::NVEncFilterParamResize() :
    nvvfxSuperRes(), nvvfxSubAlgo(RGY_VPP_RESIZE_SPLINE36), interp(RGY_VPP_RESIZE_SPLINE36) {
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
    return strsprintf(_T("resize(%s): %dx%d -> %dx%d"),
        get_chr_from_value(list_vpp_resize, interp),
        frameIn.width, frameIn.height,
        frameOut.width, frameOut.height);
}

RGY_ERR NVEncFilterResize::run_filter(const RGYFrameInfo *pInputFrame, RGYFrameInfo **ppOutputFrames, int *pOutputFrameNum, cudaStream_t stream) {
    RGY_ERR sts = RGY_ERR_NONE;
    if (pInputFrame->ptr == nullptr) {
        return sts;
    }

    *pOutputFrameNum = 1;
    if (ppOutputFrames[0] == nullptr) {
        auto pOutFrame = m_pFrameBuf[m_nFrameIdx].get();
        ppOutputFrames[0] = &pOutFrame->frame;
        m_nFrameIdx = (m_nFrameIdx + 1) % m_pFrameBuf.size();
    }
    ppOutputFrames[0]->picstruct = pInputFrame->picstruct;
    if (interlaced(*pInputFrame)) {
        return filter_as_interlaced_pair(pInputFrame, ppOutputFrames[0], stream);
    }
    const auto memcpyKind = getCudaMemcpyKind(pInputFrame->deivce_mem, ppOutputFrames[0]->deivce_mem);
    if (memcpyKind != cudaMemcpyDeviceToDevice) {
        AddMessage(RGY_LOG_ERROR, _T("only supported on device memory.\n"));
        return RGY_ERR_UNSUPPORTED;
    }
    if (m_pParam->frameOut.csp != m_pParam->frameIn.csp) {
        AddMessage(RGY_LOG_ERROR, _T("csp does not match.\n"));
        return RGY_ERR_UNSUPPORTED;
    }

    auto pResizeParam = std::dynamic_pointer_cast<NVEncFilterParamResize>(m_pParam);
    if (!pResizeParam) {
        AddMessage(RGY_LOG_ERROR, _T("Invalid parameter type.\n"));
        return RGY_ERR_INVALID_PARAM;
    }

    auto resizeInterp = pResizeParam->interp;
    if (m_nvvfxSuperRes) {
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
        resizeInterp = pResizeParam->nvvfxSubAlgo;
    }

    static const auto supportedCspYV12   = make_array<RGY_CSP>(RGY_CSP_YV12, RGY_CSP_YV12_09, RGY_CSP_YV12_10, RGY_CSP_YV12_12, RGY_CSP_YV12_14, RGY_CSP_YV12_16);
    static const auto supportedCspYUV444 = make_array<RGY_CSP>(RGY_CSP_YUV444, RGY_CSP_YUV444_09, RGY_CSP_YUV444_10, RGY_CSP_YUV444_12, RGY_CSP_YUV444_14, RGY_CSP_YUV444_16);

    if (   pInputFrame->width != ppOutputFrames[0]->width
        || pInputFrame->height != ppOutputFrames[0]->height) {
        if (isNppResizeFiter(pResizeParam->interp)) {
            if (std::find(supportedCspYV12.begin(), supportedCspYV12.end(), m_pParam->frameIn.csp) != supportedCspYV12.end()) {
                sts = resizeNppiYV12(ppOutputFrames[0], pInputFrame);
            } else if (std::find(supportedCspYUV444.begin(), supportedCspYUV444.end(), m_pParam->frameIn.csp) != supportedCspYUV444.end()) {
                sts = resizeNppiYUV444(ppOutputFrames[0], pInputFrame);
            } else {
                AddMessage(RGY_LOG_ERROR, _T("unsupported csp.\n"));
                sts = RGY_ERR_UNSUPPORTED;
            }
        } else {
            static const std::map<RGY_CSP, decltype(resize_frame<uint8_t, 8>)*> resize_list = {
                { RGY_CSP_YV12,       resize_frame<uint8_t,   8> },
                { RGY_CSP_YV12_16,    resize_frame<uint16_t, 16> },
                { RGY_CSP_YUV444,     resize_frame<uint8_t,   8> },
                { RGY_CSP_YUV444_16,  resize_frame<uint16_t, 16> },
                { RGY_CSP_YUVA444,    resize_frame<uint8_t,   8> },
                { RGY_CSP_YUVA444_16, resize_frame<uint16_t, 16> },
            };
            if (resize_list.count(pInputFrame->csp) == 0) {
                AddMessage(RGY_LOG_ERROR, _T("unsupported csp %s.\n"), RGY_CSP_NAMES[pInputFrame->csp]);
                return RGY_ERR_UNSUPPORTED;
            }
            sts = resize_list.at(pInputFrame->csp)(ppOutputFrames[0], pInputFrame, resizeInterp, (float *)m_weightSpline.ptr, stream);
            if (sts != RGY_ERR_NONE) {
                AddMessage(RGY_LOG_ERROR, _T("error at resize(%s): %s.\n"),
                    RGY_CSP_NAMES[pInputFrame->csp],
                    get_err_mes(sts));
                return sts;
            }
        }
    } else {
        auto cudaerr = copyFrameAsync(ppOutputFrames[0], pInputFrame, stream);
        if (cudaerr != cudaSuccess) {
            AddMessage(RGY_LOG_ERROR, _T("Failed to copy frame.\n"));
            return RGY_ERR_CUDA;
        }
    }
    return sts;
}

void NVEncFilterResize::close() {
    m_pFrameBuf.clear();
    m_bInterlacedWarn = false;
}
