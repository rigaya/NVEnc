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
#include <limits>
#include "convert_csp.h"
#include "NVEncFilterWarpsharp.h"
#include "NVEncParam.h"
#pragma warning (push)
#pragma warning (disable: 4819)
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#pragma warning (pop)
#include "rgy_cuda_util_kernel.h"

static const int WARPSHARP_BLOCK_X = 32;
static const int WARPSHARP_BLOCK_Y = 16;

template<typename Type, int bit_depth>
__global__ void kernel_sobel(
    uint8_t *__restrict__ pDst, const int dstPitch,
    const uint8_t *__restrict__ pSrc, const int srcPitch,
    const int width, const int height,
    const int threshold) {
    const int ix = blockIdx.x * blockDim.x + threadIdx.x;
    const int iy = blockIdx.y * blockDim.y + threadIdx.y;
    const int pixel_max = (1 << (bit_depth)) - 1;

    if (ix < width && iy < height) {
        const int x0 = max(ix - 1, 0);
        const int x1 = ix;
        const int x2 = min(ix + 1, width - 1);
        const int y0 = max(iy - 1, 0);
        const int y1 = iy;
        const int y2 = min(iy + 1, height - 1);

        // p00 01 02
        //  10 11 12
        //  20 21 22
        #define SRC(x, y) *(Type *)(pSrc + (y) * srcPitch + (x) * sizeof(Type))
        const int p00 = SRC(x0, y0);
        const int p01 = SRC(x1, y0);
        const int p02 = SRC(x2, y0);
        const int p10 = SRC(x0, y1);
        //const int p11 = SRC(x1, y1);
        const int p12 = SRC(x2, y1);
        const int p20 = SRC(x0, y2);
        const int p21 = SRC(x1, y2);
        const int p22 = SRC(x2, y2);
        #undef SRC

        const int avg_u = (p01 + ((p00 + p02 + 1) >> 1) + 1) >> 1;
        const int avg_d = (p21 + ((p20 + p22 + 1) >> 1) + 1) >> 1;
        const int avg_l = (p10 + ((p00 + p20 + 1) >> 1) + 1) >> 1;
        const int avg_r = (p12 + ((p02 + p22 + 1) >> 1) + 1) >> 1;
        const int abs_v = abs(avg_u - avg_d);
        const int abs_h = abs(avg_l - avg_r);
        const int abs_max = max(abs_v, abs_h);

        int absolute = min(abs_v + abs_h, pixel_max);
        absolute = min(absolute + abs_max, pixel_max);
        absolute = min(min(absolute * 2, pixel_max) + absolute, pixel_max);
        absolute = min(absolute * 2, pixel_max);
        absolute = min(absolute, threshold);

        Type* ptr = (Type*)(pDst + iy * dstPitch + ix * sizeof(Type));
        ptr[0] = (Type)absolute;
    }
}

template<int range>
__device__ __inline__
float calc_blur(float center, float avg_range[range]) {
    static_assert(range == 6 || range == 2, "range == 6 || range == 2");
    if (range == 6) {
        float avg012 = (avg_range[0] + avg_range[1]) * 0.25f + center * 0.5f;
        float avg3456 = (avg_range[2] + avg_range[3] + avg_range[4] + avg_range[5]) * 0.25f;
        float avg0123456 = (avg012 + avg3456) * 0.5f;
        return (avg012 + avg0123456) * 0.5f;
    } else if (range == 2) {
        return center * 0.5f + avg_range[0] * 0.375f + avg_range[1] * 0.125f;
    }
    return 0.0f;
}


template<typename Type, int bit_depth, int range>
__global__ void kernel_blur(
    uint8_t *__restrict__ pDst, const int dstPitch,
    const uint8_t *__restrict__ pSrc, const int srcPitch,
    const int width, const int height) {
    const int lx = threadIdx.x;
    const int ly = threadIdx.y;
    const int imgx = blockIdx.x * blockDim.x + threadIdx.x;
    const int imgy = blockIdx.y * blockDim.y + threadIdx.y;
    #define SY_SIZE (WARPSHARP_BLOCK_Y + range * 2)
    #define SX_SIZE (WARPSHARP_BLOCK_X + range * 2)
    __shared__ float stmp0[SY_SIZE][SX_SIZE];
    __shared__ float stmp1[SY_SIZE][WARPSHARP_BLOCK_X];
    for (int sy = ly, loady = imgy - range; sy < SY_SIZE; sy += blockDim.y, loady += blockDim.y) {
        for (int sx = lx, loadx = imgx - range; sx < SX_SIZE; sx += blockDim.x, loadx += blockDim.x) {
            const int y = clamp(loady, 0, height - 1);
            const int x = clamp(loadx, 0, width - 1);
            Type value = *(const Type*)(pSrc + y * srcPitch + x * sizeof(Type));
            stmp0[sy][sx] = (float)value;
        }
    }
    __syncthreads();

    // 横方向
    for (int sy = ly; sy < SY_SIZE; sy += blockDim.y) {
        float avg_range[range];
        #pragma unroll
        for (int i = 1; i <= range; i++) {
            avg_range[i-1] = (stmp0[sy][lx + range - i] + stmp0[sy][lx + range + i]) * 0.5f;
        }
        stmp1[sy][lx] = calc_blur<range>(stmp0[sy][lx + range], avg_range);
    }
    __syncthreads();

    // 縦方向
    if (imgx < width && imgy < height) {
        float avg_range[range];
        #pragma unroll
        for (int i = 1; i <= range; i++) {
            avg_range[i-1] = (stmp1[ly + range - i][lx] + stmp1[ly + range + i][lx]) * 0.5f;
        }
        float avg = calc_blur<range>(stmp1[ly + range][lx], avg_range);

        Type* ptr = (Type*)(pDst + imgy * dstPitch + imgx * sizeof(Type));
        ptr[0] = (Type)clamp((int)(avg + 0.5f), 0, (1<<bit_depth)-1);
    }
    #undef SY_SIZE
    #undef SX_SIZE
}

template<typename Type, int bit_depth>
__global__ void kernel_warp(
    uint8_t *__restrict__ pDst, const int dstPitch,
    cudaTextureObject_t texSrc,
    const uint8_t *__restrict__ pEdge, const int edgePitch,
    const int width, const int height,
    const float depth) {
    const int imgx = blockIdx.x * blockDim.x + threadIdx.x;
    const int imgy = blockIdx.y * blockDim.y + threadIdx.y;

    if (imgx < width && imgy < height) {
        pDst  += imgy * dstPitch  + imgx * sizeof(Type);
        pEdge += imgy * edgePitch + imgx * sizeof(Type);

        const int above = *(Type *)((imgy == 0)          ? pEdge : pEdge - edgePitch);
        const int below = *(Type *)((imgy == height - 1) ? pEdge : pEdge + edgePitch);
        const int left  = *(Type *)((imgx == 0)          ? pEdge : pEdge - sizeof(Type));
        const int right = *(Type *)((imgx == width - 1)  ? pEdge : pEdge + sizeof(Type));

        float h = (float)(left - right);
        float v = (float)(above - below);

        h *= depth * ((1.0f / 256.0f) / (float)(1 << (bit_depth - 8)));
        v *= depth * ((1.0f / 256.0f) / (float)(1 << (bit_depth - 8)));

        float val = tex2D<float>(texSrc, imgx + 0.5f + h, imgy + 0.5f + v);

        *(Type *)pDst = (Type)(clamp(val, 0.0f, 1.0f - RGY_FLT_EPS) * ((1 << bit_depth)-1));
    }
}

template<typename Type>
__global__ void kernel_downscale(
    uint8_t *__restrict__ pDst, const int dstPitch, const int dstWidth, const int dstHeight,
    const uint8_t *__restrict__ pSrc, const int srcPitch) {
    const int idstx = blockIdx.x * blockDim.x + threadIdx.x;
    const int idsty = blockIdx.y * blockDim.y + threadIdx.y;

    if (idstx < dstWidth && idsty < dstHeight) {
        const int isrcx = idstx << 1;
        const int isrcy = idsty << 1;
        pSrc += isrcy * srcPitch + isrcx * sizeof(Type);

        int srcY0 = *(Type*)(pSrc);
        int srcY1 = *(Type*)(pSrc + srcPitch);

        pDst += idsty * dstPitch + idstx * sizeof(Type);
        *(Type*)pDst = (Type)((srcY0 + srcY1 + 1) >> 1);
    }
}

template<typename Type>
cudaError_t textureCreateWarpsharp(cudaTextureObject_t &tex, cudaTextureFilterMode filterMode, cudaTextureReadMode readMode, uint8_t *ptr, int pitch, int width, int height) {
    cudaResourceDesc resDesc;
    memset(&resDesc, 0, sizeof(resDesc));
    resDesc.resType = cudaResourceTypePitch2D;
    resDesc.res.pitch2D.devPtr = ptr;
    resDesc.res.pitch2D.pitchInBytes = pitch;
    resDesc.res.pitch2D.width = width;
    resDesc.res.pitch2D.height = height;
    resDesc.res.pitch2D.desc = cudaCreateChannelDesc<Type>();

    cudaTextureDesc texDesc;
    memset(&texDesc, 0, sizeof(texDesc));
    texDesc.addressMode[0] = cudaAddressModeClamp;
    texDesc.addressMode[1] = cudaAddressModeClamp;
    texDesc.filterMode = filterMode;
    texDesc.readMode = readMode;
    texDesc.normalizedCoords = 0;

    return cudaCreateTextureObject(&tex, &resDesc, &texDesc, nullptr);
}

template<typename Type, int bit_depth>
static RGY_ERR warpsharp_sobel_plane(RGYFrameInfo *pOutputFrame, const RGYFrameInfo *pInputFrame, const float threshold, cudaStream_t stream) {
    dim3 blockSize(WARPSHARP_BLOCK_X, WARPSHARP_BLOCK_Y);
    dim3 gridSize(divCeil(pOutputFrame->width, blockSize.x), divCeil(pOutputFrame->height, blockSize.y));

    kernel_sobel<Type, bit_depth> << <gridSize, blockSize, 0, stream >> > (
        (uint8_t*)pOutputFrame->ptr, pOutputFrame->pitch,
        (uint8_t*)pInputFrame->ptr, pInputFrame->pitch,
        pOutputFrame->width, pOutputFrame->height,
        (int)(threshold * (1<<(bit_depth - 8)) + 0.5f));
    auto cudaerr = cudaGetLastError();
    if (cudaerr != cudaSuccess) {
        return err_to_rgy(cudaerr);
    }
    return RGY_ERR_NONE;
}

template<typename Type, int bit_depth, int range>
static RGY_ERR warpsharp_blur_plane(RGYFrameInfo *pOutputFrame, const RGYFrameInfo *pInputFrame, cudaStream_t stream) {
    dim3 blockSize(WARPSHARP_BLOCK_X, WARPSHARP_BLOCK_Y);
    dim3 gridSize(divCeil(pOutputFrame->width, blockSize.x), divCeil(pOutputFrame->height, blockSize.y));

    kernel_blur<Type, bit_depth, range> << <gridSize, blockSize, 0, stream >> > (
        (uint8_t*)pOutputFrame->ptr, pOutputFrame->pitch,
        (uint8_t*)pInputFrame->ptr, pInputFrame->pitch,
        pOutputFrame->width, pOutputFrame->height);
    auto cudaerr = cudaGetLastError();
    if (cudaerr != cudaSuccess) {
        return err_to_rgy(cudaerr);
    }
    return RGY_ERR_NONE;
}

template<typename Type, int bit_depth>
static RGY_ERR warpsharp_warp_plane(RGYFrameInfo *pOutputFrame, const RGYFrameInfo *pMaskFrame, const RGYFrameInfo *pInputFrame, const float depth, cudaStream_t stream) {
    dim3 blockSize(WARPSHARP_BLOCK_X, WARPSHARP_BLOCK_Y);
    dim3 gridSize(divCeil(pOutputFrame->width, blockSize.x), divCeil(pOutputFrame->height, blockSize.y));

    cudaTextureObject_t texSrc = 0;
    auto cudaerr = textureCreateWarpsharp<Type>(texSrc, cudaFilterModeLinear, cudaReadModeNormalizedFloat, pInputFrame->ptr, pInputFrame->pitch, pInputFrame->width, pInputFrame->height);
    if (cudaerr != cudaSuccess) {
        return err_to_rgy(cudaerr);
    }
    kernel_warp<Type, bit_depth> << <gridSize, blockSize, 0, stream >> > (
        (uint8_t*)pOutputFrame->ptr, pOutputFrame->pitch,
        texSrc,
        (uint8_t*)pMaskFrame->ptr, pMaskFrame->pitch,
        pOutputFrame->width, pOutputFrame->height,
        depth);
    cudaerr = cudaGetLastError();
    if (cudaerr != cudaSuccess) {
        return err_to_rgy(cudaerr);
    }
    cudaerr = cudaDestroyTextureObject(texSrc);
    if (cudaerr != cudaSuccess) {
    }
    return RGY_ERR_NONE;
}

template<typename Type>
static RGY_ERR warpsharp_downscale_plane(RGYFrameInfo *pOutputFrame, const RGYFrameInfo *pInputFrame, cudaStream_t stream) {
    dim3 blockSize(WARPSHARP_BLOCK_X, WARPSHARP_BLOCK_Y);
    dim3 gridSize(divCeil(pOutputFrame->width, blockSize.x), divCeil(pOutputFrame->height, blockSize.y));

    kernel_downscale<Type> << <gridSize, blockSize, 0, stream >> > (
        (uint8_t*)pOutputFrame->ptr, pOutputFrame->pitch, pOutputFrame->width, pOutputFrame->height,
        (uint8_t*)pInputFrame->ptr, pInputFrame->pitch);
    auto cudaerr = cudaGetLastError();
    if (cudaerr != cudaSuccess) {
        return err_to_rgy(cudaerr);
    }
    return RGY_ERR_NONE;
}

template<typename Type, int bit_depth>
static RGY_ERR warpsharp_plane(RGYFrameInfo *pOutputFrame, RGYFrameInfo *pMaskFrame0, RGYFrameInfo *pMaskFrame1, const RGYFrameInfo *pInputFrame,
    const float threshold, const float depth, const int blur, const int type, cudaStream_t stream) {
#if 1
    auto err = warpsharp_sobel_plane<Type, bit_depth>(pMaskFrame0, pInputFrame, threshold, stream);
    if (err != RGY_ERR_NONE) {
        return err;
    }

    for (int i = 0; i < blur; i++) {
        err = (type == 0) ? warpsharp_blur_plane<Type, bit_depth, 6>(pMaskFrame1, pMaskFrame0, stream)
                          : warpsharp_blur_plane<Type, bit_depth, 2>(pMaskFrame1, pMaskFrame0, stream);
        if (err != RGY_ERR_NONE) {
            return err;
        }
        std::swap(pMaskFrame1, pMaskFrame0);
    }
    err = warpsharp_warp_plane<Type, bit_depth>(pOutputFrame, pMaskFrame0, pInputFrame, depth, stream);
#elif 0
    auto err = warpsharp_sobel_plane<Type, bit_depth>(pOutputFrame, pInputFrame, threshold, stream);
#else
    auto err = warpsharp_sobel_plane<Type, bit_depth>(pMaskFrame0, pInputFrame, threshold, stream);
    if (err != RGY_ERR_NONE) {
        return err;
    }
    err = warpsharp_blur_plane<Type, bit_depth, 6>(pOutputFrame, pMaskFrame0, stream);
#endif
    if (err != RGY_ERR_NONE) {
        return err;
    }
    return RGY_ERR_NONE;
}

template<typename Type, int bit_depth>
static RGY_ERR warpsharp_frame(RGYFrameInfo *pOutputFrame, RGYFrameInfo *pMaskFrame0, RGYFrameInfo *pMaskFrame1, const RGYFrameInfo *pInputFrame,
    const float threshold, const float depth, const int blur, const int type, const int chroma, cudaStream_t stream) {
    const auto planeInputY = getPlane(pInputFrame, RGY_PLANE_Y);
    const auto planeInputU = getPlane(pInputFrame, RGY_PLANE_U);
    const auto planeInputV = getPlane(pInputFrame, RGY_PLANE_V);
    auto planeMask0Y = getPlane(pMaskFrame0, RGY_PLANE_Y);
    auto planeMask0U = getPlane(pMaskFrame0, RGY_PLANE_U);
    auto planeMask0V = getPlane(pMaskFrame0, RGY_PLANE_V);
    auto planeMask1Y = getPlane(pMaskFrame1, RGY_PLANE_Y);
    auto planeMask1U = getPlane(pMaskFrame1, RGY_PLANE_U);
    auto planeMask1V = getPlane(pMaskFrame1, RGY_PLANE_V);
    auto planeOutputY = getPlane(pOutputFrame, RGY_PLANE_Y);
    auto planeOutputU = getPlane(pOutputFrame, RGY_PLANE_U);
    auto planeOutputV = getPlane(pOutputFrame, RGY_PLANE_V);
    auto err = warpsharp_plane<Type, bit_depth>(&planeOutputY, &planeMask0Y, &planeMask1Y, &planeInputY, threshold, depth, blur, type, stream);
    if (err != RGY_ERR_NONE) {
        return err;
    }
    const float depthUV = (RGY_CSP_CHROMA_FORMAT[pOutputFrame->csp] == RGY_CHROMAFMT_YUV420) ? depth * 0.5f : depth;
    if (chroma == 0) {
        RGYFrameInfo *pMaskUV = &planeMask0Y;
        if (RGY_CSP_CHROMA_FORMAT[pOutputFrame->csp] == RGY_CHROMAFMT_YUV420) {
            err = warpsharp_downscale_plane<Type>(&planeMask0U, &planeMask0Y, stream);
            if (err != RGY_ERR_NONE) {
                return err;
            }
            pMaskUV = &planeMask0U;
        }
        err = warpsharp_warp_plane<Type, bit_depth>(&planeOutputU, pMaskUV, &planeInputU, depthUV, stream);
        if (err != RGY_ERR_NONE) {
            return err;
        }
        err = warpsharp_warp_plane<Type, bit_depth>(&planeOutputV, pMaskUV, &planeInputV, depthUV, stream);
        if (err != RGY_ERR_NONE) {
            return err;
        }
    } else {
        err = warpsharp_plane<Type, bit_depth>(&planeOutputU, &planeMask0U, &planeMask1U, &planeInputU, threshold, depthUV, blur, type, stream);
        if (err != RGY_ERR_NONE) {
            return err;
        }
        err = warpsharp_plane<Type, bit_depth>(&planeOutputV, &planeMask0V, &planeMask1V, &planeInputV, threshold, depthUV, blur, type, stream);
        if (err != RGY_ERR_NONE) {
            return err;
        }
    }
    return RGY_ERR_NONE;
}

NVEncFilterWarpsharp::NVEncFilterWarpsharp() : m_mask() {
    m_sFilterName = _T("warpsharp");
}

NVEncFilterWarpsharp::~NVEncFilterWarpsharp() {
    close();
}

RGY_ERR NVEncFilterWarpsharp::checkParam(const std::shared_ptr<NVEncFilterParamWarpsharp> prm) {
    //パラメータチェック
    if (prm->frameOut.height <= 0 || prm->frameOut.width <= 0) {
        AddMessage(RGY_LOG_ERROR, _T("Invalid parameter.\n"));
        return RGY_ERR_INVALID_PARAM;
    }
    if (prm->warpsharp.threshold < 0.0f || 255.0f < prm->warpsharp.threshold) {
        prm->warpsharp.threshold = clamp(prm->warpsharp.threshold, 0.0f, 255.0f);
        AddMessage(RGY_LOG_WARN, _T("threshold should be in range of %.1f - %.1f.\n"), 0.0f, 255.0f);
    }
    if (prm->warpsharp.blur < 0) {
        prm->warpsharp.blur = std::max(prm->warpsharp.blur, 0);
        AddMessage(RGY_LOG_WARN, _T("blur should be a positive value.\n"));
    }
    if (prm->warpsharp.type < 0 || 1 < prm->warpsharp.type) {
        prm->warpsharp.type = clamp(prm->warpsharp.type, 0, 1);
        AddMessage(RGY_LOG_WARN, _T("type should be in range of %d - %d.\n"), 0, 1);
    }
    if (prm->warpsharp.depth < -128.0f || 128.0f < prm->warpsharp.depth) {
        prm->warpsharp.depth = clamp(prm->warpsharp.depth, -128.0f, 128.0f);
        AddMessage(RGY_LOG_WARN, _T("depth should be in range of %.1f - %.1f.\n"), -128.0f, 128.0f);
    }
    if (prm->warpsharp.chroma < 0 || 1 < prm->warpsharp.chroma) {
        prm->warpsharp.chroma = clamp(prm->warpsharp.chroma, 0, 1);
        AddMessage(RGY_LOG_WARN, _T("chroma should be in range of %d - %d.\n"), 0, 1);
    }
    return RGY_ERR_NONE;
}

RGY_ERR NVEncFilterWarpsharp::init(shared_ptr<NVEncFilterParam> pParam, shared_ptr<RGYLog> pPrintMes) {
    RGY_ERR sts = RGY_ERR_NONE;
    m_pPrintMes = pPrintMes;
    auto prm = std::dynamic_pointer_cast<NVEncFilterParamWarpsharp>(pParam);
    if (!prm) {
        AddMessage(RGY_LOG_ERROR, _T("Invalid parameter type.\n"));
        return RGY_ERR_INVALID_PARAM;
    }
    //パラメータチェック
    if ((sts = checkParam(prm)) != RGY_ERR_NONE) {
        return sts;
    }
    if (!m_pParam || std::dynamic_pointer_cast<NVEncFilterParamWarpsharp>(m_pParam)->warpsharp != prm->warpsharp) {

        auto cudaerr = AllocFrameBuf(prm->frameOut, 1);
        if (cudaerr != cudaSuccess) {
            AddMessage(RGY_LOG_ERROR, _T("failed to allocate memory: %s.\n"), char_to_tstring(cudaGetErrorName(cudaerr)).c_str());
            return RGY_ERR_MEMORY_ALLOC;
        }
        prm->frameOut.pitch = m_pFrameBuf[0]->frame.pitch;
    }

    if (cmpFrameInfoCspResolution(&m_mask[0].frame, &prm->frameOut)) {
        for (auto& m : m_mask) {
            m.frame.width = prm->frameOut.width;
            m.frame.height = prm->frameOut.height;
            m.frame.pitch = prm->frameOut.pitch;
            m.frame.picstruct = prm->frameOut.picstruct;
            m.frame.deivce_mem = prm->frameOut.deivce_mem;
            m.frame.csp = prm->frameOut.csp;
            auto cudaerr = m.alloc();
            if (cudaerr != cudaSuccess) {
                AddMessage(RGY_LOG_ERROR, _T("failed to allocate memory: %s.\n"), char_to_tstring(cudaGetErrorName(cudaerr)).c_str());
                return RGY_ERR_MEMORY_ALLOC;
            }
        }
    }

    setFilterInfo(pParam->print());
    m_pParam = prm;
    return sts;
}

tstring NVEncFilterParamWarpsharp::print() const {
    return warpsharp.print();
}

RGY_ERR NVEncFilterWarpsharp::run_filter(const RGYFrameInfo *pInputFrame, RGYFrameInfo **ppOutputFrames, int *pOutputFrameNum, cudaStream_t stream) {
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
        return filter_as_interlaced_pair(pInputFrame, ppOutputFrames[0], cudaStreamDefault);
    }
    const auto memcpyKind = getCudaMemcpyKind(pInputFrame->deivce_mem, ppOutputFrames[0]->deivce_mem);
    if (memcpyKind != cudaMemcpyDeviceToDevice) {
        AddMessage(RGY_LOG_ERROR, _T("only supported on device memory.\n"));
        return RGY_ERR_INVALID_PARAM;
    }
    if (m_pParam->frameOut.csp != m_pParam->frameIn.csp) {
        AddMessage(RGY_LOG_ERROR, _T("csp does not match.\n"));
        return RGY_ERR_INVALID_PARAM;
    }
    auto prm = std::dynamic_pointer_cast<NVEncFilterParamWarpsharp>(m_pParam);
    if (!prm) {
        AddMessage(RGY_LOG_ERROR, _T("Invalid parameter type.\n"));
        return RGY_ERR_INVALID_PARAM;
    }

    static const std::map<RGY_CSP, decltype(warpsharp_frame<uint8_t, 8>)*> warpsharp_list = {
        { RGY_CSP_YV12,      warpsharp_frame<uint8_t,   8> },
        { RGY_CSP_YV12_16,   warpsharp_frame<uint16_t, 16> },
        { RGY_CSP_YUV444,    warpsharp_frame<uint8_t,   8> },
        { RGY_CSP_YUV444_16, warpsharp_frame<uint16_t, 16> }
    };
    if (warpsharp_list.count(pInputFrame->csp) == 0) {
        AddMessage(RGY_LOG_ERROR, _T("unsupported csp %s.\n"), RGY_CSP_NAMES[pInputFrame->csp]);
        return RGY_ERR_UNSUPPORTED;
    }
    sts = warpsharp_list.at(pInputFrame->csp)(ppOutputFrames[0], &m_mask[0].frame, &m_mask[1].frame, pInputFrame,
        prm->warpsharp.threshold, prm->warpsharp.depth, prm->warpsharp.blur, prm->warpsharp.type, prm->warpsharp.chroma,
        stream);
    if (sts != RGY_ERR_NONE) {
        AddMessage(RGY_LOG_ERROR, _T("error at warpsharp(%s): %s.\n"),
            RGY_CSP_NAMES[pInputFrame->csp],
            get_err_mes(sts));
        return sts;
    }
    return sts;
}

void NVEncFilterWarpsharp::close() {
    m_pFrameBuf.clear();
    for (auto& m : m_mask) {
        m.clear();
    }
}
