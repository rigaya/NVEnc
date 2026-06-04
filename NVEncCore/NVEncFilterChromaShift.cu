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

#include <algorithm>
#include <map>
#include "convert_csp.h"
#include "NVEncFilterChromaShift.h"
#include "rgy_prm.h"
#pragma warning (push)
#pragma warning (disable: 4819)
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#pragma warning (pop)
#include "rgy_cuda_util_kernel.h"

static const int CHROMASHIFT_BLOCK_X = 32;
static const int CHROMASHIFT_BLOCK_Y = 8;
static const int CHROMASHIFT_LAP_GAIN = 16;
static const int CHROMASHIFT_AUTO_SEARCH_R = 4;

template<typename Type>
static cudaError_t textureCreateChromaShift(cudaTextureObject_t& tex, cudaTextureFilterMode filterMode, cudaTextureReadMode readMode,
    uint8_t *ptr, const int pitch, const int width, const int height) {
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
__global__ void kernel_chromashift_shift(uint8_t *__restrict__ pDst, const int dstPitch, const int width, const int height,
    cudaTextureObject_t texSrc, const float shift_x_chroma, const float shift_y_chroma) {
    const int ix = blockIdx.x * blockDim.x + threadIdx.x;
    const int iy = blockIdx.y * blockDim.y + threadIdx.y;
    if (ix >= width || iy >= height) return;

    const float v = tex2D<float>(texSrc, ix + 0.5f + shift_x_chroma, iy + 0.5f + shift_y_chroma);
    Type *dst = (Type *)(pDst + iy * dstPitch + ix * sizeof(Type));
    dst[0] = (Type)(clamp(v, 0.0f, 1.0f - RGY_FLT_EPS) * (float)((1 << bit_depth) - 1));
}

template<typename Type, int bit_depth>
__global__ void kernel_chromashift_laplacian(
    const uint8_t *__restrict__ pSrcC, const int srcCPitch, const int chromaW, const int chromaH,
    const int subX, const int subY, uint8_t *__restrict__ pDstY, const int dstYPitch,
    const int lumaW, const int lumaH) {
    const int ix = blockIdx.x * blockDim.x + threadIdx.x;
    const int iy = blockIdx.y * blockDim.y + threadIdx.y;
    if (ix >= lumaW || iy >= lumaH) return;

    const int cx = (subX > 0) ? (ix / subX) : ix;
    const int cy = (subY > 0) ? (iy / subY) : iy;
    const int xm = (cx > 0) ? (cx - 1) : 0;
    const int xp = (cx + 1 < chromaW) ? (cx + 1) : (chromaW - 1);
    const int ym = (cy > 0) ? (cy - 1) : 0;
    const int yp = (cy + 1 < chromaH) ? (cy + 1) : (chromaH - 1);
    const int c  = (int)(*(const Type *)(pSrcC + cy * srcCPitch + cx * sizeof(Type)));
    const int up = (int)(*(const Type *)(pSrcC + ym * srcCPitch + cx * sizeof(Type)));
    const int dn = (int)(*(const Type *)(pSrcC + yp * srcCPitch + cx * sizeof(Type)));
    const int le = (int)(*(const Type *)(pSrcC + cy * srcCPitch + xm * sizeof(Type)));
    const int ri = (int)(*(const Type *)(pSrcC + cy * srcCPitch + xp * sizeof(Type)));
    const int lap = 4 * c - up - dn - le - ri;
    const int absLap = (lap < 0) ? -lap : lap;
    const int scaled = absLap * CHROMASHIFT_LAP_GAIN;
    const int max_val = (1 << bit_depth) - 1;
    const int out = (scaled > max_val) ? max_val : scaled;

    Type *dst = (Type *)(pDstY + iy * dstYPitch + ix * sizeof(Type));
    dst[0] = (Type)out;
}

template<typename Type, int bit_depth>
__global__ void kernel_chromashift_fill_neutral(uint8_t *__restrict__ pDst, const int dstPitch, const int width, const int height) {
    const int ix = blockIdx.x * blockDim.x + threadIdx.x;
    const int iy = blockIdx.y * blockDim.y + threadIdx.y;
    if (ix >= width || iy >= height) return;

    const Type neutral = (Type)(1 << (bit_depth - 1));
    Type *dst = (Type *)(pDst + iy * dstPitch + ix * sizeof(Type));
    dst[0] = neutral;
}

__device__ __forceinline__ int chromashift_lap_clampX(int x, int w) { return (x < 0) ? 0 : (x >= w) ? (w - 1) : x; }
__device__ __forceinline__ int chromashift_lap_clampY(int y, int h) { return (y < 0) ? 0 : (y >= h) ? (h - 1) : y; }
__device__ __forceinline__ int chromashift_sign(int v) { return (v > 0) ? 1 : (v < 0) ? -1 : 0; }

template<typename Type>
__device__ __forceinline__ float chromashift_sample_chroma_bilinear(
    const uint8_t *__restrict__ pSrc, const int srcPitch, const int chromaW, const int chromaH, float fx, float fy) {
    fx = fminf(fmaxf(fx, 0.0f), (float)(chromaW - 1));
    fy = fminf(fmaxf(fy, 0.0f), (float)(chromaH - 1));
    const int x0 = (int)fx;
    const int y0 = (int)fy;
    const int x1 = (x0 + 1 < chromaW) ? (x0 + 1) : (chromaW - 1);
    const int y1 = (y0 + 1 < chromaH) ? (y0 + 1) : (chromaH - 1);
    const float fxF = fx - (float)x0;
    const float fyF = fy - (float)y0;
    const float w00 = (1.0f - fxF) * (1.0f - fyF);
    const float w10 =         fxF  * (1.0f - fyF);
    const float w01 = (1.0f - fxF) *         fyF;
    const float w11 =         fxF  *         fyF;
    return (float)(*(const Type *)(pSrc + y0 * srcPitch + x0 * sizeof(Type))) * w00
         + (float)(*(const Type *)(pSrc + y0 * srcPitch + x1 * sizeof(Type))) * w10
         + (float)(*(const Type *)(pSrc + y1 * srcPitch + x0 * sizeof(Type))) * w01
         + (float)(*(const Type *)(pSrc + y1 * srcPitch + x1 * sizeof(Type))) * w11;
}

template<typename Type>
__global__ void kernel_chromashift_lapsign_y(
    const uint8_t *__restrict__ pSrcY, const int srcYPitch, int8_t *__restrict__ pSign, const int signPitch,
    const int width, const int height) {
    const int ix = blockIdx.x * blockDim.x + threadIdx.x;
    const int iy = blockIdx.y * blockDim.y + threadIdx.y;
    if (ix >= width || iy >= height) return;

    const int xm = chromashift_lap_clampX(ix - 1, width);
    const int xp = chromashift_lap_clampX(ix + 1, width);
    const int ym = chromashift_lap_clampY(iy - 1, height);
    const int yp = chromashift_lap_clampY(iy + 1, height);
    const int c  = (int)(*(const Type *)(pSrcY + iy * srcYPitch + ix * sizeof(Type)));
    const int up = (int)(*(const Type *)(pSrcY + ym * srcYPitch + ix * sizeof(Type)));
    const int dn = (int)(*(const Type *)(pSrcY + yp * srcYPitch + ix * sizeof(Type)));
    const int le = (int)(*(const Type *)(pSrcY + iy * srcYPitch + xm * sizeof(Type)));
    const int ri = (int)(*(const Type *)(pSrcY + iy * srcYPitch + xp * sizeof(Type)));
    const int lap = 4 * c - up - dn - le - ri;
    pSign[iy * signPitch + ix] = (int8_t)chromashift_sign(lap);
}

template<typename Type>
__global__ void kernel_chromashift_lapsign_uv(
    const uint8_t *__restrict__ pSrcU, const int srcUPitch, const uint8_t *__restrict__ pSrcV, const int srcVPitch,
    const int chromaW, const int chromaH, const int subX, const int subY,
    int8_t *__restrict__ pSign, const int signPitch, const int lumaW, const int lumaH) {
    const int ix = blockIdx.x * blockDim.x + threadIdx.x;
    const int iy = blockIdx.y * blockDim.y + threadIdx.y;
    if (ix >= lumaW || iy >= lumaH) return;

    const float cx_f = ((float)ix + 0.5f) / (float)subX - 0.5f;
    const float cy_f = ((float)iy + 0.5f) / (float)subY - 0.5f;
    const float dxc  = 1.0f / (float)subX;
    const float dyc  = 1.0f / (float)subY;
    const float uC = chromashift_sample_chroma_bilinear<Type>(pSrcU, srcUPitch, chromaW, chromaH, cx_f,       cy_f);
    const float vC = chromashift_sample_chroma_bilinear<Type>(pSrcV, srcVPitch, chromaW, chromaH, cx_f,       cy_f);
    const float uU = chromashift_sample_chroma_bilinear<Type>(pSrcU, srcUPitch, chromaW, chromaH, cx_f,       cy_f - dyc);
    const float vU = chromashift_sample_chroma_bilinear<Type>(pSrcV, srcVPitch, chromaW, chromaH, cx_f,       cy_f - dyc);
    const float uD = chromashift_sample_chroma_bilinear<Type>(pSrcU, srcUPitch, chromaW, chromaH, cx_f,       cy_f + dyc);
    const float vD = chromashift_sample_chroma_bilinear<Type>(pSrcV, srcVPitch, chromaW, chromaH, cx_f,       cy_f + dyc);
    const float uL = chromashift_sample_chroma_bilinear<Type>(pSrcU, srcUPitch, chromaW, chromaH, cx_f - dxc, cy_f);
    const float vL = chromashift_sample_chroma_bilinear<Type>(pSrcV, srcVPitch, chromaW, chromaH, cx_f - dxc, cy_f);
    const float uR = chromashift_sample_chroma_bilinear<Type>(pSrcU, srcUPitch, chromaW, chromaH, cx_f + dxc, cy_f);
    const float vR = chromashift_sample_chroma_bilinear<Type>(pSrcV, srcVPitch, chromaW, chromaH, cx_f + dxc, cy_f);

    const int c  = (int)(uC + vC);
    const int up = (int)(uU + vU);
    const int dn = (int)(uD + vD);
    const int le = (int)(uL + vL);
    const int ri = (int)(uR + vR);
    pSign[iy * signPitch + ix] = (int8_t)chromashift_sign(4 * c - up - dn - le - ri);
}

__global__ void kernel_chromashift_correlate(
    const int8_t *__restrict__ pSignY, const int signYPitch, const int8_t *__restrict__ pSignUV, const int signUVPitch,
    const int width, const int height, int *__restrict__ pStats) {
    const int ix = blockIdx.x * blockDim.x + threadIdx.x;
    const int iy = blockIdx.y * blockDim.y + threadIdx.y;
    if (ix >= width - 1 || iy >= height - 1) return;

    const int s_here  = (int)pSignY[iy * signYPitch + ix];
    const int s_right = (int)pSignY[iy * signYPitch + (ix + 1)];
    const int s_down  = (int)pSignY[(iy + 1) * signYPitch + ix];
    if ((s_here == s_right) && (s_here == s_down)) return;

    int best_dx = 0;
    int best_dy = 0;
    int best_dist_sq = (CHROMASHIFT_AUTO_SEARCH_R + 1) * (CHROMASHIFT_AUTO_SEARCH_R + 1) * 2;
    bool found = false;
    for (int dy = -CHROMASHIFT_AUTO_SEARCH_R; dy <= CHROMASHIFT_AUTO_SEARCH_R; dy++) {
        const int ny = iy + dy;
        if (ny < 0 || ny >= height - 1) continue;
        for (int dx = -CHROMASHIFT_AUTO_SEARCH_R; dx <= CHROMASHIFT_AUTO_SEARCH_R; dx++) {
            const int nx = ix + dx;
            if (nx < 0 || nx >= width - 1) continue;
            const int u_here  = (int)pSignUV[ny * signUVPitch + nx];
            const int u_right = (int)pSignUV[ny * signUVPitch + (nx + 1)];
            const int u_down  = (int)pSignUV[(ny + 1) * signUVPitch + nx];
            if ((u_here == u_right) && (u_here == u_down)) continue;
            const int dist_sq = dx * dx + dy * dy;
            if (dist_sq < best_dist_sq) {
                best_dist_sq = dist_sq;
                best_dx = dx;
                best_dy = dy;
                found = true;
            }
        }
    }
    if (found) {
        atomicAdd(&pStats[0], best_dx);
        atomicAdd(&pStats[1], best_dy);
        atomicAdd(&pStats[2], 1);
    }
}

template<typename Type, int bit_depth>
static RGY_ERR chromashift_shift_plane(RGYFrameInfo *pOutputFrame, const RGYFrameInfo *pInputFrame,
    const float shift_x_chroma, const float shift_y_chroma, cudaStream_t stream) {
    cudaTextureObject_t texSrc = 0;
    auto cudaerr = textureCreateChromaShift<Type>(texSrc, cudaFilterModeLinear, cudaReadModeNormalizedFloat,
        pInputFrame->ptr[0], pInputFrame->pitch[0], pInputFrame->width, pInputFrame->height);
    if (cudaerr != cudaSuccess) return err_to_rgy(cudaerr);
    dim3 blockSize(CHROMASHIFT_BLOCK_X, CHROMASHIFT_BLOCK_Y);
    dim3 gridSize(divCeil(pOutputFrame->width, blockSize.x), divCeil(pOutputFrame->height, blockSize.y));
    kernel_chromashift_shift<Type, bit_depth><<<gridSize, blockSize, 0, stream>>>((uint8_t *)pOutputFrame->ptr[0],
        pOutputFrame->pitch[0], pOutputFrame->width, pOutputFrame->height, texSrc, shift_x_chroma, shift_y_chroma);
    cudaerr = cudaGetLastError();
    if (cudaerr != cudaSuccess) {
        cudaDestroyTextureObject(texSrc);
        return err_to_rgy(cudaerr);
    }
    cudaerr = cudaDestroyTextureObject(texSrc);
    return (cudaerr == cudaSuccess) ? RGY_ERR_NONE : err_to_rgy(cudaerr);
}

template<typename Type, int bit_depth>
static RGY_ERR chromashift_laplacian_to_luma(RGYFrameInfo *pOutputFrame, const RGYFrameInfo *pInputFrame,
    const int subX, const int subY, cudaStream_t stream) {
    dim3 blockSize(CHROMASHIFT_BLOCK_X, CHROMASHIFT_BLOCK_Y);
    dim3 gridSize(divCeil(pOutputFrame->width, blockSize.x), divCeil(pOutputFrame->height, blockSize.y));
    kernel_chromashift_laplacian<Type, bit_depth><<<gridSize, blockSize, 0, stream>>>(
        (const uint8_t *)pInputFrame->ptr[0], pInputFrame->pitch[0], pInputFrame->width, pInputFrame->height,
        subX, subY, (uint8_t *)pOutputFrame->ptr[0], pOutputFrame->pitch[0], pOutputFrame->width, pOutputFrame->height);
    return err_to_rgy(cudaGetLastError());
}

template<typename Type, int bit_depth>
static RGY_ERR chromashift_fill_neutral(RGYFrameInfo *pOutputFrame, cudaStream_t stream) {
    dim3 blockSize(CHROMASHIFT_BLOCK_X, CHROMASHIFT_BLOCK_Y);
    dim3 gridSize(divCeil(pOutputFrame->width, blockSize.x), divCeil(pOutputFrame->height, blockSize.y));
    kernel_chromashift_fill_neutral<Type, bit_depth><<<gridSize, blockSize, 0, stream>>>(
        (uint8_t *)pOutputFrame->ptr[0], pOutputFrame->pitch[0], pOutputFrame->width, pOutputFrame->height);
    return err_to_rgy(cudaGetLastError());
}

template<typename Type>
static RGY_ERR chromashift_auto_frame(const RGYFrameInfo *pInputFrame, int8_t *signY, int8_t *signUV, int *stats,
    int *statsHost, const int lumaW, const int lumaH, const int subX, const int subY, cudaStream_t stream) {
    dim3 blockSize(CHROMASHIFT_BLOCK_X, CHROMASHIFT_BLOCK_Y);
    dim3 gridSize(divCeil(lumaW, blockSize.x), divCeil(lumaH, blockSize.y));
    auto cudaerr = cudaMemsetAsync(stats, 0, 3 * sizeof(int), stream);
    if (cudaerr != cudaSuccess) return err_to_rgy(cudaerr);
    const auto planeY = getPlane(pInputFrame, RGY_PLANE_Y);
    const auto planeU = getPlane(pInputFrame, RGY_PLANE_U);
    const auto planeV = getPlane(pInputFrame, RGY_PLANE_V);
    kernel_chromashift_lapsign_y<Type><<<gridSize, blockSize, 0, stream>>>(
        (const uint8_t *)planeY.ptr[0], planeY.pitch[0], signY, lumaW, lumaW, lumaH);
    if ((cudaerr = cudaGetLastError()) != cudaSuccess) return err_to_rgy(cudaerr);
    kernel_chromashift_lapsign_uv<Type><<<gridSize, blockSize, 0, stream>>>(
        (const uint8_t *)planeU.ptr[0], planeU.pitch[0], (const uint8_t *)planeV.ptr[0], planeV.pitch[0],
        planeU.width, planeU.height, subX, subY, signUV, lumaW, lumaW, lumaH);
    if ((cudaerr = cudaGetLastError()) != cudaSuccess) return err_to_rgy(cudaerr);
    kernel_chromashift_correlate<<<gridSize, blockSize, 0, stream>>>(signY, lumaW, signUV, lumaW, lumaW, lumaH, stats);
    if ((cudaerr = cudaGetLastError()) != cudaSuccess) return err_to_rgy(cudaerr);
    cudaerr = cudaMemcpyAsync(statsHost, stats, 3 * sizeof(int), cudaMemcpyDeviceToHost, stream);
    if (cudaerr != cudaSuccess) return err_to_rgy(cudaerr);
    return err_to_rgy(cudaStreamSynchronize(stream));
}

template<typename Type, int bit_depth>
static RGY_ERR chromashift_frame(RGYFrameInfo *pOutputFrame, const RGYFrameInfo *pInputFrame,
    const float effective_x, const float effective_y, const int show, const int subX, const int subY, cudaStream_t stream) {
    const int planes = RGY_CSP_PLANES[pInputFrame->csp];
    if (planes < 3) return RGY_ERR_UNSUPPORTED;
    if (show != 1) {
        const auto planeSrcY = getPlane(pInputFrame, RGY_PLANE_Y);
        auto planeDstY = getPlane(pOutputFrame, RGY_PLANE_Y);
        auto sts = copyPlaneAsync(&planeDstY, &planeSrcY, stream);
        if (sts != RGY_ERR_NONE) return sts;
    }

    if (show == 1) {
        const auto planeSrcU = getPlane(pInputFrame, RGY_PLANE_U);
        auto planeDstY = getPlane(pOutputFrame, RGY_PLANE_Y);
        auto sts = chromashift_laplacian_to_luma<Type, bit_depth>(&planeDstY, &planeSrcU, subX, subY, stream);
        if (sts != RGY_ERR_NONE) return sts;
        for (int i = 1; i < planes; i++) {
            auto planeDst = getPlane(pOutputFrame, (RGY_PLANE)i);
            sts = chromashift_fill_neutral<Type, bit_depth>(&planeDst, stream);
            if (sts != RGY_ERR_NONE) return sts;
        }
        return copyPlaneAlphaAsync(pOutputFrame, pInputFrame, stream);
    }

    if (effective_x == 0.0f && effective_y == 0.0f) {
        for (int i = 1; i < planes; i++) {
            const auto planeSrc = getPlane(pInputFrame, (RGY_PLANE)i);
            auto planeDst = getPlane(pOutputFrame, (RGY_PLANE)i);
            auto sts = copyPlaneAsync(&planeDst, &planeSrc, stream);
            if (sts != RGY_ERR_NONE) return sts;
        }
        return copyPlaneAlphaAsync(pOutputFrame, pInputFrame, stream);
    }

    const float shift_x_chroma = effective_x / (float)subX;
    const float shift_y_chroma = effective_y / (float)subY;
    for (int i = 1; i < planes; i++) {
        const auto planeSrc = getPlane(pInputFrame, (RGY_PLANE)i);
        auto planeDst = getPlane(pOutputFrame, (RGY_PLANE)i);
        auto sts = chromashift_shift_plane<Type, bit_depth>(&planeDst, &planeSrc, shift_x_chroma, shift_y_chroma, stream);
        if (sts != RGY_ERR_NONE) return sts;
    }
    return copyPlaneAlphaAsync(pOutputFrame, pInputFrame, stream);
}

NVEncFilterChromaShift::NVEncFilterChromaShift() :
    m_signY(),
    m_signUV(),
    m_statsBuf(),
    m_statsHost(),
    m_acceptedDx(),
    m_acceptedDy(),
    m_seenAnalysisFrames(0),
    m_skippedAutoFrames(0),
    m_warmupSkippedFrames(0),
    m_analysisComplete(false),
    m_resolvedShiftX(0.0f),
    m_resolvedShiftY(0.0f) {
    m_name = _T("chromashift");
}

NVEncFilterChromaShift::~NVEncFilterChromaShift() {
    close();
}

RGY_ERR NVEncFilterChromaShift::checkParam(const std::shared_ptr<NVEncFilterParamChromaShift> prm) {
    if (prm->frameOut.height <= 0 || prm->frameOut.width <= 0) {
        AddMessage(RGY_LOG_ERROR, _T("Invalid parameter.\n"));
        return RGY_ERR_INVALID_PARAM;
    }
    if (prm->chromashift.x < -4.0f || 4.0f < prm->chromashift.x) {
        prm->chromashift.x = clamp(prm->chromashift.x, -4.0f, 4.0f);
        AddMessage(RGY_LOG_WARN, _T("x should be in range of %.1f - %.1f.\n"), -4.0f, 4.0f);
    }
    if (prm->chromashift.y < -4.0f || 4.0f < prm->chromashift.y) {
        prm->chromashift.y = clamp(prm->chromashift.y, -4.0f, 4.0f);
        AddMessage(RGY_LOG_WARN, _T("y should be in range of %.1f - %.1f.\n"), -4.0f, 4.0f);
    }
    if (prm->chromashift.show < 0 || prm->chromashift.show > 1) {
        AddMessage(RGY_LOG_ERROR, _T("Invalid show=%d: must be 0 (normal) or 1 (laplacian).\n"), prm->chromashift.show);
        return RGY_ERR_INVALID_PARAM;
    }
    if (prm->chromashift.auto_frames < 1 || 100 < prm->chromashift.auto_frames) {
        prm->chromashift.auto_frames = clamp(prm->chromashift.auto_frames, 1, 100);
        AddMessage(RGY_LOG_WARN, _T("auto_frames should be in range of %d - %d.\n"), 1, 100);
    }
    if (prm->chromashift.auto_min_pairs < 10 || 10000 < prm->chromashift.auto_min_pairs) {
        prm->chromashift.auto_min_pairs = clamp(prm->chromashift.auto_min_pairs, 10, 10000);
        AddMessage(RGY_LOG_WARN, _T("auto_min_pairs should be in range of %d - %d.\n"), 10, 10000);
    }
    if (prm->chromashift.auto_detect && (prm->chromashift.x != 0.0f || prm->chromashift.y != 0.0f)) {
        AddMessage(RGY_LOG_WARN,
            _T("chromashift: auto=true takes precedence over x=%.2f, y=%.2f. ")
            _T("Those values are used only as a fallback if auto-analysis rejects too many frames.\n"),
            prm->chromashift.x, prm->chromashift.y);
    }
    return RGY_ERR_NONE;
}

RGY_ERR NVEncFilterChromaShift::init(shared_ptr<NVEncFilterParam> pParam, shared_ptr<RGYLog> pPrintMes) {
    m_pLog = pPrintMes;
    auto prm = std::dynamic_pointer_cast<NVEncFilterParamChromaShift>(pParam);
    if (!prm) {
        AddMessage(RGY_LOG_ERROR, _T("Invalid parameter type.\n"));
        return RGY_ERR_INVALID_PARAM;
    }
    auto sts = checkParam(prm);
    if (sts != RGY_ERR_NONE) return sts;

    prm->frameOut.picstruct = prm->frameIn.picstruct;
    sts = AllocFrameBuf(prm->frameOut, 1);
    if (sts != RGY_ERR_NONE) {
        AddMessage(RGY_LOG_ERROR, _T("failed to allocate output buffer: %s.\n"), get_err_mes(sts));
        return sts;
    }
    for (int i = 0; i < RGY_CSP_PLANES[pParam->frameOut.csp]; i++) {
        prm->frameOut.pitch[i] = m_frameBuf[0]->frame.pitch[i];
    }

    if (prm->chromashift.auto_detect) {
        const size_t lumaPx = (size_t)prm->frameOut.width * (size_t)prm->frameOut.height;
        m_signY = std::make_unique<CUMemBuf>(lumaPx);
        m_signUV = std::make_unique<CUMemBuf>(lumaPx);
        m_statsBuf = std::make_unique<CUMemBuf>(3 * sizeof(int));
        if (   RGY_ERR_NONE != (sts = m_signY->alloc())
            || RGY_ERR_NONE != (sts = m_signUV->alloc())
            || RGY_ERR_NONE != (sts = m_statsBuf->alloc())) {
            AddMessage(RGY_LOG_ERROR, _T("chromashift: failed to allocate auto-detection buffers: %s.\n"), get_err_mes(sts));
            return sts;
        }
        m_statsHost.assign(3, 0);
    } else {
        m_signY.reset();
        m_signUV.reset();
        m_statsBuf.reset();
        m_statsHost.clear();
    }

    setFilterInfo(pParam->print());
    m_param = prm;
    return RGY_ERR_NONE;
}

tstring NVEncFilterParamChromaShift::print() const {
    return chromashift.print();
}

RGY_ERR NVEncFilterChromaShift::run_filter(const RGYFrameInfo *pInputFrame, RGYFrameInfo **ppOutputFrames, int *pOutputFrameNum, cudaStream_t stream) {
    RGY_ERR sts = RGY_ERR_NONE;
    if (pInputFrame->ptr[0] == nullptr) return sts;
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
    if (m_param->frameOut.csp != m_param->frameIn.csp) {
        AddMessage(RGY_LOG_ERROR, _T("csp does not match.\n"));
        return RGY_ERR_INVALID_PARAM;
    }
    auto prm = std::dynamic_pointer_cast<NVEncFilterParamChromaShift>(m_param);
    if (!prm) {
        AddMessage(RGY_LOG_ERROR, _T("Invalid parameter type.\n"));
        return RGY_ERR_INVALID_PARAM;
    }
    if (RGY_CSP_CHROMA_FORMAT[pInputFrame->csp] == RGY_CHROMAFMT_RGB || RGY_CSP_PLANES[pInputFrame->csp] < 3) {
        AddMessage(RGY_LOG_ERROR, _T("unsupported csp %s.\n"), RGY_CSP_NAMES[pInputFrame->csp]);
        return RGY_ERR_UNSUPPORTED;
    }

    const auto pY = getPlane(pInputFrame, RGY_PLANE_Y);
    const auto pU = getPlane(pInputFrame, RGY_PLANE_U);
    const int subX = (pU.width  > 0) ? std::max(1, pY.width  / pU.width)  : 1;
    const int subY = (pU.height > 0) ? std::max(1, pY.height / pU.height) : 1;

    static const std::map<RGY_DATA_TYPE, decltype(chromashift_auto_frame<uint8_t>)*> auto_list = {
        { RGY_DATA_TYPE_U8,  chromashift_auto_frame<uint8_t> },
        { RGY_DATA_TYPE_U16, chromashift_auto_frame<uint16_t> }
    };
    if (prm->chromashift.auto_detect && !m_analysisComplete) {
        if (auto_list.count(RGY_CSP_DATA_TYPE[pInputFrame->csp]) == 0) {
            AddMessage(RGY_LOG_ERROR, _T("unsupported csp %s.\n"), RGY_CSP_NAMES[pInputFrame->csp]);
            return RGY_ERR_UNSUPPORTED;
        }
        sts = auto_list.at(RGY_CSP_DATA_TYPE[pInputFrame->csp])(pInputFrame,
            (int8_t *)m_signY->ptr, (int8_t *)m_signUV->ptr, (int *)m_statsBuf->ptr, m_statsHost.data(),
            pY.width, pY.height, subX, subY, stream);
        if (sts != RGY_ERR_NONE) {
            AddMessage(RGY_LOG_ERROR, _T("error at chromashift auto-analysis(%s): %s.\n"), RGY_CSP_NAMES[pInputFrame->csp], get_err_mes(sts));
            return sts;
        }

        const int sum_dx = m_statsHost[0];
        const int sum_dy = m_statsHost[1];
        const int count  = m_statsHost[2];
        const bool warmupSkip = (count == 0) && m_acceptedDx.empty();
        if (warmupSkip) {
            m_warmupSkippedFrames++;
            AddMessage(RGY_LOG_DEBUG, _T("chromashift: analysis frame %d bypassed (0 pairs, no accepted yet)\n"),
                m_warmupSkippedFrames + m_seenAnalysisFrames - 1);
        } else {
            m_seenAnalysisFrames++;
            if (count >= prm->chromashift.auto_min_pairs) {
                const double dx = (double)sum_dx / (double)count;
                const double dy = (double)sum_dy / (double)count;
                m_acceptedDx.push_back(dx);
                m_acceptedDy.push_back(dy);
                AddMessage(RGY_LOG_DEBUG, _T("chromashift: analysis frame %d -> dx=%+.3f dy=%+.3f (%d pairs)\n"),
                    m_seenAnalysisFrames - 1, dx, dy, count);
            } else {
                m_skippedAutoFrames++;
                AddMessage(RGY_LOG_DEBUG, _T("chromashift: analysis frame %d skipped (only %d pairs, min=%d)\n"),
                    m_seenAnalysisFrames - 1, count, prm->chromashift.auto_min_pairs);
            }
        }

        static constexpr int ABS_SAFETY_CAP = 1000;
        const int hardCap = prm->chromashift.auto_frames * 3 + 10;
        const bool haveTarget = (int)m_acceptedDx.size() >= prm->chromashift.auto_frames;
        const bool hitTimeout = (m_seenAnalysisFrames >= hardCap)
                             || ((m_seenAnalysisFrames + m_warmupSkippedFrames) >= ABS_SAFETY_CAP);
        if (haveTarget || hitTimeout) {
            const int acceptedCount = (int)m_acceptedDx.size();
            const int minTrusted = std::max(1, prm->chromashift.auto_frames / 2);
            if (acceptedCount >= minTrusted) {
                double sumDx = 0.0;
                double sumDy = 0.0;
                for (double d : m_acceptedDx) sumDx += d;
                for (double d : m_acceptedDy) sumDy += d;
                m_resolvedShiftX = (float)(sumDx / (double)acceptedCount);
                m_resolvedShiftY = (float)(sumDy / (double)acceptedCount);
                AddMessage(RGY_LOG_INFO, _T("chromashift: auto-detected x=%+.2f, y=%+.2f (from %d frames, skipped %d, warmup %d)\n"),
                    m_resolvedShiftX, m_resolvedShiftY, acceptedCount, m_skippedAutoFrames, m_warmupSkippedFrames);
            } else {
                m_resolvedShiftX = prm->chromashift.x;
                m_resolvedShiftY = prm->chromashift.y;
                AddMessage(RGY_LOG_WARN,
                    _T("chromashift: auto-analysis insufficient (only %d of %d target frames accepted ")
                    _T("after %d seen, %d bypassed in warmup). Falling back to x=%+.2f, y=%+.2f%s\n"),
                    acceptedCount, prm->chromashift.auto_frames, m_seenAnalysisFrames, m_warmupSkippedFrames,
                    m_resolvedShiftX, m_resolvedShiftY,
                    (m_resolvedShiftX == 0.0f && m_resolvedShiftY == 0.0f) ? _T(" (no manual shift set)") : _T(""));
            }
            m_analysisComplete = true;
            m_signY.reset();
            m_signUV.reset();
            m_statsBuf.reset();
            m_statsHost.clear();
            m_acceptedDx.clear();
            m_acceptedDy.clear();
        }
    }

    float effective_x = prm->chromashift.x;
    float effective_y = prm->chromashift.y;
    if (prm->chromashift.auto_detect) {
        if (m_analysisComplete) {
            effective_x = m_resolvedShiftX;
            effective_y = m_resolvedShiftY;
        } else {
            effective_x = 0.0f;
            effective_y = 0.0f;
        }
    }

    static const std::map<RGY_CSP, decltype(chromashift_frame<uint8_t, 8>)*> chromashift_list = {
        { RGY_CSP_YV12,      chromashift_frame<uint8_t,   8> },
        { RGY_CSP_YV12_16,   chromashift_frame<uint16_t, 16> },
        { RGY_CSP_YUV444,    chromashift_frame<uint8_t,   8> },
        { RGY_CSP_YUV444_16, chromashift_frame<uint16_t, 16> }
    };
    if (chromashift_list.count(pInputFrame->csp) == 0) {
        AddMessage(RGY_LOG_ERROR, _T("unsupported csp %s.\n"), RGY_CSP_NAMES[pInputFrame->csp]);
        return RGY_ERR_UNSUPPORTED;
    }
    sts = chromashift_list.at(pInputFrame->csp)(ppOutputFrames[0], pInputFrame,
        effective_x, effective_y, prm->chromashift.show, subX, subY, stream);
    if (sts != RGY_ERR_NONE) {
        AddMessage(RGY_LOG_ERROR, _T("error at chromashift(%s): %s.\n"), RGY_CSP_NAMES[pInputFrame->csp], get_err_mes(sts));
        return sts;
    }
    return sts;
}

void NVEncFilterChromaShift::close() {
    if (m_skippedAutoFrames > 0) {
        AddMessage(RGY_LOG_INFO, _T("chromashift: skipped %d frames during auto-analysis (insufficient zero-crossing pairs).\n"),
            m_skippedAutoFrames);
    }
    m_signY.reset();
    m_signUV.reset();
    m_statsBuf.reset();
    m_statsHost.clear();
    m_acceptedDx.clear();
    m_acceptedDy.clear();
    m_seenAnalysisFrames = 0;
    m_skippedAutoFrames = 0;
    m_warmupSkippedFrames = 0;
    m_analysisComplete = false;
    m_resolvedShiftX = 0.0f;
    m_resolvedShiftY = 0.0f;
    m_frameBuf.clear();
}
