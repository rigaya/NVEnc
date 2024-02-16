// -----------------------------------------------------------------------------------------
// NVEnc by rigaya
// -----------------------------------------------------------------------------------------
//
// The MIT License
//
// Copyright (c) 2019 rigaya
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

#pragma warning (push)
#pragma warning (disable: 4819)
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <device_launch_parameters.h>
#include <vector_types.h>
#include <stdint.h>
#pragma warning (pop)
#include "rgy_cuda_util.h"
#include "rgy_cuda_util_kernel.h"
#include "NVEncFilterSsim.h"

#if ENABLE_SSIM

static const int SSIM_BLOCK_X = 32;
static const int SSIM_BLOCK_Y = 8;
static const int SSIM_LOOP_Y = 1;

template<int bit_depth>
__device__ __inline__
float ssim_end1x(int64_t s1, int64_t s2, int64_t ss, int64_t s12) {
    const int64_t max = ((1 << bit_depth) - 1);
    const int64_t ssim_c1 = (int64_t)(0.01 * 0.01 * max * max * 64.0 + 0.5);
    const int64_t ssim_c2 = (int64_t)(0.03 * 0.03 * max * max * 64.0 * 63.0 + 0.5);

    const int64_t vars = ss * 64 - s1 * s1 - s2 * s2;
    const int64_t covar = s12 * 64 - s1 * s2;

    float f = ((float)(2 * s1 * s2 + ssim_c1) * (float)(2 * covar + ssim_c2))
        / ((float)(s1 * s1 + s2 * s2 + ssim_c1) * (float)(vars + ssim_c2));
    return f;
}

__device__ __inline__
void func_ssim_pix(longlong4 &ss, int a, int b) {
    ss.x += a;
    ss.y += b;
    ss.z += a*a;
    ss.z += b*b;
    ss.w += a*b;
}

template<typename Type4>
__device__ __inline__
longlong4 func_ssim_block(
    const uint8_t *p0, const int p0_pitch,
    const uint8_t *p1, const int p1_pitch) {
    longlong4 ss = make_longlong4(0,0,0,0);
    #pragma unroll
    for (int y = 0; y < 4; y++, p0 += p0_pitch, p1 += p1_pitch) {
        Type4 pix0 = *(Type4 *)p0;
        Type4 pix1 = *(Type4 *)p1;
        func_ssim_pix(ss, pix0.x, pix1.x);
        func_ssim_pix(ss, pix0.y, pix1.y);
        func_ssim_pix(ss, pix0.z, pix1.z);
        func_ssim_pix(ss, pix0.w, pix1.w);
    }
    return ss;
}

template<typename Type4, int bit_depth>
__global__ void kernel_ssim(
    const uint8_t *p0, const int p0_pitch,
    const uint8_t *p1, const int p1_pitch,
    const int width, const int height,
    float *__restrict__ pDst) {
    const int lx = threadIdx.x; //スレッド数=SSIM_BLOCK_X
    int ly = threadIdx.y;       //スレッド数=SSIM_BLOCK_Y
    const int blockoffset_x = blockIdx.x * SSIM_BLOCK_X;
    const int blockoffset_y = blockIdx.y * SSIM_BLOCK_Y * SSIM_LOOP_Y;
    const int imgx = (blockoffset_x + lx) * 4;
    int imgy = (blockoffset_y + ly) * 4;

    __shared__ longlong4 stmp[SSIM_BLOCK_Y * SSIM_LOOP_Y + 1][SSIM_BLOCK_X + 1];
#define STMP(x, y) ((stmp)[(y)][x])
    float ssim = 0.0f;
#if 1
    if (ly == 0) {
        if (imgx < width) {
            STMP(lx, ly) = func_ssim_block<Type4>(
                p0 + imgy * p0_pitch + imgx * sizeof(Type4::x), p0_pitch,
                p1 + imgy * p1_pitch + imgx * sizeof(Type4::x), p1_pitch);
        } else {
            STMP(lx, ly) = make_longlong4(0, 0, 0, 0);
        }
        if (lx == 0) {
            const int sx = SSIM_BLOCK_X;
            const int sy = 0;
            const int gx = (blockoffset_x + sx) * 4;
            const int gy = (blockoffset_y + sy) * 4;
            if (gx < width && gy < height) {
                STMP(sx, sy) = func_ssim_block<Type4>(
                    p0 + gy * p0_pitch + gx * sizeof(Type4::x), p0_pitch,
                    p1 + gy * p1_pitch + gx * sizeof(Type4::x), p1_pitch);
            } else {
                STMP(sx, sy) = make_longlong4(0, 0, 0, 0);
            }
        }
    }
    __syncthreads();
    imgy += 4;


    for (int y_loop = 0; y_loop < SSIM_LOOP_Y; y_loop++,
        imgy += SSIM_BLOCK_Y * 4, ly += SSIM_BLOCK_Y) {
        if (imgx < width && imgy < height) {
            STMP(lx, ly + 1) = func_ssim_block<Type4>(
                p0 + imgy * p0_pitch + imgx * sizeof(Type4::x), p0_pitch,
                p1 + imgy * p1_pitch + imgx * sizeof(Type4::x), p1_pitch);
        } else {
            STMP(lx, ly + 1) = make_longlong4(0,0,0,0);
        }
        if (ly == 0 && lx < SSIM_BLOCK_Y) {
            const int sx = SSIM_BLOCK_X;
            const int sy = (y_loop * SSIM_BLOCK_Y) + lx + 1;
            const int gx = (blockoffset_x + sx) * 4;
            const int gy = (blockoffset_y + sy) * 4;
            if (gx < width && gy < height) {
                STMP(sx, sy) = func_ssim_block<Type4>(
                    p0 + gy * p0_pitch + gx * sizeof(Type4::x), p0_pitch,
                    p1 + gy * p1_pitch + gx * sizeof(Type4::x), p1_pitch);
            } else {
                STMP(sx, sy) = make_longlong4(0,0,0,0);
            }
        }
        __syncthreads();
        if (imgx < (width - 4) && imgy < height) {
            longlong4 sx0y0 = STMP(lx + 0, ly + 0);
            longlong4 sx1y0 = STMP(lx + 1, ly + 0);
            longlong4 sx0y1 = STMP(lx + 0, ly + 1);
            longlong4 sx1y1 = STMP(lx + 1, ly + 1);
            ssim += ssim_end1x<bit_depth>(
                sx0y0.x + sx1y0.x + sx0y1.x + sx1y1.x,
                sx0y0.y + sx1y0.y + sx0y1.y + sx1y1.y,
                sx0y0.z + sx1y0.z + sx0y1.z + sx1y1.z,
                sx0y0.w + sx1y0.w + sx0y1.w + sx1y1.w);
        }
        __syncthreads();
    }
#else
    if (imgx < (width - 4) && imgy < (height - 4)) {
        longlong4 sx0y0 = func_ssim_block<Type4>(
            p0 + (imgy+0) * p0_pitch + (imgx+0) * sizeof(Type4::x), p0_pitch,
            p1 + (imgy+0) * p1_pitch + (imgx+0) * sizeof(Type4::x), p1_pitch);
        longlong4 sx1y0 = func_ssim_block<Type4>(
            p0 + (imgy+0) * p0_pitch + (imgx+4) * sizeof(Type4::x), p0_pitch,
            p1 + (imgy+0) * p1_pitch + (imgx+4) * sizeof(Type4::x), p1_pitch);
        longlong4 sx0y1 = func_ssim_block<Type4>(
            p0 + (imgy+4) * p0_pitch + (imgx+0) * sizeof(Type4::x), p0_pitch,
            p1 + (imgy+4) * p1_pitch + (imgx+0) * sizeof(Type4::x), p1_pitch);
        longlong4 sx1y1 = func_ssim_block<Type4>(
            p0 + (imgy+4) * p0_pitch + (imgx+4) * sizeof(Type4::x), p0_pitch,
            p1 + (imgy+4) * p1_pitch + (imgx+4) * sizeof(Type4::x), p1_pitch);
        ssim += ssim_end1x<bit_depth>(
            sx0y0.x + sx1y0.x + sx0y1.x + sx1y1.x,
            sx0y0.y + sx1y0.y + sx0y1.y + sx1y1.y,
            sx0y0.z + sx1y0.z + sx0y1.z + sx1y1.z,
            sx0y0.w + sx1y0.w + sx0y1.w + sx1y1.w);
    }
#endif

    ssim = block_sum<float, SSIM_BLOCK_X, SSIM_BLOCK_Y>(ssim, (float *)stmp);

    const int lid = threadIdx.y * SSIM_BLOCK_X + threadIdx.x;
    if (lid == 0) {
        const int gid = blockIdx.y * gridDim.x + blockIdx.x;
        pDst[gid] = ssim;
    }
}

template<typename Type4, int bit_depth>
RGY_ERR calc_ssim_plane(const RGYFrameInfo *p0, const RGYFrameInfo *p1, CUMemBufPair& tmp, cudaStream_t stream) {
    const int width = p0->width & (~3);
    const int height = p0->height & (~3);
    dim3 blockSize(SSIM_BLOCK_X, SSIM_BLOCK_Y);
    dim3 gridSize(divCeil(width, blockSize.x * 4), divCeil(height, blockSize.y * 4 * SSIM_LOOP_Y));

    const int grid_count = gridSize.x * gridSize.y;
    if (tmp.nSize < grid_count * sizeof(float)) {
        tmp.clear();
        auto sts = tmp.alloc(grid_count * sizeof(float));
        if (sts != RGY_ERR_NONE) {
            return sts;
        }
        float *ptrHost = (float *)tmp.ptrHost;
        for (int i = 0; i < grid_count; i++) {
            ptrHost[i] = 0.0f;
        }
        sts = tmp.copyHtoDAsync(stream);
        if (sts != RGY_ERR_NONE) {
            return sts;
        }
    }
    auto sts = err_to_rgy(cudaGetLastError());
    if (sts != RGY_ERR_NONE) {
        return sts;
    }
    kernel_ssim<Type4, bit_depth> <<< gridSize, blockSize, 0, stream >>> (
        (const uint8_t *)p0->ptr[0], p0->pitch[0],
        (const uint8_t *)p1->ptr[0], p1->pitch[0],
        width,
        height,
        (float *)tmp.ptrDevice);
    sts = err_to_rgy(cudaGetLastError());
    if (sts != RGY_ERR_NONE) {
        return sts;
    }
    sts = tmp.copyDtoHAsync(stream);
    if (sts != RGY_ERR_NONE) {
        return sts;
    }
    return err_to_rgy(cudaGetLastError());
}

template<typename Type4, int bit_depth>
RGY_ERR calc_ssim_frame(const RGYFrameInfo *p0, const RGYFrameInfo *p1, std::array<CUMemBufPair, 3> &tmp, std::array<std::unique_ptr<cudaStream_t, cudastream_deleter>, 3> &streamCalc) {
    for (int i = 0; i < RGY_CSP_PLANES[p0->csp]; i++) {
        const auto plane0 = getPlane(p0, (RGY_PLANE)i);
        const auto plane1 = getPlane(p1, (RGY_PLANE)i);
        auto sts = calc_ssim_plane<Type4, bit_depth>(&plane0, &plane1, tmp[i], *streamCalc[i].get());
        if (sts != RGY_ERR_NONE) {
            return sts;
        }
    }
    return RGY_ERR_NONE;
}

__device__ __inline__
int func_psnr_pix(int a, int b) {
    int i = a - b;
    return i * i;
}

template<typename Type4, int bit_depth>
__global__ void kernel_psnr(
    const uint8_t *p0, const int p0_pitch,
    const uint8_t *p1, const int p1_pitch,
    const int width, const int height,
    int *__restrict__ pDst) {
    const int lx = threadIdx.x; //スレッド数=SSIM_BLOCK_X
    const int ly = threadIdx.y; //スレッド数=SSIM_BLOCK_Y
    const int blockoffset_x = blockIdx.x * SSIM_BLOCK_X;
    const int blockoffset_y = blockIdx.y * SSIM_BLOCK_Y;
    const int imgx = (blockoffset_x + lx) * 4;
    const int imgy = (blockoffset_y + ly);

    int psnr = 0;
    if (imgx < width && imgy < height) {
        p0 += imgy * p0_pitch + imgx * sizeof(Type4::x);
        p1 += imgy * p1_pitch + imgx * sizeof(Type4::x);
        Type4 pix0 = *(Type4 *)p0;
        Type4 pix1 = *(Type4 *)p1;
        psnr += func_psnr_pix(pix0.x, pix1.x);
        if (imgx + 1 < width) psnr += func_psnr_pix(pix0.y, pix1.y);
        if (imgx + 2 < width) psnr += func_psnr_pix(pix0.z, pix1.z);
        if (imgx + 3 < width) psnr += func_psnr_pix(pix0.w, pix1.w);
    }

    __shared__ int tmp[SSIM_BLOCK_X * SSIM_BLOCK_Y / WARP_SIZE];
    psnr = block_sum<int, SSIM_BLOCK_X, SSIM_BLOCK_Y>(psnr, (int *)tmp);

    const int lid = threadIdx.y * SSIM_BLOCK_X + threadIdx.x;
    if (lid == 0) {
        const int gid = blockIdx.y * gridDim.x + blockIdx.x;
        pDst[gid] = psnr;
    }
}

template<typename Type4, int bit_depth>
RGY_ERR calc_psnr_plane(const RGYFrameInfo *p0, const RGYFrameInfo *p1, CUMemBufPair &tmp, cudaStream_t stream) {
    const int width = p0->width;
    const int height = p0->height;
    dim3 blockSize(SSIM_BLOCK_X, SSIM_BLOCK_Y);
    dim3 gridSize(divCeil(width, blockSize.x * 4), divCeil(height, blockSize.y));

    const int grid_count = gridSize.x * gridSize.y;
    if (tmp.nSize < grid_count * sizeof(int)) {
        tmp.clear();
        auto sts = tmp.alloc(grid_count * sizeof(int));
        if (sts != RGY_ERR_NONE) {
            return sts;
        }
        int *ptrHost = (int *)tmp.ptrHost;
        for (int i = 0; i < grid_count; i++) {
            ptrHost[i] = 0;
        }
        sts = tmp.copyHtoDAsync(stream);
        if (sts != RGY_ERR_NONE) {
            return sts;
        }
    }
    auto sts = err_to_rgy(cudaGetLastError());
    if (sts != RGY_ERR_NONE) {
        return sts;
    }
    kernel_psnr<Type4, bit_depth> << < gridSize, blockSize, 0, stream >> > (
        (const uint8_t *)p0->ptr[0], p0->pitch[0],
        (const uint8_t *)p1->ptr[0], p1->pitch[0],
        width,
        height,
        (int *)tmp.ptrDevice);
    sts = err_to_rgy(cudaGetLastError());
    if (sts != RGY_ERR_NONE) {
        return sts;
    }
    sts = tmp.copyDtoHAsync(stream);
    if (sts != RGY_ERR_NONE) {
        return sts;
    }
    return err_to_rgy(cudaGetLastError());
}

template<typename Type4, int bit_depth>
RGY_ERR calc_psnr_frame(const RGYFrameInfo *p0, const RGYFrameInfo *p1, std::array<CUMemBufPair, 3> &tmp, std::array<std::unique_ptr<cudaStream_t, cudastream_deleter>, 3> &streamCalc) {
    for (int i = 0; i < RGY_CSP_PLANES[p0->csp]; i++) {
        const auto plane0 = getPlane(p0, (RGY_PLANE)i);
        const auto plane1 = getPlane(p1, (RGY_PLANE)i);
        auto sts = calc_psnr_plane<Type4, bit_depth>(&plane0, &plane1, tmp[i], *streamCalc[i].get());
        if (sts != RGY_ERR_NONE) {
            return sts;
        }
    }
    return RGY_ERR_NONE;
}

RGY_ERR NVEncFilterSsim::calc_ssim_psnr(const RGYFrameInfo *p0, const RGYFrameInfo *p1) {
    auto prm = std::dynamic_pointer_cast<NVEncFilterParamSsim>(m_pParam);
    if (!prm) {
        AddMessage(RGY_LOG_ERROR, _T("Invalid parameter type.\n"));
        return RGY_ERR_INVALID_PARAM;
    }
    if (prm->ssim) {
        static const std::map<RGY_CSP, decltype(calc_ssim_frame<uchar4, 8>) *> ssim_list = {
            { RGY_CSP_YV12,      calc_ssim_frame<uchar4,   8> },
            { RGY_CSP_YV12_10,   calc_ssim_frame<ushort4, 10> },
            { RGY_CSP_YV12_12,   calc_ssim_frame<ushort4, 12> },
            { RGY_CSP_YV12_16,   calc_ssim_frame<ushort4, 16> },
            { RGY_CSP_YUV444,    calc_ssim_frame<uchar4,   8> },
            { RGY_CSP_YUV444_10, calc_ssim_frame<ushort4, 10> },
            { RGY_CSP_YUV444_12, calc_ssim_frame<ushort4, 12> },
            { RGY_CSP_YUV444_16, calc_ssim_frame<ushort4, 16> }
        };
        if (ssim_list.count(p0->csp) == 0) {
            AddMessage(RGY_LOG_ERROR, _T("unsupported csp %s.\n"), RGY_CSP_NAMES[p0->csp]);
            return RGY_ERR_UNSUPPORTED;
        }
        auto sts = ssim_list.at(p0->csp)(p0, p1, m_tmpSsim, m_streamCalcSsim);
        if (sts != RGY_ERR_NONE) {
            AddMessage(RGY_LOG_ERROR, _T("error at ssim(%s): %s.\n"),
                RGY_CSP_NAMES[p0->csp],
                get_err_mes(sts));
            return sts;
        }
    }

    if (prm->psnr) {
        static const std::map<RGY_CSP, decltype(calc_psnr_frame<uchar4, 8>) *> psnr_list = {
            { RGY_CSP_YV12,      calc_psnr_frame<uchar4,   8> },
            { RGY_CSP_YV12_10,   calc_psnr_frame<ushort4, 10> },
            { RGY_CSP_YV12_12,   calc_psnr_frame<ushort4, 12> },
            { RGY_CSP_YV12_16,   calc_psnr_frame<ushort4, 16> },
            { RGY_CSP_YUV444,    calc_psnr_frame<uchar4,   8> },
            { RGY_CSP_YUV444_10, calc_psnr_frame<ushort4, 10> },
            { RGY_CSP_YUV444_12, calc_psnr_frame<ushort4, 12> },
            { RGY_CSP_YUV444_16, calc_psnr_frame<ushort4, 16> }
        };
        if (psnr_list.count(p0->csp) == 0) {
            AddMessage(RGY_LOG_ERROR, _T("unsupported csp %s.\n"), RGY_CSP_NAMES[p0->csp]);
            return RGY_ERR_UNSUPPORTED;
        }
        auto sts = psnr_list.at(p0->csp)(p0, p1, m_tmpPsnr, m_streamCalcPsnr);
        if (sts != RGY_ERR_NONE) {
            AddMessage(RGY_LOG_ERROR, _T("error at ssim(%s): %s.\n"),
                RGY_CSP_NAMES[p0->csp],
                get_err_mes(sts));
            return sts;
        }
    }

    if (prm->ssim) {
        double ssimv = 0.0;
        for (int i = 0; i < RGY_CSP_PLANES[p0->csp]; i++) {
            cudaStreamSynchronize(*m_streamCalcSsim[i].get());

            const int count = (int)m_tmpSsim[i].nSize / sizeof(float);
            float *ptrHost = (float *)m_tmpSsim[i].ptrHost;
            std::sort(ptrHost, ptrHost + count);
            double ssimPlane = 0.0;
            for (int j = 0; j < count; j++) {
                ssimPlane += (double)ptrHost[j];
            }
            const auto plane0 = getPlane(p0, (RGY_PLANE)i);
            ssimPlane /= (double)(((plane0.width >> 2) - 1) *((plane0.height >> 2) - 1));
            m_ssimTotalPlane[i] += ssimPlane;
            ssimv += ssimPlane * m_planeCoef[i];
            AddMessage(RGY_LOG_TRACE, _T("ssimPlane = %.16e, m_ssimTotalPlane[i] = %.16e"), ssimPlane, m_ssimTotalPlane[i]);
        }
        m_ssimTotal += ssimv;
    }

    if (prm->psnr) {
        double psnrv = 0.0;
        for (int i = 0; i < RGY_CSP_PLANES[p0->csp]; i++) {
            cudaStreamSynchronize(*m_streamCalcPsnr[i].get());

            const int count = (int)m_tmpPsnr[i].nSize / sizeof(int);
            int *ptrHost = (int *)m_tmpPsnr[i].ptrHost;
            int64_t psnrPlane = 0;
            for (int j = 0; j < count; j++) {
                psnrPlane += ptrHost[j];
            }
            const auto plane0 = getPlane(p0, (RGY_PLANE)i);
            double psnrPlaneF = psnrPlane / (double)(plane0.width * plane0.height);
            m_psnrTotalPlane[i] += psnrPlaneF;
            psnrv += psnrPlaneF * m_planeCoef[i];
            AddMessage(RGY_LOG_TRACE, _T("psnrPlane = %.16e, m_psnrTotalPlane[i] = %.16e"), psnrPlane, m_psnrTotalPlane[i]);
        }
        m_psnrTotal += psnrv;
    }
    return RGY_ERR_NONE;
}

#endif //#if ENABLE_SSIM
