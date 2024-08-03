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

#include <array>
#include <map>
#include "convert_csp.h"
#include "NVEncFilterDeband.h"
#include "rgy_prm.h"
#pragma warning (push)
#pragma warning (disable: 4819)
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "curand_kernel.h"
#pragma warning (pop)

static const int GEN_RAND_BLOCK_LOOP_Y = 4;
static const int GEN_RAND_THREAD_X = 32;
static const int GEN_RAND_THREAD_Y = 8;

static const int DEBAND_BLOCK_THREAD_X = 32;
static const int DEBAND_BLOCK_THREAD_Y = 16;
static const int DEBAND_BLOCK_LOOP_X_OUTER = 2;
static const int DEBAND_BLOCK_LOOP_Y_OUTER = 2;
static const int DEBAND_BLOCK_LOOP_X_INNER = 1;
static const int DEBAND_BLOCK_LOOP_Y_INNER = 2;

#ifndef max
#define max( a, b ) ( ((a) > (b)) ? (a) : (b) )
#endif

#ifndef min
#define min( a, b ) ( ((a) < (b)) ? (a) : (b) )
#endif


__device__ int random_range(int random, int range) {
    return ((((range << 1) + 1) * random) >> 8) - range;
}
__device__ float random_range_float(int random, float range) {
    return (range * random) * (2.0f / 256.0f) - range;
}

template<bool interlaced, bool ref_y>
__device__ float get_ref(int random, int range) {
    if (interlaced && ref_y) {
        return (float)(random_range(random, range) & -2);
    } else {
        return random_range_float(random, range + 0.5f);
    }
}

__device__ float get_diff_abs(float a, float b) {
    return fabs(a - b);
}
__device__ float get_avg(float a, float b) {
    return (a + b) * 0.5f;
}
__device__ float get_avg(float a, float b, float c, float d) {
    return (a + b + c + d) * 0.25f;
}
__device__ int min4(int a, int b, int c, int d) {
    return min(min(a, b), min(c, d));
}
__device__ float get_max(float a, float b) {
    return fmaxf(a, b);
}
__device__ float get_max(float a, float b, float c, float d) {
    return fmaxf(fmaxf(a, b), fmaxf(c, d));
}

__global__ void kernel_rand_init(curandState *__restrict__ pState, int seed) {
    const int gtid_x = blockIdx.x * blockDim.x + threadIdx.x;
    const int gtid_y = blockIdx.y * blockDim.y + threadIdx.y;
    const int gtid = gtid_y * gridDim.x * blockDim.x + gtid_x;
    curandState state;
    curand_init(seed, gtid, 0, &state);
    pState[gtid] = state;
}

//block size 32x8
//
//乱数の各バイトの中身
//1pixelあたり32bit
//pRandY           [ refA0, refB0, ditherV0, 0, refA1, refB1, ditherY1, 0, ... ]
//pRandUV (yuv420) [ refA0, refB0, ditherU0, ditherV0, refA2, refB2, ditherU2, ditherV2, ... ]
//pRandUV (yuv444) [ refA0, refB0, ditherU0, ditherV0, refA1, refB1, ditherU1, ditherV1, ... ]
template<int block_loop_y, bool yuv420>
__global__ void kernel_gen_rand(int8_t *__restrict__ pRandY, int8_t *__restrict__ pRandUV,
    int pitchY, int pitchUV, int width, int height, curandState *pState) {
    const int gid_i_half = blockIdx.x * blockDim.x /* 32 */ + threadIdx.x;
    int gid_j_half = blockIdx.y * block_loop_y * blockDim.y /* 8 */ + threadIdx.y;
    if ((gid_i_half << 1) < width) {
        const int thread_x_num = gridDim.x * blockDim.x;
        const int gid = (blockIdx.y * blockDim.y + threadIdx.y) * thread_x_num + gid_i_half;
        curandState state = pState[gid];

        #pragma unroll
        for (int iyb_loop = 0; iyb_loop < block_loop_y; iyb_loop++, gid_j_half += blockDim.y) {
            if ((gid_j_half << 1) < height) {
                uint32_t rand0 = curand(&state);
                const uint32_t refAB0 = rand0 & 0xffff;

                uint32_t rand1 = curand(&state);
                const uint32_t refAB1 = rand1 & 0xffff;

                uint32_t rand2 = curand(&state);
                const uint32_t refAB2 = rand2 & 0xffff;

                uint32_t rand3 = curand(&state);
                const uint32_t refAB3 = rand3 & 0xffff;

                //const uint8_t dithY0 = (rand0 & 0x00ff0000) >> 16;
                //const uint8_t dithY1 = (rand1 & 0x00ff0000) >> 16;
                //const uint8_t dithY2 = (rand2 & 0x00ff0000) >> 16;
                //const uint8_t dithY3 = (rand3 & 0x00ff0000) >> 16;
                //const uint8_t dithU0 = (rand0 & 0xff000000) >> 24;
                //const uint8_t dithV0 = (rand1 & 0xff000000) >> 24;

                //y line0
                //char8 data_y0 = { refA0, refB0, dithY0, 0, refA1, refB1, dithY1, 0 };
                uint2 data_y0 = make_uint2(rand0, rand1);
                //y line1
                //charB data_y1 = { refA2, refB2, dithY2, 0, refA3, refB3, dithY3, 0 };
                uint2 data_y1 = make_uint2(rand2, rand3);

                *(uint2 *)(pRandY + (gid_j_half << 1) * pitchY + (gid_i_half << 1) * sizeof(uint32_t) + 0)      = data_y0;
                *(uint2 *)(pRandY + (gid_j_half << 1) * pitchY + (gid_i_half << 1) * sizeof(uint32_t) + pitchY) = data_y1;
                if (yuv420) {
                    // { refAB0, dithU0, dithV0 }
                    uint32_t data_c0 = refAB0 | ((rand0 & 0xff000000) >> 8)| (rand1 & 0xff000000);
                    *(uint32_t *)(pRandUV + gid_j_half * pitchUV + gid_i_half * sizeof(uint32_t)) = data_c0;
                } else {
                    //const uint8_t dithU1 = (rand2 & 0xff000000) >> 24;
                    //const uint8_t dithV1 = (rand3 & 0xff000000) >> 24;
                    uint32_t rand4 = curand(&state);
                    //const uint8_t dithU2 = (rand4 & 0x000000ff);
                    //const uint8_t dithV2 = (rand4 & 0x0000ff00) >> 8;
                    //const uint8_t dithU3 = (rand4 & 0x00ff0000) >> 16;
                    //const uint8_t dithV3 = (rand4 & 0xff000000) >> 24;
                    //c line0
                    //{ refAB0, dithU0, dithV0, refAB1, dithU1, dithV1 }
                    uint2 data_c0 = make_uint2(refAB0 | ((rand0 & 0xff000000) >> 8) | (rand1 & 0xff000000),
                                               refAB1 | ((rand2 & 0xff000000) >> 8) | (rand3 & 0xff000000));
                    //c line1
                    //{ refAB2, rand4 & 0x0000ffff, refAB3, rand4 & 0xffff0000 };
                    uint2 data_c1 = make_uint2(refAB2 | ((rand4 & 0x0000ffff) << 16), refAB3 | (rand4 & 0xffff0000));
                    *(uint2 *)(pRandUV + (gid_j_half << 1) * pitchUV + (gid_i_half << 1) * sizeof(uint32_t) +       0) = data_c0;
                    *(uint2 *)(pRandUV + (gid_j_half << 1) * pitchUV + (gid_i_half << 1) * sizeof(uint32_t) + pitchUV) = data_c1;
                }
            }
        }
        pState[gid] = state;
    }
}

enum DebandPlane {
    MODE_Y,
    MODE_U,
    MODE_V
};

//threshold = (fp->track[1,2,3] << (!(sample_mode && blur_first) + 1)) * (1.0f / (1 << 10));
//range = (yuv420 && target_is_uv) ? fp->track[0] >> 1 : fp->track[0];
//dither_range = (float)dither * pow(2.0f, bit_depth-10) + 0.5
//field_mask = fp->check[2] ? -2 : -1;
template<typename Type, int bit_depth, int sample_mode, DebandPlane mode_yuv, bool blur_first, int block_loop_x_inner, int block_loop_y_inner, int block_loop_x_outer, int block_loop_y_outer>
__global__ void kernel_deband(uint8_t * __restrict__ pDst, const int dstPitch, const int dstWidth, const int dstHeight,
    uchar4 *__restrict__ pRand, int pitchRand,
    cudaTextureObject_t texSrc, const int range, const float dither_range, const float threshold, const int field_mask) {
    const int itx = blockIdx.x * blockDim.x * block_loop_x_inner * block_loop_x_outer + threadIdx.x;
    const int ity = blockIdx.y * blockDim.y * block_loop_y_inner * block_loop_y_outer + threadIdx.y;
    #pragma unroll
    for (int jby = 0; jby < block_loop_y_outer; jby++) {
        #pragma unroll
        for (int jbx = 0; jbx < block_loop_x_outer; jbx++) {
            #pragma unroll
            for (int iby = 0; iby < block_loop_y_inner; iby++) {
                const int iy = ity + (jby * block_loop_y_inner + iby) * blockDim.y;
                if (iy < dstHeight) {
                    #pragma unroll
                    for (int ibx = 0; ibx < block_loop_x_inner; ibx++) {
                        const int ix = itx + (jbx * block_loop_x_inner + ibx) * blockDim.x;
                        if (ix < dstWidth) {
                            const float x = (float)ix + 0.5f;
                            const float y = (float)iy + 0.5f;
                            const int gid = iy * (pitchRand >> 2 /* pitchRand / slzeof(uchar4)*/) + ix;

                            const int y_limit = min(iy, dstHeight - iy - 1);
                            const int range_limited = min4(range, y_limit, ix, dstWidth - ix - 1);
                            const uchar4 rand = pRand[gid];
                            const int refA = random_range(rand.x, range_limited);
                            const int refB = random_range(rand.y, range_limited);

                            const float clr_center = tex2D<float>(texSrc, x, y);
                            float clr_avg, clr_diff;
                            if (sample_mode == 0) {
                                const float clr_ref0 = tex2D<float>(texSrc, x + refB, y + (refA & field_mask));
                                clr_avg = clr_ref0;
                                clr_diff = get_diff_abs(clr_center, clr_ref0);
                            } else if (sample_mode == 1) {
                                const float clr_ref0 = tex2D<float>(texSrc, x + refB, y + (refA & field_mask));
                                const float clr_ref1 = tex2D<float>(texSrc, x - refB, y - (refA & field_mask));
                                clr_avg = get_avg(clr_ref0, clr_ref1);
                                clr_diff = (blur_first) ? get_diff_abs(clr_center, clr_avg)
                                                        : get_max(get_diff_abs(clr_center, clr_ref0),
                                                                  get_diff_abs(clr_center, clr_ref1));
                            } else {
                                const float clr_ref00 = tex2D<float>(texSrc, x + refB, y + (refA & field_mask));
                                const float clr_ref01 = tex2D<float>(texSrc, x - refB, y - (refA & field_mask));
                                const float clr_ref10 = tex2D<float>(texSrc, x + refA, y + (refB & field_mask));
                                const float clr_ref11 = tex2D<float>(texSrc, x - refA, y - (refB & field_mask));
                                clr_avg = get_avg(clr_ref00, clr_ref01, clr_ref10, clr_ref11);
                                clr_diff = (blur_first) ? get_diff_abs(clr_center, clr_avg)
                                                        : get_max(get_diff_abs(clr_center, clr_ref00),
                                                                  get_diff_abs(clr_center, clr_ref01),
                                                                  get_diff_abs(clr_center, clr_ref10),
                                                                  get_diff_abs(clr_center, clr_ref11));
                            }
                            const float clr_out = (clr_diff < threshold) ? clr_avg : clr_center;
                            float pix_out = clr_out * (float)(1<<bit_depth);
                            if (sample_mode != 0) {
                                const uint8_t randu8 = ((mode_yuv == MODE_V) ? rand.w : rand.z);
                                pix_out += random_range_float((int)(randu8), dither_range);
                            }
                            Type *ptr = (Type *)(pDst + iy * dstPitch + ix * sizeof(Type));
                            ptr[0] = (Type)clamp(pix_out + 0.5f, 0.0f, (float)(1<<bit_depth)-1.0f);
                        }
                    }
                }
            }
        }
    }
}

template<typename Type, int bit_depth, int sample_mode, DebandPlane mode_yuv, bool blur_first, int block_loop_x_inner, int block_loop_y_inner, int block_loop_x_outer, int block_loop_y_outer>
RGY_ERR deband_plane(
    uint8_t *pDst, const int dstPitch, const int dstWidth, const int dstHeight,
    uint8_t *pSrc, const int srcPitch, const int srcWidth, const int srcHeight,
    uint8_t *pRand, const int randPitch,
    const bool isYUV420, const int range, const int dither, const int threshold, const bool interlaced,
    cudaStream_t stream) {
    const float dither_range = dither * (float)std::pow(2.0f, bit_depth-12) + 0.5f;
    const float threshold_float = (threshold << (!(sample_mode && blur_first) + 1)) * (1.0f / (1 << 12));
    const int range_plane = (isYUV420 && mode_yuv != MODE_Y) ? range >> 1 : range;
    const int field_mask = (interlaced) ? -2 : -1;

    cudaResourceDesc resDescSrc;
    memset(&resDescSrc, 0, sizeof(resDescSrc));
    resDescSrc.resType = cudaResourceTypePitch2D;
    resDescSrc.res.pitch2D.devPtr = pSrc;
    resDescSrc.res.pitch2D.pitchInBytes = srcPitch;
    resDescSrc.res.pitch2D.width = srcWidth;
    resDescSrc.res.pitch2D.height = srcHeight;
    resDescSrc.res.pitch2D.desc = cudaCreateChannelDesc<Type>();

    cudaTextureDesc texDescSrc;
    memset(&texDescSrc, 0, sizeof(texDescSrc));
    texDescSrc.addressMode[0]   = cudaAddressModeClamp;
    texDescSrc.addressMode[1]   = cudaAddressModeClamp;
    texDescSrc.filterMode       = cudaFilterModePoint;
    texDescSrc.readMode         = cudaReadModeNormalizedFloat;
    texDescSrc.normalizedCoords = 0;

    cudaTextureObject_t texSrc = 0;
    auto cudaerr = cudaCreateTextureObject(&texSrc, &resDescSrc, &texDescSrc, nullptr);
    if (cudaerr != cudaSuccess) {
        return err_to_rgy(cudaerr);
    }
    dim3 blockSize(DEBAND_BLOCK_THREAD_X, DEBAND_BLOCK_THREAD_Y);
    dim3 gridSize(
        divCeil(dstWidth, blockSize.x * block_loop_x_inner * block_loop_x_outer),
        divCeil(dstHeight, blockSize.y * block_loop_y_inner * block_loop_y_outer));
    kernel_deband<Type, bit_depth, sample_mode, mode_yuv, blur_first, block_loop_x_inner, block_loop_y_inner, block_loop_x_outer, block_loop_y_outer>
        <<<gridSize, blockSize, 0, stream>>>(
        pDst, dstPitch, dstWidth, dstHeight,
        (uchar4 *)pRand, randPitch,
        texSrc,
        range_plane, dither_range, threshold_float, field_mask);
    cudaerr = cudaGetLastError();
    if (cudaerr != cudaSuccess) {
        return err_to_rgy(cudaerr);
    }

    cudaerr = cudaDestroyTextureObject(texSrc);
    return err_to_rgy(cudaerr);
}

template<typename Type, int bit_depth, int sample_mode, bool blur_first>
static RGY_ERR deband_frame(RGYFrameInfo *pOutputFrame, const RGYFrameInfo *pInputFrame, RGYFrameInfo *pRandY, RGYFrameInfo *pRandUV,
    const int range, const int threY, const int threCb, const int threCr, const int ditherY, const int ditherC, bool randEachFrame,
    curandState *pState, cudaStream_t stream) {
    if (randEachFrame) {
        dim3 threads(GEN_RAND_THREAD_X, GEN_RAND_THREAD_Y, 1);
        dim3 grids(divCeil(pRandY->width >> 1, threads.x), divCeil(pRandY->height >> 1, threads.y * GEN_RAND_BLOCK_LOOP_Y), 1);
        kernel_gen_rand<GEN_RAND_BLOCK_LOOP_Y, false> << <grids, threads, 0, stream >> > (
            (int8_t *)pRandY->ptr[0], (int8_t *)pRandUV->ptr[0],
            pRandY->pitch[0], pRandUV->pitch[0],
            pRandY->width, pRandY->height,
            pState);
        auto err = cudaGetLastError();
        if (err != cudaSuccess) {
            return err_to_rgy(err);
        }
    }

    const auto planeInputY = getPlane(pInputFrame, RGY_PLANE_Y);
    const auto planeInputU = getPlane(pInputFrame, RGY_PLANE_U);
    const auto planeInputV = getPlane(pInputFrame, RGY_PLANE_V);
    auto planeOutputY = getPlane(pOutputFrame, RGY_PLANE_Y);
    auto planeOutputU = getPlane(pOutputFrame, RGY_PLANE_U);
    auto planeOutputV = getPlane(pOutputFrame, RGY_PLANE_V);

    auto sts = deband_plane<Type, bit_depth, sample_mode, MODE_Y, blur_first, DEBAND_BLOCK_LOOP_X_INNER, DEBAND_BLOCK_LOOP_Y_INNER, DEBAND_BLOCK_LOOP_X_OUTER, DEBAND_BLOCK_LOOP_Y_OUTER>(
        planeOutputY.ptr[0], planeOutputY.pitch[0], planeOutputY.width, planeOutputY.height,
        planeInputY.ptr[0], planeInputY.pitch[0], planeInputY.width, planeInputY.height,
        pRandY->ptr[0], pRandY->pitch[0],
        RGY_CSP_CHROMA_FORMAT[pInputFrame->csp] == RGY_CHROMAFMT_YUV420,
        range, ditherY, threY, interlaced(*pInputFrame),
        stream);
    if (sts != RGY_ERR_NONE) {
        return sts;
    }
    sts = deband_plane<Type, bit_depth, sample_mode, MODE_U, blur_first, DEBAND_BLOCK_LOOP_X_INNER, DEBAND_BLOCK_LOOP_Y_INNER, DEBAND_BLOCK_LOOP_X_OUTER, DEBAND_BLOCK_LOOP_Y_OUTER>(
        planeOutputU.ptr[0], planeOutputU.pitch[0], planeOutputU.width, planeOutputU.height,
        planeInputU.ptr[0], planeInputU.pitch[0], planeInputU.width, planeInputU.height,
        pRandUV->ptr[0], pRandUV->pitch[0],
        RGY_CSP_CHROMA_FORMAT[pInputFrame->csp] == RGY_CHROMAFMT_YUV420,
        range, ditherC, threCb, interlaced(*pInputFrame),
        stream);
    if (sts != RGY_ERR_NONE) {
        return sts;
    }
    sts = deband_plane<Type, bit_depth, sample_mode, MODE_V, blur_first, DEBAND_BLOCK_LOOP_X_INNER, DEBAND_BLOCK_LOOP_Y_INNER, DEBAND_BLOCK_LOOP_X_OUTER, DEBAND_BLOCK_LOOP_Y_OUTER>(
        planeOutputV.ptr[0], planeOutputV.pitch[0], planeOutputV.width, planeOutputV.height,
        planeInputV.ptr[0], planeInputV.pitch[0], planeInputV.width, planeInputV.height,
        pRandUV->ptr[0], pRandUV->pitch[0],
        RGY_CSP_CHROMA_FORMAT[pInputFrame->csp] == RGY_CHROMAFMT_YUV420,
        range, ditherC, threCr, interlaced(*pInputFrame),
        stream);
    if (sts != RGY_ERR_NONE) {
        return sts;
    }
    sts = copyPlaneAlphaAsync(pOutputFrame, pInputFrame, stream);
    if (sts != RGY_ERR_NONE) {
        return sts;
    }
    return RGY_ERR_NONE;
}


RGY_ERR NVEncFilterDeband::deband(RGYFrameInfo *pOutputFrame, const RGYFrameInfo *pInputFrame, cudaStream_t stream) {
    if (m_param->frameOut.csp != m_param->frameIn.csp) {
        AddMessage(RGY_LOG_ERROR, _T("csp does not match.\n"));
        return RGY_ERR_INVALID_CALL;
    }
    auto pParam = std::dynamic_pointer_cast<NVEncFilterParamDeband>(m_param);
    if (!pParam) {
        AddMessage(RGY_LOG_ERROR, _T("Invalid parameter type.\n"));
        return RGY_ERR_INVALID_PARAM;
    }
    struct deband_func {
        decltype(deband_frame<uint8_t, 8, 0, true>)* func[3][2]; /* sample_mode, blur_first */
        deband_func(
            decltype(deband_frame<uint8_t, 8, 0, true>)* sample0_blur_first0,
            decltype(deband_frame<uint8_t, 8, 0, true>)* sample0_blur_first1,
            decltype(deband_frame<uint8_t, 8, 0, true>)* sample1_blur_first0,
            decltype(deband_frame<uint8_t, 8, 0, true>)* sample1_blur_first1,
            decltype(deband_frame<uint8_t, 8, 0, true>)* sample2_blur_first0,
            decltype(deband_frame<uint8_t, 8, 0, true>)* sample2_blur_first1
            ) {
            func[0][0] = sample0_blur_first0;
            func[0][1] = sample0_blur_first1;
            func[1][0] = sample1_blur_first0;
            func[1][1] = sample1_blur_first1;
            func[2][0] = sample2_blur_first0;
            func[2][1] = sample2_blur_first1;
        };
    };

    static const std::map<RGY_DATA_TYPE, deband_func> deband_func_list = {
        { RGY_DATA_TYPE_U8, deband_func(
            deband_frame<uint8_t,   8, 0, false>, deband_frame<uint8_t,   8, 0, true>,
            deband_frame<uint8_t,   8, 1, false>, deband_frame<uint8_t,   8, 1, true>,
            deband_frame<uint8_t,   8, 2, false>, deband_frame<uint8_t,   8, 2, true>) },
        { RGY_DATA_TYPE_U16, deband_func(
            deband_frame<uint16_t, 16, 0, false>, deband_frame<uint16_t, 16, 0, true>,
            deband_frame<uint16_t, 16, 1, false>, deband_frame<uint16_t, 16, 1, true>,
            deband_frame<uint16_t, 16, 2, false>, deband_frame<uint16_t, 16, 2, true>) }
    };
    if (deband_func_list.count(RGY_CSP_DATA_TYPE[pParam->frameIn.csp]) == 0) {
        AddMessage(RGY_LOG_ERROR, _T("unsupported csp for deband: %s\n"), RGY_CSP_NAMES[pParam->frameIn.csp]);
        return RGY_ERR_UNSUPPORTED;
    }
    auto err = deband_func_list.at(RGY_CSP_DATA_TYPE[pParam->frameIn.csp]).func[pParam->deband.sample][pParam->deband.blurFirst ? 1 : 0](
        pOutputFrame, pInputFrame, &m_RandY.frame, &m_RandUV.frame,
        pParam->deband.range, pParam->deband.threY, pParam->deband.threCb, pParam->deband.threCr, pParam->deband.ditherY, pParam->deband.ditherC, pParam->deband.randEachFrame,
        (curandState *)m_RandState.ptr, stream);
    if (err != RGY_ERR_NONE) {
        return RGY_ERR_CUDA;
    }
    return RGY_ERR_NONE;
}

NVEncFilterDeband::NVEncFilterDeband() : m_RandY(), m_RandUV() {
    m_name = _T("deband");
}

NVEncFilterDeband::~NVEncFilterDeband() {
    close();
}

RGY_ERR NVEncFilterDeband::init(shared_ptr<NVEncFilterParam> pParam, shared_ptr<RGYLog> pPrintMes) {
    RGY_ERR sts = RGY_ERR_NONE;
    m_pLog = pPrintMes;
    auto pDebandParam = std::dynamic_pointer_cast<NVEncFilterParamDeband>(pParam);
    if (!pDebandParam) {
        AddMessage(RGY_LOG_ERROR, _T("Invalid parameter type.\n"));
        return RGY_ERR_INVALID_PARAM;
    }
    //パラメータチェック
    if (pDebandParam->frameOut.height <= 0 || pDebandParam->frameOut.width <= 0) {
        AddMessage(RGY_LOG_ERROR, _T("Invalid parameter.\n"));
        return RGY_ERR_INVALID_PARAM;
    }
    if (pDebandParam->deband.range < 0 || 127 < pDebandParam->deband.range) {
        AddMessage(RGY_LOG_WARN, _T("range must be in range of 0 - 127.\n"));
        pDebandParam->deband.range = clamp(pDebandParam->deband.range, 0, 127);
    }
    if (pDebandParam->deband.threY < 0 || 31 < pDebandParam->deband.threY) {
        AddMessage(RGY_LOG_WARN, _T("threY must be in range of 0 - 31.\n"));
        pDebandParam->deband.threY = clamp(pDebandParam->deband.threY, 0, 31);
    }
    if (pDebandParam->deband.threCb < 0 || 31 < pDebandParam->deband.threCb) {
        AddMessage(RGY_LOG_WARN, _T("threCb must be in range of 0 - 31.\n"));
        pDebandParam->deband.threCb = clamp(pDebandParam->deband.threCb, 0, 31);
    }
    if (pDebandParam->deband.threCr < 0 || 31 < pDebandParam->deband.threCr) {
        AddMessage(RGY_LOG_WARN, _T("threCr must be in range of 0 - 31.\n"));
        pDebandParam->deband.threCr = clamp(pDebandParam->deband.threCr, 0, 31);
    }
    if (pDebandParam->deband.ditherY < 0 || 31 < pDebandParam->deband.ditherY) {
        AddMessage(RGY_LOG_WARN, _T("ditherY must be in range of 0 - 31.\n"));
        pDebandParam->deband.ditherY = clamp(pDebandParam->deband.ditherY, 0, 31);
    }
    if (pDebandParam->deband.ditherC < 0 || 31 < pDebandParam->deband.ditherC) {
        AddMessage(RGY_LOG_WARN, _T("ditherC must be in range of 0 - 31.\n"));
        pDebandParam->deband.ditherC = clamp(pDebandParam->deband.ditherC, 0, 31);
    }
    if (pDebandParam->deband.sample < 0 || 2 < pDebandParam->deband.sample) {
        AddMessage(RGY_LOG_WARN, _T("mode must be in range of 0 - 2.\n"));
        pDebandParam->deband.sample = clamp(pDebandParam->deband.sample, 0, 2);
    }

    sts = AllocFrameBuf(pDebandParam->frameOut, 1);
    if (sts != RGY_ERR_NONE) {
        AddMessage(RGY_LOG_ERROR, _T("failed to allocate memory: %s.\n"), get_err_mes(sts));
        return sts;
    }
    for (int i = 0; i < RGY_CSP_PLANES[pParam->frameOut.csp]; i++) {
        pDebandParam->frameOut.pitch[i] = m_frameBuf[0]->frame.pitch[i];
    }

    bool resChanged = cmpFrameInfoCspResolution(&m_RandUV.frame, &pDebandParam->frameOut);
    if (resChanged) {
        m_RandY.frame.width = pDebandParam->frameOut.width;
        m_RandY.frame.height = pDebandParam->frameOut.height;
        for (int i = 0; i < RGY_CSP_PLANES[pParam->frameOut.csp]; i++) {
            m_RandY.frame.pitch[i] = pDebandParam->frameOut.pitch[i];
        }
        m_RandY.frame.picstruct = pDebandParam->frameOut.picstruct;
        m_RandY.frame.mem_type = pDebandParam->frameOut.mem_type;
        m_RandY.frame.csp = RGY_CSP_RGB32;
        m_RandY.clear();
        sts = m_RandY.alloc();
        if (sts != RGY_ERR_NONE) {
            AddMessage(RGY_LOG_ERROR, _T("failed to allocate memory: %s.\n"), get_err_mes(sts));
            return sts;
        }

        m_RandUV.frame.width = pDebandParam->frameOut.width;
        m_RandUV.frame.height = pDebandParam->frameOut.height;
        for (int i = 0; i < RGY_CSP_PLANES[pParam->frameOut.csp]; i++) {
            m_RandUV.frame.pitch[i] = pDebandParam->frameOut.pitch[i];
        }
        m_RandUV.frame.picstruct = pDebandParam->frameOut.picstruct;
        m_RandUV.frame.mem_type = pDebandParam->frameOut.mem_type;
        m_RandUV.frame.csp = pDebandParam->frameOut.csp;
        m_RandUV.clear();
        sts = m_RandUV.alloc();
        if (sts != RGY_ERR_NONE) {
            AddMessage(RGY_LOG_ERROR, _T("failed to allocate memory: %s.\n"), get_err_mes(sts));
            return sts;
        }
    }

    if (resChanged
        || !m_param
        || std::dynamic_pointer_cast<NVEncFilterParamDeband>(m_param)->deband.seed != pDebandParam->deband.seed) {
        dim3 threads(GEN_RAND_THREAD_X, GEN_RAND_THREAD_Y, 1);
        dim3 grids(divCeil(pDebandParam->frameOut.width >> 1, threads.x), divCeil(pDebandParam->frameOut.height >> 1, threads.y * GEN_RAND_BLOCK_LOOP_Y), 1);
        m_RandState.nSize = sizeof(curandState) * (threads.x * grids.x) * (threads.y * grids.y);
        sts = m_RandState.alloc();
        if (sts != RGY_ERR_NONE) {
            AddMessage(RGY_LOG_ERROR, _T("failed to allocate memory: %s.\n"), get_err_mes(sts));
            return sts;
        }

        kernel_rand_init<<<grids, threads>>>((curandState *)m_RandState.ptr, pDebandParam->deband.seed);
        sts = err_to_rgy(cudaGetLastError());
        if (sts != RGY_ERR_NONE) {
            AddMessage(RGY_LOG_ERROR, _T("failed to run kernel_rand_init: %s.\n"), get_err_mes(sts));
            return sts;
        }

        kernel_gen_rand<GEN_RAND_BLOCK_LOOP_Y, false><<<grids, threads>>>(
            (int8_t *)m_RandY.frame.ptr[0], (int8_t *)m_RandUV.frame.ptr[0],
            m_RandY.frame.pitch[0], m_RandUV.frame.pitch[0],
            m_RandY.frame.width, m_RandY.frame.height,
            (curandState *)m_RandState.ptr);
        sts = err_to_rgy(cudaGetLastError());
        if (sts != RGY_ERR_NONE) {
            AddMessage(RGY_LOG_ERROR, _T("failed to run kernel_gen_rand: %s.\n"), get_err_mes(sts));
            return sts;
        }
    }

    setFilterInfo(pParam->print());
    m_param = pParam;
    return sts;
}

tstring NVEncFilterParamDeband::print() const {
    return deband.print();
}

RGY_ERR NVEncFilterDeband::run_filter(const RGYFrameInfo *pInputFrame, RGYFrameInfo **ppOutputFrames, int *pOutputFrameNum, cudaStream_t stream) {

    if (pInputFrame->ptr[0] == nullptr) {
        return RGY_ERR_NONE;
    }
    auto pDebandParam = std::dynamic_pointer_cast<NVEncFilterParamDeband>(m_param);
    if (!pDebandParam) {
        AddMessage(RGY_LOG_ERROR, _T("Invalid parameter type.\n"));
        return RGY_ERR_INVALID_PARAM;
    }

    *pOutputFrameNum = 1;
    if (ppOutputFrames[0] == nullptr) {
        auto pOutFrame = m_frameBuf[m_nFrameIdx].get();
        ppOutputFrames[0] = &pOutFrame->frame;
        m_nFrameIdx = (m_nFrameIdx + 1) % m_frameBuf.size();
    }

    const auto memcpyKind = getCudaMemcpyKind(pInputFrame->mem_type, ppOutputFrames[0]->mem_type);
    if (memcpyKind != cudaMemcpyDeviceToDevice) {
        AddMessage(RGY_LOG_ERROR, _T("only supported on device memory.\n"));
        return RGY_ERR_INVALID_CALL;
    }
    if (m_param->frameOut.csp != m_param->frameIn.csp) {
        AddMessage(RGY_LOG_ERROR, _T("csp does not match.\n"));
        return RGY_ERR_INVALID_CALL;
    }

    return deband(ppOutputFrames[0], pInputFrame, stream);
}

void NVEncFilterDeband::close() {
    m_RandState.clear();
    m_RandY.clear();
    m_RandUV.clear();
}
