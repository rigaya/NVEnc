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
#include <iostream>
#include <fstream>
#include <algorithm>
#include <numeric>
#define _USE_MATH_DEFINES
#include <cmath>
#include "convert_csp.h"
#include "NVEncFilterTransform.h"
#include "NVEncParam.h"
#pragma warning (push)
#pragma warning (disable: 4819)
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#pragma warning (pop)
#include "rgy_cuda_util_kernel.h"

static const int TRASNPOSE_BLOCK_DIM = 16;
static const int TRASNPOSE_TILE_DIM  = 64;

static const int FLIP_BLOCK_DIM = 16;

template<typename TypePixel4, bool flipX, bool flipY>
__global__ void kernel_transpose_plane(
    uint8_t *__restrict__ pDst,
    const int dstPitch,
    const int dstWidth,  // = srcHeight
    const int dstHeight, // = srcWidth
    const uint8_t *__restrict__ pSrc,
    const int srcPitch
    ) {
    __shared__ decltype(TypePixel4::x) stemp[TRASNPOSE_TILE_DIM][TRASNPOSE_TILE_DIM + 4];
    const int srcHeight = dstWidth;
    const int srcWidth  = dstHeight;
    const int dstBlockX = blockIdx.x;
    const int dstBlockY = blockIdx.y;
    const int srcBlockX = (flipX) ? gridDim.y - 1 - blockIdx.y : blockIdx.y;
    const int srcBlockY = (flipY) ? gridDim.x - 1 - blockIdx.x : blockIdx.x;
    const int offsetX = (flipX) ? srcWidth - ALIGN(srcWidth, TRASNPOSE_TILE_DIM) : 0;
    const int offsetY = (flipY) ? srcHeight - ALIGN(srcHeight, TRASNPOSE_TILE_DIM) : 0;
    {
        for (int j = threadIdx.y; j < TRASNPOSE_TILE_DIM; j += TRASNPOSE_BLOCK_DIM) {
            const int srcX = srcBlockX * TRASNPOSE_TILE_DIM + threadIdx.x * 4 + offsetX;
            const int srcY = srcBlockY * TRASNPOSE_TILE_DIM + j + offsetY;
            TypePixel4 val = { 128, 128, 128, 128 };
            if (srcX < srcWidth && srcY < srcHeight) {
                TypePixel4 *ptr_src = (TypePixel4 *)(pSrc + srcY * srcPitch + srcX * sizeof(TypePixel4::x));
                if ((offsetX & 3) == 0) {
                    val = ptr_src[0];
                } else {
                    decltype(TypePixel4::x) *ptr_src_elem = (decltype(TypePixel4::x) *)ptr_src;
                    val.x = ptr_src_elem[0];
                    val.y = ptr_src_elem[1];
                    val.z = ptr_src_elem[2];
                    val.w = ptr_src_elem[3];
                }
            }
            *(TypePixel4 *)&stemp[j][threadIdx.x * 4] = val;
        }
    }
    __syncthreads();

    {
        for (int j = threadIdx.y; j < TRASNPOSE_TILE_DIM; j += TRASNPOSE_BLOCK_DIM) {
            const int dstX = dstBlockX * TRASNPOSE_TILE_DIM + threadIdx.x * 4;
            const int dstY = dstBlockY * TRASNPOSE_TILE_DIM + j;
            const int tmpY = (flipX) ? TRASNPOSE_TILE_DIM - 1 - j : j;
            if (dstX < dstWidth && dstY < dstHeight) {
                TypePixel4 val = { 0, 0, 0, 0 };
                if (flipY) {
                    val.x = stemp[TRASNPOSE_TILE_DIM - (threadIdx.x+1) * 4 + 3][tmpY];
                    val.y = stemp[TRASNPOSE_TILE_DIM - (threadIdx.x+1) * 4 + 2][tmpY];
                    val.z = stemp[TRASNPOSE_TILE_DIM - (threadIdx.x+1) * 4 + 1][tmpY];
                    val.w = stemp[TRASNPOSE_TILE_DIM - (threadIdx.x+1) * 4 + 0][tmpY];
                } else {
                    val.x = stemp[threadIdx.x * 4 + 0][tmpY];
                    val.y = stemp[threadIdx.x * 4 + 1][tmpY];
                    val.z = stemp[threadIdx.x * 4 + 2][tmpY];
                    val.w = stemp[threadIdx.x * 4 + 3][tmpY];
                }
                TypePixel4 *ptr_dst = (TypePixel4 *)(pDst + dstY * dstPitch + dstX * sizeof(TypePixel4::x));
                *ptr_dst = val;
            }
        }
    }
};

template<typename TypePixel4, bool flipX, bool flipY>
RGY_ERR transpose_plane(
    RGYFrameInfo *pOutputFrame,
    const RGYFrameInfo *pInputFrame,
    cudaStream_t stream
) {
    dim3 blockSize(TRASNPOSE_BLOCK_DIM, TRASNPOSE_BLOCK_DIM);
    dim3 gridSize(
        divCeil(pOutputFrame->width, TRASNPOSE_TILE_DIM),
        divCeil(pOutputFrame->height, TRASNPOSE_TILE_DIM));

    kernel_transpose_plane<TypePixel4, flipX, flipY><<<gridSize, blockSize, 0, stream>>>(
        (uint8_t *)pOutputFrame->ptr[0],
        pOutputFrame->pitch[0],
        pOutputFrame->width,  // = srcHeight
        pOutputFrame->height, // = srcWidth
        (const uint8_t *)pInputFrame->ptr[0],
        pInputFrame->pitch[0]);
    auto cudaerr = cudaGetLastError();
    if (cudaerr != cudaSuccess) {
        return err_to_rgy(cudaerr);
    }
    return RGY_ERR_NONE;
}

template<typename TypePixel4, bool flipX, bool flipY>
__global__ void kernel_flip_plane(
    uint8_t *__restrict__ pDst,
    const int dstPitch,
    const int dstWidth,
    const int dstHeight,
    const uint8_t *__restrict__ pSrc,
    const int srcPitch
) {
    __shared__ decltype(TypePixel4::x) stemp[FLIP_BLOCK_DIM][FLIP_BLOCK_DIM*4];
    const int dstBlockX = blockIdx.x;
    const int dstBlockY = blockIdx.y;
    const int srcBlockX = (flipX) ? gridDim.x - 1 - blockIdx.x : blockIdx.x;
    const int srcBlockY = (flipY) ? gridDim.y - 1 - blockIdx.y : blockIdx.y;
    const int offsetX = (flipX) ? dstWidth - ALIGN(dstWidth, FLIP_BLOCK_DIM*4) : 0;
    const int offsetY = (flipY) ? dstHeight - ALIGN(dstHeight, FLIP_BLOCK_DIM) : 0;
    const int srcX = (srcBlockX * FLIP_BLOCK_DIM + threadIdx.x) * 4 + offsetX;
    const int srcY = srcBlockY * FLIP_BLOCK_DIM + threadIdx.y + offsetY;

    TypePixel4 val = { 128, 128, 128, 128 };
    if (srcX < dstWidth && srcY < dstHeight) {
        TypePixel4 *ptr_src = (TypePixel4 *)(pSrc + srcY * srcPitch + srcX * sizeof(TypePixel4::x));
        if ((offsetX & 3) == 0) {
            val = ptr_src[0];
        } else {
            decltype(TypePixel4::x) *ptr_src_elem = (decltype(TypePixel4::x) *)ptr_src;
            val.x = ptr_src_elem[0];
            val.y = ptr_src_elem[1];
            val.z = ptr_src_elem[2];
            val.w = ptr_src_elem[3];
        }
    }
    *(TypePixel4 *)&stemp[threadIdx.y][threadIdx.x * 4] = val;
    __syncthreads();

    const int dstX = (dstBlockX * FLIP_BLOCK_DIM + threadIdx.x) * 4;
    const int dstY = dstBlockY * FLIP_BLOCK_DIM + threadIdx.y;
    const int tmpY = (flipY) ? FLIP_BLOCK_DIM - 1 - threadIdx.y : threadIdx.y;
    val = *(TypePixel4 *)&stemp[tmpY][threadIdx.x * 4];
    if (flipX) {
        TypePixel4 val2 = *(TypePixel4 *)&stemp[tmpY][FLIP_BLOCK_DIM * 4 - (threadIdx.x + 1) * 4];
        val.x = val2.w;
        val.y = val2.z;
        val.z = val2.y;
        val.w = val2.x;
    } else {
        val = *(TypePixel4 *)&stemp[tmpY][threadIdx.x * 4];
    }
    if (dstX < dstWidth && dstY < dstHeight) {
        TypePixel4 *ptr_dst = (TypePixel4 *)(pDst + dstY * dstPitch + dstX * sizeof(TypePixel4::x));
        *ptr_dst = val;
    }
};

template<typename TypePixel4, bool flipX, bool flipY>
RGY_ERR flip_plane(
    RGYFrameInfo *pOutputFrame,
    const RGYFrameInfo *pInputFrame,
    cudaStream_t stream
) {
    dim3 blockSize(FLIP_BLOCK_DIM, FLIP_BLOCK_DIM);
    dim3 gridSize(
        divCeil(pOutputFrame->width, FLIP_BLOCK_DIM*4),
        divCeil(pOutputFrame->height, FLIP_BLOCK_DIM));

    kernel_flip_plane<TypePixel4, flipX, flipY> << <gridSize, blockSize, 0, stream >> > (
        (uint8_t *)pOutputFrame->ptr[0],
        pOutputFrame->pitch[0],
        pOutputFrame->width,
        pOutputFrame->height,
        (const uint8_t *)pInputFrame->ptr[0],
        pInputFrame->pitch[0]);
    auto cudaerr = cudaGetLastError();
    if (cudaerr != cudaSuccess) {
        return err_to_rgy(cudaerr);
    }
    return RGY_ERR_NONE;
}

template<typename TypePixel4>
RGY_ERR transform_plane(
    RGYFrameInfo *pOutputPlane,
    const RGYFrameInfo *pInputPlane,
    const std::shared_ptr<NVEncFilterParamTransform> pParam,
    cudaStream_t stream
) {
    if (pParam->trans.transpose) {
        if (pParam->trans.flipX && pParam->trans.flipY) {
            return transpose_plane<TypePixel4, true, true>(pOutputPlane, pInputPlane, stream);
        } else if (pParam->trans.flipX) {
            return transpose_plane<TypePixel4, true, false>(pOutputPlane, pInputPlane, stream);
        } else if (pParam->trans.flipY) {
            return transpose_plane<TypePixel4, false, true>(pOutputPlane, pInputPlane, stream);
        } else {
            return transpose_plane<TypePixel4, false, false>(pOutputPlane, pInputPlane, stream);
        }
    } else {
        if (pParam->trans.flipX && pParam->trans.flipY) {
            return flip_plane<TypePixel4, true, true>(pOutputPlane, pInputPlane, stream);
        } else if (pParam->trans.flipX) {
            return flip_plane<TypePixel4, true, false>(pOutputPlane, pInputPlane, stream);
        } else if (pParam->trans.flipY) {
            return flip_plane<TypePixel4, false, true>(pOutputPlane, pInputPlane, stream);
        } else {
            return flip_plane<TypePixel4, false, false>(pOutputPlane, pInputPlane, stream);
        }
    }
}

template<typename TypePixel4>
RGY_ERR transform_frame(RGYFrameInfo *pOutputFrame,
    const RGYFrameInfo *pInputFrame,
    const std::shared_ptr<NVEncFilterParamTransform> pParam,
    cudaStream_t stream
) {
    auto sts = RGY_ERR_NONE;
    const auto planeInputY = getPlane(pInputFrame, RGY_PLANE_Y);
    const auto planeInputU = getPlane(pInputFrame, RGY_PLANE_U);
    const auto planeInputV = getPlane(pInputFrame, RGY_PLANE_V);
    auto planeOutputY = getPlane(pOutputFrame, RGY_PLANE_Y);
    auto planeOutputU = getPlane(pOutputFrame, RGY_PLANE_U);
    auto planeOutputV = getPlane(pOutputFrame, RGY_PLANE_V);

    sts = transform_plane<TypePixel4>(&planeOutputY, &planeInputY, pParam, stream);
    if (sts != RGY_ERR_NONE) {
        return sts;
    }
    sts = transform_plane<TypePixel4>(&planeOutputU, &planeInputU, pParam, stream);
    if (sts != RGY_ERR_NONE) {
        return sts;
    }
    sts = transform_plane<TypePixel4>(&planeOutputV, &planeInputV, pParam, stream);
    if (sts != RGY_ERR_NONE) {
        return sts;
    }
    return sts;
}

NVEncFilterTransform::NVEncFilterTransform() : m_weight0(), m_weight1() {
    m_name = _T("transform");
}

NVEncFilterTransform::~NVEncFilterTransform() {
    close();
}

RGY_ERR NVEncFilterTransform::checkParam(const std::shared_ptr<NVEncFilterParamTransform> pNnediParam) {
    if (pNnediParam->frameOut.height <= 0 || pNnediParam->frameOut.width <= 0) {
        AddMessage(RGY_LOG_ERROR, _T("Invalid frame size.\n"));
        return RGY_ERR_INVALID_PARAM;
    }
    return RGY_ERR_NONE;
}

RGY_ERR NVEncFilterTransform::init(shared_ptr<NVEncFilterParam> pParam, shared_ptr<RGYLog> pPrintMes) {
    RGY_ERR sts = RGY_ERR_NONE;
    m_pLog = pPrintMes;
    auto prm = std::dynamic_pointer_cast<NVEncFilterParamTransform>(pParam);
    if (!prm) {
        AddMessage(RGY_LOG_ERROR, _T("Invalid parameter type.\n"));
        return RGY_ERR_INVALID_PARAM;
    }
    //パラメータチェック
    if ((sts = checkParam(prm)) != RGY_ERR_NONE) {
        return sts;
    }
    if (prm->trans.transpose) {
        prm->frameOut.width = prm->frameIn.height;
        prm->frameOut.height = prm->frameIn.width;
    }

    sts = AllocFrameBuf(prm->frameOut, 1);
    if (sts != RGY_ERR_NONE) {
        AddMessage(RGY_LOG_ERROR, _T("failed to allocate memory: %s.\n"), get_err_mes(sts));
        return sts;
    }
    for (int i = 0; i < RGY_CSP_PLANES[pParam->frameOut.csp]; i++) {
        prm->frameOut.pitch[i] = m_frameBuf[0]->frame.pitch[i];
    }

    setFilterInfo(pParam->print());
    m_param = pParam;
    return sts;
}

tstring NVEncFilterParamTransform::print() const {
    return trans.print();
}

RGY_ERR NVEncFilterTransform::run_filter(const RGYFrameInfo *pInputFrame, RGYFrameInfo **ppOutputFrames, int *pOutputFrameNum, cudaStream_t stream) {
    RGY_ERR sts = RGY_ERR_NONE;
    if (pInputFrame->ptr[0] == nullptr) {
        return sts;
    }
    auto prm = std::dynamic_pointer_cast<NVEncFilterParamTransform>(m_param);
    if (!prm) {
        AddMessage(RGY_LOG_ERROR, _T("Invalid parameter type.\n"));
        return RGY_ERR_INVALID_PARAM;
    }

    *pOutputFrameNum = 1;
    if (ppOutputFrames[0] == nullptr) {
        auto pOutFrame = m_frameBuf[m_nFrameIdx].get();
        ppOutputFrames[0] = &pOutFrame->frame;
        ppOutputFrames[0]->picstruct = pInputFrame->picstruct;
        m_nFrameIdx = (m_nFrameIdx + 1) % m_frameBuf.size();
    }

    const auto memcpyKind = getCudaMemcpyKind(pInputFrame->mem_type, ppOutputFrames[0]->mem_type);
    if (memcpyKind != cudaMemcpyDeviceToDevice) {
        AddMessage(RGY_LOG_ERROR, _T("only supported on device memory.\n"));
        return RGY_ERR_UNSUPPORTED;
    }
    if (m_param->frameOut.csp != m_param->frameIn.csp) {
        AddMessage(RGY_LOG_ERROR, _T("csp does not match.\n"));
        return RGY_ERR_UNSUPPORTED;
    }

    static const std::map<RGY_CSP, decltype(transform_frame<uchar4>)*> func_list = {
        { RGY_CSP_YV12,      transform_frame<uchar4> },
        { RGY_CSP_YV12_16,   transform_frame<ushort4> },
        { RGY_CSP_YUV444,    transform_frame<uchar4> },
        { RGY_CSP_YUV444_16, transform_frame<ushort4> }
    };
    if (func_list.count(pInputFrame->csp) == 0) {
        AddMessage(RGY_LOG_ERROR, _T("unsupported csp %s.\n"), RGY_CSP_NAMES[pInputFrame->csp]);
        return RGY_ERR_UNSUPPORTED;
    }
    sts = func_list.at(pInputFrame->csp)(ppOutputFrames[0], pInputFrame,
        prm, stream
        );
    if (sts != RGY_ERR_NONE) {
        AddMessage(RGY_LOG_ERROR, _T("error at transform(%s): %s.\n"),
            RGY_CSP_NAMES[pInputFrame->csp],
            get_err_mes(sts));
        return sts;
    }
    return sts;
}

void NVEncFilterTransform::close() {
    m_frameBuf.clear();
}
