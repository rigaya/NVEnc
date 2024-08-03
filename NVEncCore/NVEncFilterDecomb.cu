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
#include "convert_csp.h"
#include "NVEncFilterDecomb.h"
#include "NVEncParam.h"
#include "rgy_cuda_util_kernel.h"

using TypeMask = uint8_t;

#define DECOMB_BLOCK_X  (32) //work groupサイズ(x) = スレッド数/work group
#define DECOMB_BLOCK_Y   (8) //work groupサイズ(y) = スレッド数/work group

template<typename TypePixel, bool full>
__global__ void kernel_motion_map(
    uint8_t *__restrict__ dmaskp, uint8_t *__restrict__ fmaskp, const int dpitch,
    const uint8_t *__restrict__ srcp, const  int pitch,
    const int w, const int h,
    const float threshold, const float dthreshold) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < w && y < h) {
        TypeMask fmask = 0, dmask = 0;
        if (y > 0 && y < h - 1) {
            const float srcp0 = (float)((const TypePixel*)(srcp + max(y - 1, 0)     * pitch + x * sizeof(TypePixel)))[0];
            const float srcp1 = (float)((const TypePixel*)(srcp + y                 * pitch + x * sizeof(TypePixel)))[0];
            const float srcp2 = (float)((const TypePixel*)(srcp + min(y + 1, h - 1) * pitch + x * sizeof(TypePixel)))[0];
            const float val = (float)(srcp2 - srcp1) * (float)(srcp0 - srcp1);
            if (!full && val > threshold) {
                fmask = (TypeMask)0xff;
            }
            if (val > dthreshold) {
                dmask = (TypeMask)0xff;
            }
        } else {
            dmask = (TypeMask)0xff;
        }
        *(TypeMask *)(dmaskp + y * dpitch + x * sizeof(TypeMask)) = dmask;
        *(TypeMask *)(fmaskp + y * dpitch + x * sizeof(TypeMask)) = fmask;
    }
}

template<typename TypePixel, bool full>
RGY_ERR create_motion_map(
    RGYFrameInfo *pDmaskPlane,
    RGYFrameInfo *pFmaskPlane,
    const RGYFrameInfo *pSrcPlane,
    const float threshold, const float dthreshold,
    cudaStream_t stream) {

    dim3 blockSize(DECOMB_BLOCK_X, DECOMB_BLOCK_Y);
    dim3 gridSize(divCeil(pSrcPlane->width, blockSize.x), divCeil(pSrcPlane->height, blockSize.y));

    kernel_motion_map<TypePixel, full><<<gridSize, blockSize, 0, stream>>>(
        pDmaskPlane->ptr[0],
        pFmaskPlane->ptr[0], pDmaskPlane->pitch[0],
        pSrcPlane->ptr[0], pSrcPlane->pitch[0],
        pSrcPlane->width, pSrcPlane->height,
        threshold, dthreshold);
    auto sts = err_to_rgy(cudaGetLastError());
    if (sts != RGY_ERR_NONE) {
        return sts;
    }
    return sts;
}

template<typename TypePixel>
RGY_ERR motion_map(
    RGYFrameInfo *pDmaskPlane,
    RGYFrameInfo *pFmaskPlane,
    const RGYFrameInfo *pSrcPlane,
    const float threshold, const float dthreshold,
    const bool full, cudaStream_t stream) {

    if (full) {
        return create_motion_map<TypePixel, true>(pDmaskPlane, pFmaskPlane, pSrcPlane, threshold, dthreshold, stream);
    } else {
        return create_motion_map<TypePixel, false>(pDmaskPlane, pFmaskPlane, pSrcPlane, threshold, dthreshold, stream);
    }
}

__inline__ __device__
int is_combed_count(const TypeMask fm0, const TypeMask fm1, const TypeMask fm2) {
    return (fm0 == 0xff && fm1 == 0xff && fm2 == 0xff) ? 1 : 0;
}

__inline__ __device__
int is_combed_box_x_count(const uchar4 fm0, const uchar4 fm1, const uchar4 fm2) {
    return is_combed_count(fm0.x, fm1.x, fm2.x)
         + is_combed_count(fm0.y, fm1.y, fm2.y)
         + is_combed_count(fm0.z, fm1.z, fm2.z)
         + is_combed_count(fm0.w, fm1.w, fm2.w);
}

template<typename TypeMaskVec, int BOX_X_LOG2, int BOX_Y_LOG2>
__global__ void kernel_is_combed(
    int *__restrict__ isCombed,
    const uint8_t *__restrict__ fmaskp, const int dpitch,
    const int w, const int h,
    const int CT
) {
    static_assert(sizeof(TypeMask) == sizeof(TypeMaskVec::x), "size mismatch!");
    static_assert((sizeof(TypeMask) << BOX_X_LOG2) == sizeof(TypeMaskVec), "size mismatch!");
    static_assert((1 << BOX_Y_LOG2) <= WARP_SIZE, "(1 << BOX_Y_LOG2) <= WARP_SIZE");
    // 1threadはx方向にBOX_X_LOG2 pixelを処理、さらにthread.xでy方向にBOX_Y_LOG2pixelを処理
    // thread.yはx方向にBOX_X_LOG2 pixelずつ処理
    const int x = (blockIdx.x * blockDim.x + threadIdx.y) << BOX_X_LOG2;
    const int y = blockIdx.y * blockDim.y + threadIdx.x;

    __shared__ int block_result;
    if (threadIdx.x == 0 && threadIdx.y == 0) {
        block_result = 0;
    }
    __syncthreads();

    int count = 0;
    if (x < w && y < h) {

        const TypeMaskVec *fmask0 = (const TypeMaskVec *)(fmaskp + max(y - 1, 0)     * dpitch + x * sizeof(TypeMask));
        const TypeMaskVec *fmask1 = (const TypeMaskVec *)(fmaskp + y                 * dpitch + x * sizeof(TypeMask));
        const TypeMaskVec *fmask2 = (const TypeMaskVec *)(fmaskp + min(y + 1, h - 1) * dpitch + x * sizeof(TypeMask));

        const TypeMaskVec fm0 = fmask0[0];
        const TypeMaskVec fm1 = fmask1[0];
        const TypeMaskVec fm2 = fmask2[0];

        count = is_combed_box_x_count(fm0, fm1, fm2);
    }

    count = warp_sum<decltype(count), (1 << BOX_Y_LOG2)>(count);
    if (threadIdx.x == 0) {
        // threadIdx.x == 0のcountは各boxの判定結果
        // const int box = (y >> BOX_Y_LOG2) * (x >> BOX_X_LOG2);
        // まずsharedメモリ内でこのブロックの判定結果を作る
        // とりあえず1になればよいので、atomicAddで書き込む必要はなさそう
        if (count > CT) {
            block_result = 1;
        }
    }
    __syncthreads();
    // このブロックの判定結果をグローバルに書き出し
    // とりあえず1になればよいので、atomicAddで書き込む必要はなさそう
    if (threadIdx.x == 0 && threadIdx.y == 0 && block_result > 0) {
        isCombed[0] = 1;
    }
}

RGY_ERR is_combed(
    CUMemBuf *pResultIsCombed,
    const RGYFrameInfo *pFmaskPlane,
    const bool full,
    cudaStream_t stream) {
    static const int BOX_X_LOG2 = 2;
    static const int BOX_Y_LOG2 = 3;
    static const int CT = 15;

    auto sts = err_to_rgy(cudaMemsetAsync(pResultIsCombed->ptr, full ? 1 : 0, sizeof(int), stream));
    if (full || sts != RGY_ERR_NONE) {
        return sts;
    }

    dim3 blockSize(1 << BOX_Y_LOG2, 64);
    dim3 gridSize(divCeil(pFmaskPlane->width, blockSize.x), divCeil(pFmaskPlane->height, blockSize.y));

    kernel_is_combed<uchar4, BOX_X_LOG2, BOX_Y_LOG2><<<gridSize, blockSize>>>(
        (int *)pResultIsCombed->ptr,
        pFmaskPlane->ptr[0], pFmaskPlane->pitch[0],
        pFmaskPlane->width, pFmaskPlane->height,
        CT
    );
    sts = err_to_rgy(cudaGetLastError());
    if (sts != RGY_ERR_NONE) {
        return sts;
    }
    return sts;
}

template<typename TypePixel, bool blend>
__global__ void kernel_deinterlace(
    unsigned char *__restrict__ ptr_dst,
    const unsigned char *__restrict__ ptr_src, const int pitch,
    const int w, const int h,
    const unsigned char *__restrict__ dmaskp, const int dpitch,
    const int *__restrict__ isCombed,
    const bool uv420
) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < w && y < h) {
        TypePixel       *ptrDst  = (TypePixel *      )(ptr_dst + y                 * pitch  + x * sizeof(TypePixel));
        const TypePixel *ptrSrc0 = (const TypePixel *)(ptr_src + max(y - 1, 0)     * pitch + x * sizeof(TypePixel));
        const TypePixel *ptrSrc1 = (const TypePixel *)(ptr_src + y                 * pitch + x * sizeof(TypePixel));
        const TypePixel *ptrSrc2 = (const TypePixel *)(ptr_src + min(y + 1, h - 1) * pitch + x * sizeof(TypePixel));
        const TypeMask  *ptrMask = (const TypeMask * )(dmaskp  + ((uv420) ? y*2 : y)   * dpitch + ((uv420) ? x * 2 : x) * sizeof(TypeMask));

        if (blend) {
            TypePixel pix = ptrSrc1[0];
            if (isCombed[0]) {
                if (y == 0) {
                    pix = (TypePixel)(((int)pix + (int)ptrSrc2[0] + 1) >> 1);
                } else if (y == h - 1) {
                    pix = (TypePixel)(((int)pix + (int)ptrSrc0[0] + 1) >> 1);
                } else if (ptrMask[0]) {
                    pix = (TypePixel)((((int)pix * 2) + (int)ptrSrc0[0] + (int)ptrSrc2[0] + 3) >> 2);
                }
            }
            ptrDst[0] = pix;
        } else {
            TypePixel pix;
            if ((y & 1) && y < h-1 && isCombed[0] && ptrMask[0]) {
                pix = (TypePixel)(((int)ptrSrc0[0] + (int)ptrSrc2[0] + 1) >> 1);
            } else {
                pix = ptrSrc1[0];
            }
            ptrDst[0] = pix;
        }
    }
}

template<typename TypePixel>
RGY_ERR deinterlace_plane(
    RGYFrameInfo *pDstPlane,
    const RGYFrameInfo *pSrcPlane,
    const RGYFrameInfo *pDmaskPlane,
    const CUMemBuf *pResultIsCombed,
    const bool blend, const bool uv420, cudaStream_t stream
) {
    dim3 blockSize(DECOMB_BLOCK_X, DECOMB_BLOCK_Y);
    dim3 gridSize(divCeil(pSrcPlane->width, blockSize.x), divCeil(pSrcPlane->height, blockSize.y));

    if (blend) {
        kernel_deinterlace<TypePixel, true><<<gridSize, blockSize, 0, stream>>>(
            pDstPlane->ptr[0],
            pSrcPlane->ptr[0], pSrcPlane->pitch[0],
            pSrcPlane->width, pSrcPlane->height,
            pDmaskPlane->ptr[0], pDmaskPlane->pitch[0],
            (int *)pResultIsCombed->ptr, uv420);
    } else {
        kernel_deinterlace<TypePixel, false><<<gridSize, blockSize, 0, stream>>>(
            pDstPlane->ptr[0],
            pSrcPlane->ptr[0], pSrcPlane->pitch[0],
            pSrcPlane->width, pSrcPlane->height,
            pDmaskPlane->ptr[0], pDmaskPlane->pitch[0],
            (int *)pResultIsCombed->ptr, uv420);
    }
    auto sts = err_to_rgy(cudaGetLastError());
    if (sts != RGY_ERR_NONE) {
        return sts;
    }
    return sts;
}

template<typename TypePixel>
RGY_ERR run_decomb_frame(
    RGYFrameInfo *pOutputFrame,
    RGYFrameInfo *pDmaskFrame,
    RGYFrameInfo *pFmaskFrame,
    CUMemBuf *pResultIsCombed,
    const RGYFrameInfo *pSrcFrame,
    const int threshold, const int dthreshold,
    const bool blend, const bool full,
    cudaStream_t stream) {
    const float threshould_mul = (sizeof(TypePixel) == 2) ? 65535.0f : 1.0f;
    auto sts = motion_map<TypePixel>(pDmaskFrame, pFmaskFrame, pSrcFrame, (float)threshold * threshould_mul, (float)dthreshold * threshould_mul, full, stream);
    if (sts != RGY_ERR_NONE) {
        return sts;
    }

    sts = is_combed(pResultIsCombed, pFmaskFrame, full, stream);
    if (sts != RGY_ERR_NONE) {
        return sts;
    }

    for (int iplane = 0; iplane < RGY_CSP_PLANES[pSrcFrame->csp]; iplane++) {
        const auto planeSrc = getPlane(pSrcFrame, (RGY_PLANE)iplane);
        auto planeOutput = getPlane(pOutputFrame, (RGY_PLANE)iplane);
        const bool uv420 = RGY_CSP_CHROMA_FORMAT[pSrcFrame->csp] == RGY_CHROMAFMT_YUV420
            && ((RGY_PLANE)iplane == RGY_PLANE_U || ((RGY_PLANE)iplane == RGY_PLANE_V));
        sts = deinterlace_plane<TypePixel>(&planeOutput, &planeSrc, pDmaskFrame, pResultIsCombed, blend, uv420, stream);
        if (sts != RGY_ERR_NONE) {
            return sts;
        }
    }
    return RGY_ERR_NONE;
}

NVEncFilterDecomb::NVEncFilterDecomb() : m_dmask(), m_fmask(), m_isCombed() {
    m_name = _T("decomb");
}

NVEncFilterDecomb::~NVEncFilterDecomb() {
    close();
}

RGY_ERR NVEncFilterDecomb::check_param(shared_ptr<NVEncFilterParamDecomb> prm) {
    if (prm->frameOut.height <= 0 || prm->frameOut.width <= 0) {
        AddMessage(RGY_LOG_ERROR, _T("Invalid parameter.\n"));
        return RGY_ERR_INVALID_PARAM;
    }
    const int hight_mul = (RGY_CSP_CHROMA_FORMAT[prm->frameOut.csp] == RGY_CHROMAFMT_YUV420) ? 4 : 2;
    if ((prm->frameOut.height % hight_mul) != 0) {
        AddMessage(RGY_LOG_ERROR, _T("Height must be multiple of %d.\n"), hight_mul);
        return RGY_ERR_INVALID_PARAM;
    }
    if (prm->decomb.threshold < 0 && 255 <= prm->decomb.threshold) {
        AddMessage(RGY_LOG_ERROR, _T("Invalid parameter (threshold).\n"));
        return RGY_ERR_INVALID_PARAM;
    }
    if (prm->decomb.dthreshold < 0 && 255 <= prm->decomb.dthreshold) {
        AddMessage(RGY_LOG_ERROR, _T("Invalid parameter (dthreshold).\n"));
        return RGY_ERR_INVALID_PARAM;
    }
    return RGY_ERR_NONE;
}

RGY_ERR NVEncFilterDecomb::init(shared_ptr<NVEncFilterParam> pParam, shared_ptr<RGYLog> pPrintMes) {
    RGY_ERR sts = RGY_ERR_NONE;
    m_pLog = pPrintMes;
    auto prm = std::dynamic_pointer_cast<NVEncFilterParamDecomb>(pParam);
    if (!prm) {
        AddMessage(RGY_LOG_ERROR, _T("Invalid parameter type.\n"));
        return RGY_ERR_INVALID_PARAM;
    }
    //パラメータチェック
    if (check_param(prm) != RGY_ERR_NONE) {
        return RGY_ERR_INVALID_PARAM;
    }

    sts = AllocFrameBuf(prm->frameOut, 1);
    if (sts != RGY_ERR_NONE) {
        AddMessage(RGY_LOG_ERROR, _T("failed to allocate memory: %s.\n"), get_err_mes(sts));
        return sts;
    }
    for (int i = 0; i < RGY_CSP_PLANES[pParam->frameOut.csp]; i++) {
        prm->frameOut.pitch[i] = m_frameBuf[0]->frame.pitch[i];
    }
    AddMessage(RGY_LOG_DEBUG, _T("allocated output buffer: %dx%pixym1[3], pitch %pixym1[3], %s.\n"),
        m_frameBuf[0]->frame.width, m_frameBuf[0]->frame.height, m_frameBuf[0]->frame.pitch[0], RGY_CSP_NAMES[m_frameBuf[0]->frame.csp]);

    if (!m_dmask
        || m_dmask->width() != prm->frameOut.width
        || m_dmask->height() != prm->frameOut.height) {
        auto frame = prm->frameOut;
        frame.csp = RGY_CSP_Y8;
        m_dmask = std::make_unique<CUFrameBuf>(frame);
        sts = m_dmask->alloc();
        if (sts != RGY_ERR_NONE) {
            AddMessage(RGY_LOG_ERROR, _T("failed to allocate memory: %s.\n"), get_err_mes(sts));
            return sts;
        }
    }
    if (!m_fmask
        || m_fmask->width()  != prm->frameOut.width
        || m_fmask->height() != prm->frameOut.height) {
        auto frame = prm->frameOut;
        frame.csp = RGY_CSP_Y8;
        m_fmask = std::make_unique<CUFrameBuf>(frame);
        sts = m_fmask->alloc();
        if (sts != RGY_ERR_NONE) {
            AddMessage(RGY_LOG_ERROR, _T("failed to allocate memory: %s.\n"), get_err_mes(sts));
            return sts;
        }
    }
    if (!m_isCombed) {
        m_isCombed = std::make_unique<CUMemBuf>(sizeof(int));
        sts = m_isCombed->alloc();
        if (sts != RGY_ERR_NONE) {
            AddMessage(RGY_LOG_ERROR, _T("failed to allocate memory: %s.\n"), get_err_mes(sts));
            return sts;
        }
    }

    prm->frameOut.picstruct = RGY_PICSTRUCT_FRAME;
    m_pathThrough &= (~(FILTER_PATHTHROUGH_PICSTRUCT | FILTER_PATHTHROUGH_FLAGS));

    setFilterInfo(pParam->print());
    m_param = pParam;
    return sts;
}

tstring NVEncFilterParamDecomb::print() const {
    return decomb.print();
}

RGY_ERR NVEncFilterDecomb::run_filter(const RGYFrameInfo *pInputFrame, RGYFrameInfo **ppOutputFrames, int *pOutputFrameNum, cudaStream_t stream) {
    RGY_ERR sts = RGY_ERR_NONE;
    if (pInputFrame->ptr[0] == nullptr) {
        return sts;
    }

    auto prm = std::dynamic_pointer_cast<NVEncFilterParamDecomb>(m_param);
    if (!prm) {
        AddMessage(RGY_LOG_ERROR, _T("Invalid parameter type.\n"));
        return RGY_ERR_INVALID_PARAM;
    }

    *pOutputFrameNum = 1;
    if (ppOutputFrames[0] == nullptr) {
        auto pOutFrame = m_frameBuf[m_nFrameIdx].get();
        ppOutputFrames[0] = &pOutFrame->frame;
        m_nFrameIdx = (m_nFrameIdx + 1) % m_frameBuf.size();
    }
    
    if (interlaced(*pInputFrame)) {
        static const std::map<RGY_DATA_TYPE, decltype(run_decomb_frame<uint8_t>)*> func_list = {
            { RGY_DATA_TYPE_U8,  run_decomb_frame<uint8_t > },
            { RGY_DATA_TYPE_U16, run_decomb_frame<uint16_t> }
        };
        if (func_list.count(RGY_CSP_DATA_TYPE[pInputFrame->csp]) == 0) {
            AddMessage(RGY_LOG_ERROR, _T("unsupported csp %s.\n"), RGY_CSP_NAMES[pInputFrame->csp]);
            return RGY_ERR_UNSUPPORTED;
        }
        sts = func_list.at(RGY_CSP_DATA_TYPE[pInputFrame->csp])(ppOutputFrames[0],
            &m_dmask->frame, &m_fmask->frame, m_isCombed.get(), pInputFrame,
            prm->decomb.threshold, prm->decomb.dthreshold,
            prm->decomb.blend, prm->decomb.full, stream);
        if (sts != RGY_ERR_NONE) {
            AddMessage(RGY_LOG_ERROR, _T("failed to deinterlace frame: %s.\n"), get_err_mes(sts));
            return sts;
        }
    } else {
        //ppOutputFrames[0]にコピー
        sts = copyFrameAsync(ppOutputFrames[0], pInputFrame, stream);
        if (sts != RGY_ERR_NONE) {
            AddMessage(RGY_LOG_ERROR, _T("failed to copy frame: %s.\n"), get_err_mes(sts));
            return sts;
        }
    }
    ppOutputFrames[0]->picstruct = RGY_PICSTRUCT_FRAME;
    return RGY_ERR_NONE;
}

void NVEncFilterDecomb::close() {
    m_dmask.reset();
    m_fmask.reset();
    m_isCombed.reset();
    AddMessage(RGY_LOG_DEBUG, _T("closed decomb filter.\n"));
}
