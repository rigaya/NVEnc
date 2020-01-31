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
#include "NVEncFilterYadif.h"
#include "NVEncParam.h"

template<typename T>
__inline__ __device__
T max3(T a, T b, T c) {
    return max(max(a, b), c);
}
template<typename T>
__inline__ __device__
T min3(T a, T b, T c) {
    return min(min(a, b), c);
}

template<typename TypePixel>
__inline__ __device__ int spatial(
    cudaTextureObject_t tex1,
    const float gIdXf,
    const float gIdYf
) {
    int ym1[7], yp1[7];
    #pragma unroll
    for (int ix = -3; ix <= 3; ix++) {
        ym1[ix+3] = (int)tex2D<TypePixel>(tex1, gIdXf + ix, gIdYf - 1.0f);
        yp1[ix+3] = (int)tex2D<TypePixel>(tex1, gIdXf + ix, gIdYf + 1.0f);
    }

    const int score[5] = {
        abs(ym1[2] - yp1[2]) + abs(ym1[3] - yp1[3]) + abs(ym1[4] - yp1[4]),
        abs(ym1[1] - yp1[3]) + abs(ym1[2] - yp1[4]) + abs(ym1[3] - yp1[5]),
        abs(ym1[0] - yp1[4]) + abs(ym1[1] - yp1[5]) + abs(ym1[2] - yp1[6]),
        abs(ym1[3] - yp1[1]) + abs(ym1[4] - yp1[2]) + abs(ym1[5] - yp1[3]),
        abs(ym1[4] - yp1[0]) + abs(ym1[5] - yp1[1]) + abs(ym1[6] - yp1[2])
    };
    int minscore = score[0];
    int minidx = 0;
    if (score[1] < minscore) {
        minscore = score[1];
        minidx = 1;
        if (score[2] < minscore) {
            minscore = score[2];
            minidx = 2;
        }
    }
    if (score[3] < minscore) {
        minscore = score[3];
        minidx = 3;
        if (score[4] < minscore) {
            minscore = score[4];
            minidx = 4;
        }
    }

    switch (minidx) {
    case 0: return (ym1[3] + yp1[3]) >> 1;
    case 1: return (ym1[2] + yp1[4]) >> 1;
    case 2: return (ym1[1] + yp1[5]) >> 1;
    case 3: return (ym1[4] + yp1[2]) >> 1;
    case 4:
    default:return (ym1[5] + yp1[1]) >> 1;
    }
}

template<typename TypePixel>
__inline__ __device__ const int temporal(
    cudaTextureObject_t tex0,
    cudaTextureObject_t tex01,
    cudaTextureObject_t tex1,
    cudaTextureObject_t tex12,
    cudaTextureObject_t tex2,
    const int valSpatial,
    const float gIdXf,
    const float gIdYf
) {
    const int t00m1 = (int)tex2D<TypePixel>(tex0,  gIdXf, gIdYf - 1.0f);
    const int t00p1 = (int)tex2D<TypePixel>(tex0,  gIdXf, gIdYf + 1.0f);
    const int t01m2 = (int)tex2D<TypePixel>(tex01, gIdXf, gIdYf - 2.0f);
    const int t01_0 = (int)tex2D<TypePixel>(tex01, gIdXf, gIdYf + 0.0f);
    const int t01p2 = (int)tex2D<TypePixel>(tex01, gIdXf, gIdYf + 2.0f);
    const int t10m1 = (int)tex2D<TypePixel>(tex1,  gIdXf, gIdYf - 1.0f);
    const int t10p1 = (int)tex2D<TypePixel>(tex1,  gIdXf, gIdYf + 1.0f);
    const int t12m2 = (int)tex2D<TypePixel>(tex12, gIdXf, gIdYf - 2.0f);
    const int t12_0 = (int)tex2D<TypePixel>(tex12, gIdXf, gIdYf + 0.0f);
    const int t12p2 = (int)tex2D<TypePixel>(tex12, gIdXf, gIdYf + 2.0f);
    const int t20m1 = (int)tex2D<TypePixel>(tex2,  gIdXf, gIdYf - 1.0f);
    const int t20p1 = (int)tex2D<TypePixel>(tex2,  gIdXf, gIdYf + 1.0f);
    const int tm2 = (t01m2 + t12m2) >> 1;
    const int t_0 = (t01_0 + t12_0) >> 1;
    const int tp2 = (t01p2 + t12p2) >> 1;


    int diff = max3(
        abs(t01_0 - t12_0),
        (abs(t00m1 - t10m1) + abs(t00p1 - t10p1)) >> 1,
        (abs(t20m1 - t10m1) + abs(t10p1 - t20p1)) >> 1);
    diff = max3(diff,
                -max3(t_0 - t10p1, t_0 - t10m1, min(tm2 - t10m1, tp2 - t10p1)),
                min3(t_0 - t10p1, t_0 - t10m1, max(tm2 - t10m1, tp2 - t10p1)));
    return max(min(valSpatial, t_0 + diff), t_0 - diff);
}

template<typename TypePixel, int bit_depth, int BLOCK_X, int BLOCK_Y>
__global__ void kernel_yadif(
    TypePixel *ptrDst,
    cudaTextureObject_t tex0,
    cudaTextureObject_t tex1,
    cudaTextureObject_t tex2,
    const int dstPitch,
    const int dstWidth,
    const int dstHeight,
    const int srcWidth,
    const int srcHeight,
    const YadifTargetField targetField,
    const RGY_PICSTRUCT picstruct) {
    const int gIdX = blockIdx.x * BLOCK_X + threadIdx.x;
    const int gIdY = blockIdx.y * BLOCK_Y + threadIdx.y;
    if (gIdX < dstWidth && gIdY < dstHeight) {
        const float gIdXf = gIdX + 0.5f;
        const float gIdYf = gIdY + 0.5f;

        TypePixel ret;
        if ((gIdY & 1) != targetField) {
            ret = tex2D<TypePixel>(tex1, gIdXf, gIdYf);
        } else {
            const int valSpatial = spatial<TypePixel>(tex1, gIdXf, gIdYf);
            const bool field2nd = ((targetField==YADIF_GEN_FIELD_TOP) == (((uint32_t)picstruct & (uint32_t)RGY_PICSTRUCT_TFF) != 0));
            cudaTextureObject_t tex01 = field2nd ? tex1 : tex0;
            cudaTextureObject_t tex12 = field2nd ? tex2 : tex1;
            ret = (TypePixel)clamp(
                temporal<TypePixel>(tex0, tex01, tex1, tex12, tex2, valSpatial, gIdXf, gIdYf),
                0, ((1<<bit_depth)-1));
        }
        *(TypePixel *)((uint8_t *)ptrDst + gIdY * dstPitch + gIdX * sizeof(TypePixel)) = ret;
    }
}

template<typename TypePixel>
cudaError_t setTexField(cudaTextureObject_t& texSrc, const FrameInfo *pFrame) {
    texSrc = 0;

    cudaResourceDesc resDescSrc;
    memset(&resDescSrc, 0, sizeof(resDescSrc));
    resDescSrc.resType = cudaResourceTypePitch2D;
    resDescSrc.res.pitch2D.desc = cudaCreateChannelDesc<TypePixel>();
    resDescSrc.res.pitch2D.pitchInBytes = pFrame->pitch;
    resDescSrc.res.pitch2D.width = pFrame->width;
    resDescSrc.res.pitch2D.height = pFrame->height;
    resDescSrc.res.pitch2D.devPtr = (uint8_t *)pFrame->ptr;

    cudaTextureDesc texDescSrc;
    memset(&texDescSrc, 0, sizeof(texDescSrc));
    texDescSrc.addressMode[0]   = cudaAddressModeWrap;
    texDescSrc.addressMode[1]   = cudaAddressModeWrap;
    texDescSrc.filterMode       = cudaFilterModePoint;
    texDescSrc.readMode         = cudaReadModeElementType;
    texDescSrc.normalizedCoords = 0;

    return cudaCreateTextureObject(&texSrc, &resDescSrc, &texDescSrc, nullptr);
}

template<typename TypePixel, int bit_depth>
cudaError_t run_yadif(FrameInfo *pOutputPlane,
    const FrameInfo *pSrc0,
    const FrameInfo *pSrc1,
    const FrameInfo *pSrc2,
    const YadifTargetField targetField,
    const RGY_PICSTRUCT picstruct,
    cudaStream_t stream) {
    cudaTextureObject_t texSrc0 = 0;
    cudaTextureObject_t texSrc1 = 0;
    cudaTextureObject_t texSrc2 = 0;
    auto cudaerr = cudaSuccess;
    if (   (cudaerr = setTexField<TypePixel>(texSrc0, pSrc0)) != cudaSuccess
        || (cudaerr = setTexField<TypePixel>(texSrc1, pSrc1)) != cudaSuccess
        || (cudaerr = setTexField<TypePixel>(texSrc2, pSrc2)) != cudaSuccess) {
        return cudaerr;
    }

    static const int YADIF_BLOCK_X = 32;
    static const int YADIF_BLOCK_Y = 8;
    dim3 blockSize(YADIF_BLOCK_X, YADIF_BLOCK_Y);
    dim3 gridSize(divCeil(pOutputPlane->width, blockSize.x), divCeil(pOutputPlane->height, blockSize.y));

    kernel_yadif<TypePixel, bit_depth, YADIF_BLOCK_X, YADIF_BLOCK_Y><<<gridSize, blockSize, 0, stream>>>(
        (TypePixel * )pOutputPlane->ptr,
        texSrc0, texSrc1, texSrc2,
        pOutputPlane->pitch,
        pOutputPlane->width,
        pOutputPlane->height,
        pSrc1->width,
        pSrc1->height,
        targetField, picstruct);
    cudaerr = cudaGetLastError();
    cudaDestroyTextureObject(texSrc0);
    cudaDestroyTextureObject(texSrc1);
    cudaDestroyTextureObject(texSrc2);
    return cudaerr;
}


template<typename TypePixel, int bit_depth>
cudaError_t run_yadif_frame(FrameInfo *pOutputFrame,
    const FrameInfo *pSrc0,
    const FrameInfo *pSrc1,
    const FrameInfo *pSrc2,
    const YadifTargetField targetField,
    const RGY_PICSTRUCT picstruct,
    cudaStream_t stream) {
    const auto planeSrc0Y = getPlane(pSrc0, RGY_PLANE_Y);
    const auto planeSrc0U = getPlane(pSrc0, RGY_PLANE_U);
    const auto planeSrc0V = getPlane(pSrc0, RGY_PLANE_V);
    const auto planeSrc1Y = getPlane(pSrc1, RGY_PLANE_Y);
    const auto planeSrc1U = getPlane(pSrc1, RGY_PLANE_U);
    const auto planeSrc1V = getPlane(pSrc1, RGY_PLANE_V);
    const auto planeSrc2Y = getPlane(pSrc2, RGY_PLANE_Y);
    const auto planeSrc2U = getPlane(pSrc2, RGY_PLANE_U);
    const auto planeSrc2V = getPlane(pSrc2, RGY_PLANE_V);
    auto planeOutputY = getPlane(pOutputFrame, RGY_PLANE_Y);
    auto planeOutputU = getPlane(pOutputFrame, RGY_PLANE_U);
    auto planeOutputV = getPlane(pOutputFrame, RGY_PLANE_V);

    auto cudaerr = run_yadif<TypePixel, bit_depth>(&planeOutputY, &planeSrc0Y, &planeSrc1Y, &planeSrc2Y, targetField, picstruct, stream);
    if (cudaerr != cudaSuccess) {
        return cudaerr;
    }
    cudaerr = run_yadif<TypePixel, bit_depth>(&planeOutputU, &planeSrc0U, &planeSrc1U, &planeSrc2U, targetField, picstruct, stream);
    if (cudaerr != cudaSuccess) {
        return cudaerr;
    }
    cudaerr = run_yadif<TypePixel, bit_depth>(&planeOutputV, &planeSrc0V, &planeSrc1V, &planeSrc2V, targetField, picstruct, stream);
    if (cudaerr != cudaSuccess) {
        return cudaerr;
    }
    return cudaerr;
}

NVEncFilterYadifSource::NVEncFilterYadifSource() : m_nFramesInput(0), m_nFramesOutput(0), m_buf() {

}

NVEncFilterYadifSource::~NVEncFilterYadifSource() {
    clear();
}

void NVEncFilterYadifSource::clear() {
    for (auto& buf : m_buf) {
        buf.clear();
    }
    m_nFramesInput = 0;
    m_nFramesOutput = 0;
}

cudaError_t NVEncFilterYadifSource::alloc(const FrameInfo& frameInfo) {
    if (!cmpFrameInfoCspResolution(&m_buf.begin()->frame, &frameInfo)) {
        //すべて確保されているか確認
        bool allocated = true;
        for (auto& buf : m_buf) {
            if (buf.frame.ptr == nullptr) {
                allocated = false;
                break;
            }
        }
        if (allocated) {
            return cudaSuccess;
        }
    }
    for (auto& buf : m_buf) {
        buf.frame = frameInfo;
        auto ret = buf.alloc();
        if (ret != cudaSuccess) {
            buf.clear();
            return ret;
        }
    }
    return cudaSuccess;
}

cudaError_t NVEncFilterYadifSource::add(const FrameInfo *pInputFrame, cudaStream_t stream) {
    const int iframe = m_nFramesInput++;
    auto pDstFrame = get(iframe);
    copyFrameProp(&pDstFrame->frame, pInputFrame);
    return copyFrameAsync(&pDstFrame->frame, pInputFrame, stream);
}

NVEncFilterYadif::NVEncFilterYadif() : m_nFrame(0), m_pts(0), m_source() {
    m_sFilterName = _T("yadif");
}

NVEncFilterYadif::~NVEncFilterYadif() {
    close();
}

RGY_ERR NVEncFilterYadif::check_param(shared_ptr<NVEncFilterParamYadif> pAfsParam) {
    if (pAfsParam->frameOut.height <= 0 || pAfsParam->frameOut.width <= 0) {
        AddMessage(RGY_LOG_ERROR, _T("Invalid parameter.\n"));
        return RGY_ERR_INVALID_PARAM;
    }
    if (pAfsParam->yadif.mode < VPP_YADIF_MODE_UNKNOWN || pAfsParam->yadif.mode >= VPP_YADIF_MODE_MAX) {
        AddMessage(RGY_LOG_ERROR, _T("Invalid parameter (mode).\n"));
        return RGY_ERR_INVALID_PARAM;
    }
    return RGY_ERR_NONE;
}

RGY_ERR NVEncFilterYadif::init(shared_ptr<NVEncFilterParam> pParam, shared_ptr<RGYLog> pPrintMes) {
    RGY_ERR sts = RGY_ERR_NONE;
    m_pPrintMes = pPrintMes;
    auto prmYadif = std::dynamic_pointer_cast<NVEncFilterParamYadif>(pParam);
    if (!prmYadif) {
        AddMessage(RGY_LOG_ERROR, _T("Invalid parameter type.\n"));
        return RGY_ERR_INVALID_PARAM;
    }
    //パラメータチェック
    if (check_param(prmYadif) != NV_ENC_SUCCESS) {
        return RGY_ERR_INVALID_PARAM;
    }

    auto cudaerr = AllocFrameBuf(prmYadif->frameOut, (prmYadif->yadif.mode & VPP_YADIF_MODE_BOB) ? 2 : 1);
    if (cudaerr != CUDA_SUCCESS) {
        AddMessage(RGY_LOG_ERROR, _T("failed to allocate memory: %s.\n"), char_to_tstring(cudaGetErrorName(cudaerr)).c_str());
        return RGY_ERR_MEMORY_ALLOC;
    }
    prmYadif->frameOut.pitch = m_pFrameBuf[0]->frame.pitch;
    AddMessage(RGY_LOG_DEBUG, _T("allocated output buffer: %dx%pixym1[3], pitch %pixym1[3], %s.\n"),
        m_pFrameBuf[0]->frame.width, m_pFrameBuf[0]->frame.height, m_pFrameBuf[0]->frame.pitch, RGY_CSP_NAMES[m_pFrameBuf[0]->frame.csp]);

    cudaerr = m_source.alloc(prmYadif->frameOut);
    if (cudaerr != CUDA_SUCCESS) {
        AddMessage(RGY_LOG_ERROR, _T("failed to allocate memory: %s.\n"), char_to_tstring(cudaGetErrorName(cudaerr)).c_str());
        return RGY_ERR_MEMORY_ALLOC;
    }

    prmYadif->frameOut.picstruct = RGY_PICSTRUCT_FRAME;
    m_nFrame = 0;
    m_pts = 0;
    m_nPathThrough &= (~(FILTER_PATHTHROUGH_PICSTRUCT | FILTER_PATHTHROUGH_FLAGS | FILTER_PATHTHROUGH_TIMESTAMP));
    if (prmYadif->yadif.mode & VPP_YADIF_MODE_BOB) {
        prmYadif->baseFps *= 2;
    }

    setFilterInfo(pParam->print());
    m_pParam = pParam;
    return sts;
}

tstring NVEncFilterParamYadif::print() const {
    return yadif.print();
}

RGY_ERR NVEncFilterYadif::run_filter(const FrameInfo *pInputFrame, FrameInfo **ppOutputFrames, int *pOutputFrameNum, cudaStream_t stream) {
    RGY_ERR sts = RGY_ERR_NONE;

    auto prmYadif = std::dynamic_pointer_cast<NVEncFilterParamYadif>(m_pParam);
    if (!prmYadif) {
        AddMessage(RGY_LOG_ERROR, _T("Invalid parameter type.\n"));
        return RGY_ERR_INVALID_PARAM;
    }

    const int iframe = m_source.inframe();
    if (pInputFrame->ptr == nullptr && m_nFrame >= iframe) {
        //終了
        *pOutputFrameNum = 0;
        ppOutputFrames[0] = nullptr;
        return sts;
    } else if (pInputFrame->ptr != nullptr) {
        //エラーチェック
        const auto memcpyKind = getCudaMemcpyKind(pInputFrame->deivce_mem, m_pFrameBuf[0]->frame.deivce_mem);
        if (memcpyKind != cudaMemcpyDeviceToDevice) {
            AddMessage(RGY_LOG_ERROR, _T("only supported on device memory.\n"));
            return RGY_ERR_INVALID_CALL;
        }
        if (m_pParam->frameOut.csp != m_pParam->frameIn.csp) {
            AddMessage(RGY_LOG_ERROR, _T("csp does not match.\n"));
            return RGY_ERR_INVALID_PARAM;
        }
        //sourceキャッシュにコピー
        auto cudaerr = m_source.add(pInputFrame, stream);
        if (cudaerr != cudaSuccess) {
            AddMessage(RGY_LOG_ERROR, _T("failed to add frame to source buffer: %s.\n"), char_to_tstring(cudaGetErrorName(cudaerr)).c_str());
            return RGY_ERR_CUDA;
        }
    }

    //十分な数のフレームがたまった、あるいはdrainモードならフレームを出力
    if (iframe >= 1 || pInputFrame == nullptr) {
        //出力先のフレーム
        CUFrameBuf *pOutFrame = nullptr;
        *pOutputFrameNum = 1;
        if (ppOutputFrames[0] == nullptr) {
            pOutFrame = m_pFrameBuf[m_nFrameIdx].get();
            ppOutputFrames[0] = &pOutFrame->frame;
            ppOutputFrames[0]->picstruct = pInputFrame->picstruct;
            m_nFrameIdx = (m_nFrameIdx + 1) % m_pFrameBuf.size();
            if (prmYadif->yadif.mode & VPP_YADIF_MODE_BOB) {
                pOutFrame = m_pFrameBuf[m_nFrameIdx].get();
                ppOutputFrames[1] = &pOutFrame->frame;
                ppOutputFrames[1]->picstruct = pInputFrame->picstruct;
                m_nFrameIdx = (m_nFrameIdx + 1) % m_pFrameBuf.size();
                *pOutputFrameNum = 2;
            }
        }

        const auto *const pSourceFrame = &m_source.get(m_nFrame)->frame;
        pOutFrame->frame.flags = pSourceFrame->flags & (~(RGY_FRAME_FLAG_RFF | RGY_FRAME_FLAG_RFF_COPY | RGY_FRAME_FLAG_RFF_BFF | RGY_FRAME_FLAG_RFF_TFF));

        YadifTargetField targetField = YADIF_GEN_FIELD_UNKNOWN;
        if (prmYadif->yadif.mode & VPP_YADIF_MODE_AUTO) {
            //エラーチェック
            const auto memcpyKind = getCudaMemcpyKind(pSourceFrame->deivce_mem, ppOutputFrames[0]->deivce_mem);
            if (memcpyKind != cudaMemcpyDeviceToDevice) {
                AddMessage(RGY_LOG_ERROR, _T("only supported on device memory.\n"));
                return RGY_ERR_INVALID_CALL;
            }
            if ((pSourceFrame->picstruct & RGY_PICSTRUCT_INTERLACED) == 0) {
                ppOutputFrames[0]->picstruct = RGY_PICSTRUCT_FRAME;
                ppOutputFrames[0]->timestamp = pSourceFrame->timestamp;
                copyFrameAsync(ppOutputFrames[0], pSourceFrame, stream);
                if (prmYadif->yadif.mode & VPP_YADIF_MODE_BOB) {
                    ppOutputFrames[1]->picstruct = RGY_PICSTRUCT_FRAME;
                    ppOutputFrames[0]->timestamp = pSourceFrame->timestamp;
                    ppOutputFrames[0]->duration = (pSourceFrame->duration + 1) / 2;
                    ppOutputFrames[1]->timestamp = ppOutputFrames[0]->timestamp + ppOutputFrames[0]->duration;
                    ppOutputFrames[1]->duration = pSourceFrame->duration - ppOutputFrames[0]->duration;
                    ppOutputFrames[1]->inputFrameId = pInputFrame->inputFrameId;
                    copyFrameAsync(ppOutputFrames[1], pSourceFrame, stream);
                }
                m_nFrame++;
                return RGY_ERR_NONE;
            } else if ((pSourceFrame->picstruct & RGY_PICSTRUCT_FRAME_TFF) == RGY_PICSTRUCT_FRAME_TFF) {
                targetField = YADIF_GEN_FIELD_BOTTOM;
            } else if ((pSourceFrame->picstruct & RGY_PICSTRUCT_FRAME_BFF) == RGY_PICSTRUCT_FRAME_BFF) {
                targetField = YADIF_GEN_FIELD_TOP;
            }
            AddMessage(RGY_LOG_ERROR, _T("picstruct: %d, %s.\n"), targetField, picstrcut_to_str(pSourceFrame->picstruct));
        } else if (prmYadif->yadif.mode & VPP_YADIF_MODE_TFF) {
            targetField = YADIF_GEN_FIELD_BOTTOM;
        } else if (prmYadif->yadif.mode & VPP_YADIF_MODE_BFF) {
            targetField = YADIF_GEN_FIELD_TOP;
        } else {
            AddMessage(RGY_LOG_ERROR, _T("Not implemented yet.\n"));
            return RGY_ERR_INVALID_PARAM;
        }

        static const std::map<RGY_CSP, decltype(run_yadif_frame<uint8_t, 8>)*> func_list = {
            { RGY_CSP_YV12,      run_yadif_frame<uint8_t,   8> },
            { RGY_CSP_YV12_16,   run_yadif_frame<uint16_t, 16> },
            { RGY_CSP_YUV444,    run_yadif_frame<uint8_t,   8> },
            { RGY_CSP_YUV444_16, run_yadif_frame<uint16_t, 16> }
        };
        if (func_list.count(pSourceFrame->csp) == 0) {
            AddMessage(RGY_LOG_ERROR, _T("unsupported csp %s.\n"), RGY_CSP_NAMES[pSourceFrame->csp]);
            return RGY_ERR_UNSUPPORTED;
        }
        func_list.at(pSourceFrame->csp)(ppOutputFrames[0],
            &m_source.get(m_nFrame-1)->frame,
            &m_source.get(m_nFrame+0)->frame,
            &m_source.get(m_nFrame+1)->frame,
            targetField,
            pSourceFrame->picstruct,
            stream
            );

        ppOutputFrames[0]->picstruct = RGY_PICSTRUCT_FRAME;
        ppOutputFrames[0]->timestamp = pSourceFrame->timestamp;
        if (prmYadif->yadif.mode & VPP_YADIF_MODE_BOB) {
            targetField = (targetField == YADIF_GEN_FIELD_BOTTOM) ? YADIF_GEN_FIELD_TOP : YADIF_GEN_FIELD_BOTTOM;

            if (func_list.count(pSourceFrame->csp) == 0) {
                AddMessage(RGY_LOG_ERROR, _T("unsupported csp %s.\n"), RGY_CSP_NAMES[pSourceFrame->csp]);
                return RGY_ERR_UNSUPPORTED;
            }
            func_list.at(pSourceFrame->csp)(ppOutputFrames[1],
                &m_source.get(m_nFrame-1)->frame,
                &m_source.get(m_nFrame+0)->frame,
                &m_source.get(m_nFrame+1)->frame,
                targetField,
                pSourceFrame->picstruct,
                stream
                );
            ppOutputFrames[1]->picstruct = RGY_PICSTRUCT_FRAME;
            ppOutputFrames[0]->timestamp = pSourceFrame->timestamp;
            ppOutputFrames[0]->duration = (pSourceFrame->duration + 1) / 2;
            ppOutputFrames[1]->timestamp = ppOutputFrames[0]->timestamp + ppOutputFrames[0]->duration;
            ppOutputFrames[1]->duration = pSourceFrame->duration - ppOutputFrames[0]->duration;
            ppOutputFrames[1]->inputFrameId = pInputFrame->inputFrameId;
        }
        m_nFrame++;
    } else {
        //出力フレームなし
        *pOutputFrameNum = 0;
        ppOutputFrames[0] = nullptr;
    }
    return sts;
}

void NVEncFilterYadif::close() {
    m_nFrame = 0;
    m_pts = 0;
    AddMessage(RGY_LOG_DEBUG, _T("closed yadif filter.\n"));
}
