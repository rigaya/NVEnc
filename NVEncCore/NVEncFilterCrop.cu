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
#include <cstdint>
#include "NVEncFilter.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

union NV_ENC_CSP_2 {
    struct {
        NV_ENC_CSP a, b;
    } csp;
    uint64_t i;

    NV_ENC_CSP_2() {

    };
    NV_ENC_CSP_2(NV_ENC_CSP _a, NV_ENC_CSP _b) {
        csp.a = _a;
        csp.b = _b;
    };
};

template<typename TypeOut, int out_bit_depth, typename TypeIn, int in_bit_depth>
__global__ void kernel_crop_nv12_nv12(TypeOut *__restrict__ pDst, const int dstPitch, const int dstWidth, const int dstHeight,
    const TypeIn *__restrict__ pSrc, const int srcPitch, const int srcHeight, const int offsetX, const int offsetY) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x < dstWidth && y < dstHeight) {
        //Y
        int idst = y * dstPitch + x;
        int isrc = (y + offsetY) * srcPitch + x + offsetX;
        pDst[idst] = pSrc[isrc];
        if (y < (dstHeight >> 1)) {
            //UV
            idst += dstHeight * dstPitch;
            isrc += (srcHeight - (offsetY >> 1)) * srcPitch;
            pDst[idst] = pSrc[isrc];
        }
    }
}

template<typename TypeOut, int out_bit_depth, typename TypeIn, int in_bit_depth>
__global__ void kernel_crop_uv_nv12_yv12(uint8_t *__restrict__ pDstU, uint8_t *__restrict__ pDstV,
    const int dstPitch, const int dstWidth, const int dstHeight,
    const uint8_t *__restrict__ pSrc, const int srcPitch, const int offsetX, const int offsetY) {
    int uv_x = blockIdx.x * blockDim.x + threadIdx.x;
    int uv_y = blockIdx.y * blockDim.y + threadIdx.y;
    if (uv_x < (dstWidth >> 1) && uv_y < (dstHeight >> 1)) {
        int idst = uv_y * dstPitch + uv_x * sizeof(TypeOut); //YV12
        int isrc = (uv_y + (offsetY >> 1)) * srcPitch + ((uv_x << 1) + offsetX) * sizeof(TypeIn); //NV12
        const TypeIn *ptr_src = (const TypeIn *)(pSrc  + isrc);
        TypeOut *ptr_dst_u = (TypeOut *)(pDstU + idst);
        TypeOut *ptr_dst_v = (TypeOut *)(pDstV + idst);
        if (out_bit_depth == in_bit_depth) {
            ptr_dst_u[0] = ptr_src[0];
            ptr_dst_v[0] = ptr_src[1];
        } else if (out_bit_depth > in_bit_depth) {
            ptr_dst_u[0] = ptr_src[0] << (out_bit_depth - in_bit_depth);
            ptr_dst_v[0] = ptr_src[1] << (out_bit_depth - in_bit_depth);
        } else {
            ptr_dst_u[0] = ptr_src[0] >> (in_bit_depth - out_bit_depth);
            ptr_dst_v[0] = ptr_src[1] >> (in_bit_depth - out_bit_depth);
        }
    }
}

template<typename TypeOut, int out_bit_depth, typename TypeIn, int in_bit_depth>
void crop_uv_nv12_yv12(FrameInfo *pOutputFrame, const FrameInfo *pInputFrame, const sInputCrop *pCrop) {
    dim3 blockSize(32, 4);
    dim3 gridSize(divCeil(pOutputFrame->width >> 1, blockSize.x), divCeil(pOutputFrame->height >> 1, blockSize.y));
    uint8_t *ptrU = (uint8_t *)pOutputFrame->ptr + pOutputFrame->pitch * pOutputFrame->height;
    uint8_t *ptrV = (uint8_t *)pOutputFrame->ptr + pOutputFrame->pitch * pOutputFrame->height * 3 / 2;
    const uint8_t *ptrC = (const uint8_t  *)pInputFrame->ptr + pInputFrame->pitch  * pInputFrame->height;
    kernel_crop_uv_nv12_yv12<TypeOut, out_bit_depth, TypeIn, in_bit_depth><<<gridSize, blockSize>>>(
        ptrU, ptrV, pOutputFrame->pitch, pOutputFrame->width, pOutputFrame->height,
        ptrC, pInputFrame->pitch, pCrop->e.left, pCrop->e.up);
}

template<typename TypeOut, int out_bit_depth, typename TypeIn, int in_bit_depth>
__global__ void kernel_crop_uv_yv12_nv12(uint8_t *__restrict__ pDst, const int dstPitch, const int dstWidth, const int dstHeight,
    const uint8_t *__restrict__ pSrcU, const uint8_t *__restrict__ pSrcV, const int srcPitch, const int offsetX, const int offsetY) {
    int uv_x = blockIdx.x * blockDim.x + threadIdx.x;
    int uv_y = blockIdx.y * blockDim.y + threadIdx.y;
    if (uv_x < (dstWidth >> 1) && uv_y < (dstHeight >> 1)) {
        int idst = uv_y * dstPitch + (uv_x << 1) * sizeof(TypeOut); //NV12
        int isrc = (uv_y + (offsetY >> 1)) * srcPitch + (uv_x + (offsetX >> 1)) * sizeof(TypeIn); //YV12
        const TypeIn *ptr_src_u = (const TypeIn *)(pSrcU + isrc);
        const TypeIn *ptr_src_v = (const TypeIn *)(pSrcV + isrc);
        TypeOut *ptr_dst = (TypeOut *)(pDst + idst);
        if (out_bit_depth == in_bit_depth) {
            ptr_dst[0] = ptr_src_u[0];
            ptr_dst[1] = ptr_src_v[0];
        } else if (out_bit_depth > in_bit_depth) {
            ptr_dst[0] = ptr_src_u[0] << (out_bit_depth - in_bit_depth);
            ptr_dst[1] = ptr_src_v[0] << (out_bit_depth - in_bit_depth);
        } else {
            ptr_dst[0] = ptr_src_u[0] >> (in_bit_depth - out_bit_depth);
            ptr_dst[1] = ptr_src_v[0] >> (in_bit_depth - out_bit_depth);
        }
    }
}

template<typename TypeOut, int out_bit_depth, typename TypeIn, int in_bit_depth>
void crop_uv_yv12_nv12(FrameInfo *pOutputFrame, const FrameInfo *pInputFrame, const sInputCrop *pCrop) {
    dim3 blockSize(32, 4);
    dim3 gridSize(divCeil(pOutputFrame->width >> 1, blockSize.x), divCeil(pOutputFrame->height >> 1, blockSize.y));
    uint8_t *ptrC = (uint8_t *)pOutputFrame->ptr + pOutputFrame->pitch * pOutputFrame->height;
    const uint8_t *ptrU = (const uint8_t *)pInputFrame->ptr + pInputFrame->pitch * pInputFrame->height;
    const uint8_t *ptrV = (const uint8_t *)pInputFrame->ptr + pInputFrame->pitch * pInputFrame->height * 3 / 2;
    kernel_crop_uv_yv12_nv12<TypeOut, out_bit_depth, TypeIn, in_bit_depth><<<gridSize, blockSize>>>(
        ptrC, pOutputFrame->pitch, pOutputFrame->width, pOutputFrame->height,
        ptrU, ptrV, pInputFrame->pitch, pCrop->e.left, pCrop->e.up);
}

NVENCSTATUS NVEncFilterCspCrop::convertCspFromNV12(FrameInfo *pOutputFrame, const FrameInfo *pInputFrame) {
    const auto frameOutInfoEx = getFrameInfoExtra(&m_filterParam.frameOut);
    //Y
    auto cudaerr = cudaMemcpy2D((uint8_t *)pOutputFrame->ptr, pOutputFrame->pitch,
        (uint8_t *)pInputFrame->ptr + m_filterParam.crop.e.left + m_filterParam.crop.e.up * pInputFrame->pitch,
        pInputFrame->pitch,
        frameOutInfoEx.width_byte, m_filterParam.frameOut.height, cudaMemcpyDeviceToDevice);
    if (cudaerr != cudaSuccess) {
        AddMessage(NV_LOG_ERROR, _T("error at cudaMemcpy2D (convertCspFromNV12(%s -> %s)): %s.\n"),
            NV_ENC_CSP_NAMES[pInputFrame->csp], NV_ENC_CSP_NAMES[pOutputFrame->csp], char_to_tstring(cudaGetErrorString(cudaerr)).c_str());
        return NV_ENC_ERR_INVALID_CALL;
    }

    //UV
    std::map<uint64_t, void (*)(FrameInfo *pOutputFrame, const FrameInfo *pInputFrame, const sInputCrop *pCrop)> crop_uv_nv12_yv12_list ={
        { NV_ENC_CSP_2(NV_ENC_CSP_NV12, NV_ENC_CSP_YV12   ).i, crop_uv_nv12_yv12<uint8_t,   8, uint8_t,   8> },
        { NV_ENC_CSP_2(NV_ENC_CSP_NV12, NV_ENC_CSP_YV12_16).i, crop_uv_nv12_yv12<uint16_t, 16, uint8_t,   8> },
        { NV_ENC_CSP_2(NV_ENC_CSP_NV12, NV_ENC_CSP_YV12_14).i, crop_uv_nv12_yv12<uint16_t, 14, uint8_t,   8> },
        { NV_ENC_CSP_2(NV_ENC_CSP_NV12, NV_ENC_CSP_YV12_12).i, crop_uv_nv12_yv12<uint16_t, 12, uint8_t,   8> },
        { NV_ENC_CSP_2(NV_ENC_CSP_NV12, NV_ENC_CSP_YV12_10).i, crop_uv_nv12_yv12<uint16_t, 10, uint8_t,   8> },
        { NV_ENC_CSP_2(NV_ENC_CSP_NV12, NV_ENC_CSP_YV12_09).i, crop_uv_nv12_yv12<uint16_t,  9, uint8_t,   8> },
        { NV_ENC_CSP_2(NV_ENC_CSP_P010, NV_ENC_CSP_YV12_16).i, crop_uv_nv12_yv12<uint16_t, 16, uint16_t, 16> },
        { NV_ENC_CSP_2(NV_ENC_CSP_P010, NV_ENC_CSP_YV12_14).i, crop_uv_nv12_yv12<uint16_t, 14, uint16_t, 16> },
        { NV_ENC_CSP_2(NV_ENC_CSP_P010, NV_ENC_CSP_YV12_12).i, crop_uv_nv12_yv12<uint16_t, 12, uint16_t, 16> },
        { NV_ENC_CSP_2(NV_ENC_CSP_P010, NV_ENC_CSP_YV12_10).i, crop_uv_nv12_yv12<uint16_t, 10, uint16_t, 16> },
        { NV_ENC_CSP_2(NV_ENC_CSP_P010, NV_ENC_CSP_YV12_09).i, crop_uv_nv12_yv12<uint16_t,  9, uint16_t, 16> },
    };
    const auto cspconv = NV_ENC_CSP_2(pInputFrame->csp, pOutputFrame->csp);
    if (crop_uv_nv12_yv12_list.count(cspconv.i) == 0) {
        AddMessage(NV_LOG_ERROR, _T("unsupported csp conversion: %s -> %s.\n"), NV_ENC_CSP_NAMES[pInputFrame->csp], NV_ENC_CSP_NAMES[pOutputFrame->csp]);
        return NV_ENC_ERR_UNIMPLEMENTED;
    }
    crop_uv_nv12_yv12_list[cspconv.i](pOutputFrame, pInputFrame, &m_filterParam.crop);
    if (cudaSuccess != (cudaerr = cudaGetLastError())) {
        AddMessage(NV_LOG_ERROR, _T("error at crop_uv_nv12_yv12_list(%s -> %s): %s.\n"),
            NV_ENC_CSP_NAMES[pInputFrame->csp], NV_ENC_CSP_NAMES[pOutputFrame->csp],
            char_to_tstring(cudaGetErrorString(cudaerr)).c_str());
        return NV_ENC_ERR_INVALID_CALL;
    }
    return NV_ENC_SUCCESS;
}
NVENCSTATUS NVEncFilterCspCrop::convertCspFromYV12(FrameInfo *pOutputFrame, const FrameInfo *pInputFrame) {
    const auto frameOutInfoEx = getFrameInfoExtra(&m_filterParam.frameOut);
    //Y
    auto cudaerr = cudaMemcpy2D((uint8_t *)pOutputFrame->ptr, pOutputFrame->pitch,
        (uint8_t *)pInputFrame->ptr + m_filterParam.crop.e.left + m_filterParam.crop.e.up * pInputFrame->pitch,
        pInputFrame->pitch,
        frameOutInfoEx.width_byte, m_filterParam.frameOut.height, cudaMemcpyDeviceToDevice);
    if (cudaerr != cudaSuccess) {
        AddMessage(NV_LOG_ERROR, _T("error at cudaMemcpy2D (convertCspFromYV12(%s -> %s)): %s.\n"),
            NV_ENC_CSP_NAMES[pInputFrame->csp], NV_ENC_CSP_NAMES[pOutputFrame->csp], char_to_tstring(cudaGetErrorString(cudaerr)).c_str());
        return NV_ENC_ERR_INVALID_CALL;
    }

    //UV
    std::map<uint64_t, void (*)(FrameInfo *pOutputFrame, const FrameInfo *pInputFrame, const sInputCrop *pCrop)> crop_uv_yv12_nv12_list = {
        { NV_ENC_CSP_2(NV_ENC_CSP_YV12,    NV_ENC_CSP_NV12).i, crop_uv_yv12_nv12<uint8_t,   8, uint8_t,   8> },
        { NV_ENC_CSP_2(NV_ENC_CSP_YV12_16, NV_ENC_CSP_P010).i, crop_uv_yv12_nv12<uint16_t, 16, uint16_t, 16> },
        { NV_ENC_CSP_2(NV_ENC_CSP_YV12_14, NV_ENC_CSP_P010).i, crop_uv_yv12_nv12<uint16_t, 14, uint16_t, 16> },
        { NV_ENC_CSP_2(NV_ENC_CSP_YV12_12, NV_ENC_CSP_P010).i, crop_uv_yv12_nv12<uint16_t, 12, uint16_t, 16> },
        { NV_ENC_CSP_2(NV_ENC_CSP_YV12_10, NV_ENC_CSP_P010).i, crop_uv_yv12_nv12<uint16_t, 10, uint16_t, 16> },
        { NV_ENC_CSP_2(NV_ENC_CSP_YV12_09, NV_ENC_CSP_P010).i, crop_uv_yv12_nv12<uint16_t,  9, uint16_t, 16> },
    };
    const auto cspconv = NV_ENC_CSP_2(pInputFrame->csp, pOutputFrame->csp);
    if (crop_uv_yv12_nv12_list.count(cspconv.i) == 0) {
        AddMessage(NV_LOG_ERROR, _T("unsupported csp conversion: %s -> %s.\n"), NV_ENC_CSP_NAMES[pInputFrame->csp], NV_ENC_CSP_NAMES[pOutputFrame->csp]);
        return NV_ENC_ERR_UNIMPLEMENTED;
    }
    crop_uv_yv12_nv12_list[cspconv.i](pOutputFrame, pInputFrame, &m_filterParam.crop);
    if (cudaSuccess != (cudaerr = cudaGetLastError())) {
        AddMessage(NV_LOG_ERROR, _T("error at crop_uv_nv12_yv12_list(%s -> %s): %s.\n"),
            NV_ENC_CSP_NAMES[pInputFrame->csp], NV_ENC_CSP_NAMES[pOutputFrame->csp],
            char_to_tstring(cudaGetErrorString(cudaerr)).c_str());
        return NV_ENC_ERR_INVALID_CALL;
    }
    return NV_ENC_SUCCESS;

}
NVENCSTATUS NVEncFilterCspCrop::convertCspFromYUV444(FrameInfo *pOutputFrame, const FrameInfo *pInputFrame) {
    AddMessage(NV_LOG_ERROR, _T("unsupported csp conversion: %s -> %s.\n"), NV_ENC_CSP_NAMES[pInputFrame->csp], NV_ENC_CSP_NAMES[pOutputFrame->csp]);
    return NV_ENC_ERR_UNIMPLEMENTED;
}

NVEncFilterCspCrop::NVEncFilterCspCrop() {
    m_sFilterName = _T("copy/cspconv/crop");
}

NVEncFilterCspCrop::~NVEncFilterCspCrop() {
    close();
}

NVENCSTATUS NVEncFilterCspCrop::init(shared_ptr<NVEncFilterParam> pParam, shared_ptr<CNVEncLog> pPrintMes) {
    NVENCSTATUS sts = NV_ENC_SUCCESS;
    m_pPrintMes = pPrintMes;
    auto pCropParam = std::dynamic_pointer_cast<NVEncFilterParamCrop>(pParam);
    if (!pCropParam) {
        AddMessage(NV_LOG_ERROR, _T("Invalid parameter type.\n"));
        return NV_ENC_ERR_INVALID_PARAM;
    }
    //フィルタ名の調整
    m_sFilterName = _T("");
    if (cropEnabled(pCropParam->crop)) {
        m_sFilterName += _T("crop");
    }
    if (pCropParam->frameOut.csp != pCropParam->frameIn.csp) {
        m_sFilterName += (m_sFilterName.length()) ? _T("/cspconv") : _T("cspconv");
    }
    if (m_sFilterName.length() == 0) {
        m_sFilterName += _T("copy");
    }
    //パラメータチェック
    for (int i = 0; i < _countof(pCropParam->crop.c); i++) {
        if ((pCropParam->crop.c[i] & 1) != 0) {
            AddMessage(NV_LOG_ERROR, _T("crop should be divided by 2.\n"));
            return NV_ENC_ERR_INVALID_PARAM;
        }
    }
    pCropParam->frameOut.height = pCropParam->frameIn.height - pCropParam->crop.e.bottom - pCropParam->crop.e.up;
    pCropParam->frameOut.width = pCropParam->frameIn.width - pCropParam->crop.e.left - pCropParam->crop.e.right;
    if (pCropParam->frameOut.height <= 0 || pCropParam->frameOut.width <= 0) {
        AddMessage(NV_LOG_ERROR, _T("crop size is too big.\n"));
        return NV_ENC_ERR_INVALID_PARAM;
    }

    if (CUDA_SUCCESS != AllocFrameBuf(pCropParam->frameOut, 2)) {
        AddMessage(NV_LOG_ERROR, _T("failed to allocate memory.\n"));
        return NV_ENC_ERR_OUT_OF_MEMORY;
    }
    pCropParam->frameOut.pitch = m_pFrameBuf[0]->frame.pitch;

    //フィルタ情報の調整
    m_sFilterInfo = _T("");
    if (cropEnabled(pCropParam->crop)) {
        m_sFilterInfo += strsprintf(_T("crop: %d,%d,%d,%d"), pCropParam->crop.e.left, pCropParam->crop.e.up, pCropParam->crop.e.right, pCropParam->crop.e.bottom);
    }
    if (pCropParam->frameOut.csp != pCropParam->frameIn.csp) {
        m_sFilterInfo += (m_sFilterInfo.length()) ? _T("/cspconv") : _T("cspconv");
        m_sFilterInfo += strsprintf(_T("(%s -> %s)"), NV_ENC_CSP_NAMES[pCropParam->frameIn.csp], NV_ENC_CSP_NAMES[pCropParam->frameOut.csp]);
    }
    if (m_sFilterInfo.length() == 0) {
        m_sFilterInfo += getCudaMemcpyKindStr(pCropParam->frameIn.deivce_mem, pCropParam->frameOut.deivce_mem);
    }

    //コピーを保存
    m_filterParam = *(pCropParam.get());
    return sts;
}

NVENCSTATUS NVEncFilterCspCrop::filter(const FrameInfo *pInputFrame, FrameInfo **ppOutputFrames, int *pOutputFrameNum) {
    NVENCSTATUS sts = NV_ENC_SUCCESS;

    *pOutputFrameNum = 1;
    if (ppOutputFrames[0] == nullptr) {
        auto pOutFrame = m_pFrameBuf[m_nFrameIdx].get();
        ppOutputFrames[0] = &pOutFrame->frame;
        m_nFrameIdx = (m_nFrameIdx + 1) % m_pFrameBuf.size();
    }
    const auto memcpyKind = getCudaMemcpyKind(pInputFrame->deivce_mem, ppOutputFrames[0]->deivce_mem);
    ppOutputFrames[0]->interlaced = pInputFrame->interlaced;
    if (m_filterParam.frameOut.csp == m_filterParam.frameIn.csp) {
        auto cudaMemcpyErrMes = [&](cudaError_t cudaerr, const TCHAR *mes) {
            AddMessage(NV_LOG_ERROR, _T("error at %s (filter(%s)): %s.\n"),
                mes, NV_ENC_CSP_NAMES[pInputFrame->csp], char_to_tstring(cudaGetErrorString(cudaerr)).c_str());
            return NV_ENC_ERR_INVALID_CALL;
        };
#if 1
        const auto frameOutInfoEx = getFrameInfoExtra(&m_filterParam.frameOut);
        if (!cropEnabled(m_filterParam.crop)) {
            //cropがなければ、一度に転送可能
            auto cudaerr = cudaMemcpy2D((uint8_t *)ppOutputFrames[0]->ptr, ppOutputFrames[0]->pitch,
                (uint8_t *)pInputFrame->ptr, pInputFrame->pitch,
                frameOutInfoEx.width_byte, frameOutInfoEx.height_total, memcpyKind);
            if (cudaerr != cudaSuccess) {
                cudaMemcpyErrMes(cudaerr, _T("cudaMemcpy2DAll"));
                return NV_ENC_ERR_INVALID_CALL;
            };
        } else {
            if (m_filterParam.frameOut.csp == NV_ENC_CSP_NV12) {
                cudaError_t cudaerr;
                //Y
                cudaerr = cudaMemcpy2D((uint8_t *)ppOutputFrames[0]->ptr, ppOutputFrames[0]->pitch,
                    (uint8_t *)pInputFrame->ptr + m_filterParam.crop.e.left + m_filterParam.crop.e.up * pInputFrame->pitch,
                    pInputFrame->pitch,
                    frameOutInfoEx.width_byte, m_filterParam.frameOut.height, memcpyKind);
                if (cudaerr != cudaSuccess) {
                    cudaMemcpyErrMes(cudaerr, _T("cudaMemcpy2D_Y"));
                    return NV_ENC_ERR_INVALID_CALL;
                };
                //UV
                cudaerr = cudaMemcpy2D((uint8_t *)ppOutputFrames[0]->ptr + ppOutputFrames[0]->pitch * m_filterParam.frameOut.height, ppOutputFrames[0]->pitch,
                    (uint8_t *)pInputFrame->ptr
                    + pInputFrame->height * pInputFrame->pitch
                    + m_filterParam.crop.e.left + (m_filterParam.crop.e.up >> 1) * pInputFrame->pitch,
                    pInputFrame->pitch,
                    frameOutInfoEx.width_byte, m_filterParam.frameOut.height >> 1, memcpyKind);
                if (cudaerr != cudaSuccess) {
                    cudaMemcpyErrMes(cudaerr, _T("cudaMemcpy2D_UV"));
                    return NV_ENC_ERR_INVALID_CALL;
                };
            } else {
                AddMessage(NV_LOG_ERROR, _T("unsupported output csp with crop.\n"));
                return NV_ENC_ERR_UNIMPLEMENTED;
            }
        }
#else
        if (m_filterParam.frameOut.csp == NV_ENC_CSP_NV12) {
            dim3 blockSize(32, 4);
            dim3 gridSize(divCeil(m_filterParam.frameOut.width, blockSize.x), divCeil(m_filterParam.frameOut.height, blockSize.y));
            kernel_crop_nv12_nv12<uint8_t><<<gridSize, blockSize>>>((uint8_t *)ppOutputFrames[0]->ptr, (uint8_t *)pInputFrame->ptr, pInputFrame->pitch);
        } else {
            AddMessage(NV_LOG_ERROR, _T("unsupported output csp.\n"));
            return NV_ENC_ERR_UNSUPPORTED_PARAM;
        }
#endif
    } else if (memcpyKind != cudaMemcpyDeviceToDevice) {
        AddMessage(NV_LOG_ERROR, _T("converting csp while copying from host to device is not supported.\n"));
        return NV_ENC_ERR_UNSUPPORTED_PARAM;
    } else {
        //色空間変換
        const auto supportedCspNV12   = make_array<NV_ENC_CSP>(NV_ENC_CSP_NV12, NV_ENC_CSP_P010);
        const auto supportedCspYV12   = make_array<NV_ENC_CSP>(NV_ENC_CSP_YV12, NV_ENC_CSP_YV12_09, NV_ENC_CSP_YV12_10, NV_ENC_CSP_YV12_12, NV_ENC_CSP_YV12_14, NV_ENC_CSP_YV12_16);
        const auto supportedCspYUV444 = make_array<NV_ENC_CSP>(NV_ENC_CSP_YUV444, NV_ENC_CSP_YUV444_09, NV_ENC_CSP_YUV444_10, NV_ENC_CSP_YUV444_12, NV_ENC_CSP_YUV444_14, NV_ENC_CSP_YUV444_16);
        if (std::find(supportedCspNV12.begin(), supportedCspNV12.end(), m_filterParam.frameIn.csp) != supportedCspNV12.end()) {
            sts = convertCspFromNV12(ppOutputFrames[0], pInputFrame);
        } else if (std::find(supportedCspYV12.begin(), supportedCspYV12.end(), m_filterParam.frameIn.csp) != supportedCspYV12.end()) {
            sts = convertCspFromYV12(ppOutputFrames[0], pInputFrame);
        } else if (std::find(supportedCspYUV444.begin(), supportedCspYUV444.end(), m_filterParam.frameIn.csp) != supportedCspYUV444.end()) {
            sts = convertCspFromYUV444(ppOutputFrames[0], pInputFrame);
        } else {
            AddMessage(NV_LOG_ERROR, _T("converting csp from %s is not supported.\n"), NV_ENC_CSP_NAMES[m_filterParam.frameIn.csp]);
            sts = NV_ENC_ERR_UNIMPLEMENTED;
        }
    }
    return sts;
}

void NVEncFilterCspCrop::close() {
    m_pFrameBuf.clear();
}
