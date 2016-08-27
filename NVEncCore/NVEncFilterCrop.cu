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

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

__constant__ int g_nInWidth;
__constant__ int g_nInHeight;
__constant__ int g_nOutWidth;
__constant__ int g_nOutHeight;
__constant__ int g_nOffsetX;
__constant__ int g_nOffsetY;
__constant__ int g_nOutPitch;

template<typename T>
__global__ void kernel_crop_nv12_nv12(T *__restrict__ dst, const T *__restrict__ src, const int srcPitch) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x < g_nOutWidth && y < g_nOutHeight) {
        //Y
        int idst = y * g_nOutPitch + x;
        int isrc = (y + g_nOffsetY) * srcPitch + x + g_nOffsetX;
        dst[idst] = src[isrc];
        if (y < (g_nOutHeight >> 1)) {
            //UV
            idst += g_nOutHeight * g_nOutPitch;
            isrc += (g_nInHeight - (g_nOffsetY >> 1)) * srcPitch;
            dst[idst] = src[isrc];
        }
    }
}

#include "NVEncFilter.h"

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
    if (pCropParam->frameOut.csp == pCropParam->frameIn.csp) {
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

    cudaMemcpyToSymbol(g_nInWidth,   &pCropParam->frameIn.width,   sizeof(g_nInWidth));
    cudaMemcpyToSymbol(g_nInHeight,  &pCropParam->frameIn.height,  sizeof(g_nInHeight));
    cudaMemcpyToSymbol(g_nOutWidth,  &pCropParam->frameOut.width,  sizeof(g_nOutWidth));
    cudaMemcpyToSymbol(g_nOutHeight, &pCropParam->frameOut.height, sizeof(g_nOutHeight));
    cudaMemcpyToSymbol(g_nOffsetX,   &pCropParam->crop.e.up,       sizeof(g_nOffsetX));
    cudaMemcpyToSymbol(g_nOffsetY,   &pCropParam->crop.e.left,     sizeof(g_nOffsetY));
    cudaMemcpyToSymbol(g_nOutPitch,  &pCropParam->frameOut.pitch,  sizeof(g_nOutPitch));

    m_sFilterInfo = strsprintf(_T("crop: %d,%d,%d,%d"), pCropParam->crop.e.left, pCropParam->crop.e.up, pCropParam->crop.e.right, pCropParam->crop.e.bottom);

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
    if (m_filterParam.frameOut.csp == m_filterParam.frameIn.csp) {
#if 1
        const auto frameOutInfoEx = getFrameInfoExtra(&m_filterParam.frameOut);
        if (!cropEnabled(m_filterParam.crop)) {
            //cropがなければ、一度に転送可能
            cudaMemcpy2D((uint8_t *)ppOutputFrames[0]->ptr, ppOutputFrames[0]->pitch,
                (uint8_t *)pInputFrame->ptr, pInputFrame->pitch,
                frameOutInfoEx.width_byte, frameOutInfoEx.height_total, memcpyKind);
        } else {
            if (m_filterParam.frameOut.csp == NV_ENC_CSP_NV12) {
                //Y
                cudaMemcpy2D((uint8_t *)ppOutputFrames[0]->ptr, ppOutputFrames[0]->pitch,
                    (uint8_t *)pInputFrame->ptr + m_filterParam.crop.e.left + m_filterParam.crop.e.up * pInputFrame->pitch,
                    pInputFrame->pitch,
                    frameOutInfoEx.width_byte, m_filterParam.frameOut.height, memcpyKind);
                //UV
                cudaMemcpy2D((uint8_t *)ppOutputFrames[0]->ptr + ppOutputFrames[0]->pitch * m_filterParam.frameOut.height, ppOutputFrames[0]->pitch,
                    (uint8_t *)pInputFrame->ptr
                    + pInputFrame->height * pInputFrame->pitch
                    + m_filterParam.crop.e.left + (m_filterParam.crop.e.up >> 1) * pInputFrame->pitch,
                    pInputFrame->pitch,
                    frameOutInfoEx.width_byte, m_filterParam.frameOut.height >> 1, memcpyKind);
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
        return NV_ENC_ERR_UNIMPLEMENTED;
    }
    return sts;
}

void NVEncFilterCspCrop::close() {
    m_pFrameBuf.clear();
}
