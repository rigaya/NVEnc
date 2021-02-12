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
#include "NVEncFilterEdgelevel.h"
#include "NVEncParam.h"
#pragma warning (push)
#pragma warning (disable: 4819)
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#pragma warning (pop)
#include "rgy_cuda_util_kernel.h"

static const int EDGELEVEL_BLOCK_X = 32;
static const int EDGELEVEL_BLOCK_Y = 16;

__device__ __inline__
void check_min_max(float& min, float& max, float value) {
    max = fmaxf(max, value);
    min = fminf(min, value);
}

template<typename Type, int bit_depth>
__global__ void kernel_edgelevel(uint8_t *__restrict__ pDst, const int dstPitch, const int dstWidth, const int dstHeight,
    cudaTextureObject_t texSrc, const float strength, const float threshold, const float black, const float white) {
    const int ix = blockIdx.x * blockDim.x + threadIdx.x;
    const int iy = blockIdx.y * blockDim.y + threadIdx.y;

    if (ix < dstWidth && iy < dstHeight) {
        float x = ix + 0.5f;
        float y = iy + 0.5f;

        float center = tex2D<float>(texSrc, x, y);
        float min = center;
        float vmin = center;
        float max = center;
        float vmax = center;

        check_min_max(min,  max,  tex2D<float>(texSrc, x - 2.0f, y));
        check_min_max(vmin, vmax, tex2D<float>(texSrc, x, y - 2.0f));
        check_min_max(min,  max,  tex2D<float>(texSrc, x - 1.0f, y));
        check_min_max(vmin, vmax, tex2D<float>(texSrc, x, y - 1.0f));
        check_min_max(min,  max,  tex2D<float>(texSrc, x + 1.0f, y));
        check_min_max(vmin, vmax, tex2D<float>(texSrc, x, y + 1.0f));
        check_min_max(min,  max,  tex2D<float>(texSrc, x + 2.0f, y));
        check_min_max(vmin, vmax, tex2D<float>(texSrc, x, y + 2.0f));

        if (max - min < vmax - vmin) {
            max = vmax, min = vmin;
        }

        if (max - min > threshold) {
            float avg = (min + max) * 0.5f;
            if (center == min)
                min -= black;
            min -= black;
            if (center == max)
                max += white;
            max += white;

            center = fminf(fmaxf((center + ((center - avg) * strength)), min), max);
        }

        Type *ptr = (Type *)(pDst + iy * dstPitch + ix * sizeof(Type));
        ptr[0] = (Type)(clamp(center, 0.0f, 1.0f - RGY_FLT_EPS) * ((1 << bit_depth) - 1));
    }
}

template<typename Type>
cudaError_t textureCreateEdgelevel(cudaTextureObject_t &tex, cudaTextureFilterMode filterMode, cudaTextureReadMode readMode, uint8_t *ptr, int pitch, int width, int height) {
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
static RGY_ERR edgelevel_plane(FrameInfo *pOutputFrame, const FrameInfo *pInputFrame,
    float strength, float threshold, float black, float white, cudaStream_t stream) {
    dim3 blockSize(EDGELEVEL_BLOCK_X, EDGELEVEL_BLOCK_Y);
    dim3 gridSize(divCeil(pOutputFrame->width, blockSize.x), divCeil(pOutputFrame->height, blockSize.y));
    strength  /= (1<<4);
    threshold /= (1<<((sizeof(Type) * 8) - 1));
    black     /= (1<<(sizeof(Type) * 8));
    white     /= (1<<(sizeof(Type) * 8));

    cudaTextureObject_t texSrc = 0;
    auto cudaerr = textureCreateEdgelevel<Type>(texSrc, cudaFilterModePoint, cudaReadModeNormalizedFloat, pInputFrame->ptr, pInputFrame->pitch, pInputFrame->width, pInputFrame->height);
    if (cudaerr != cudaSuccess) {
        return err_to_rgy(cudaerr);
    }
    kernel_edgelevel<Type, bit_depth><<<gridSize, blockSize, 0, stream>>>((uint8_t *)pOutputFrame->ptr,
        pOutputFrame->pitch, pOutputFrame->width, pOutputFrame->height,
        texSrc, strength, threshold, black, white);
    cudaerr = cudaGetLastError();
    if (cudaerr != cudaSuccess) {
        return err_to_rgy(cudaerr);
    }
    cudaerr = cudaDestroyTextureObject(texSrc);
    if (cudaerr != cudaSuccess) {
    }
    return RGY_ERR_NONE;
}

template<typename Type, int bit_depth>
static RGY_ERR edgelevel_frame(FrameInfo *pOutputFrame, const FrameInfo *pInputFrame,
    float strength, float threshold, float black, float white, cudaStream_t stream) {
    const auto planeInputY = getPlane(pInputFrame, RGY_PLANE_Y);
    const auto planeInputU = getPlane(pInputFrame, RGY_PLANE_U);
    const auto planeInputV = getPlane(pInputFrame, RGY_PLANE_V);
    auto planeOutputY = getPlane(pOutputFrame, RGY_PLANE_Y);
    auto planeOutputU = getPlane(pOutputFrame, RGY_PLANE_U);
    auto planeOutputV = getPlane(pOutputFrame, RGY_PLANE_V);
    auto err = edgelevel_plane<Type, bit_depth>(&planeOutputY, &planeInputY,
        strength,
        threshold,
        black,
        white,
        stream);
    if (err != RGY_ERR_NONE) {
        return err;
    }
    err = copyPlane(&planeOutputU, &planeInputU, stream);
    if (err != RGY_ERR_NONE) {
        return err;
    }
    err = copyPlane(&planeOutputV, &planeInputV, stream);
    if (err != RGY_ERR_NONE) {
        return err;
    }
    auto cudaerr = cudaGetLastError();
    if (cudaerr != cudaSuccess) {
        return err_to_rgy(cudaerr);
    }
    return RGY_ERR_NONE;
}

NVEncFilterEdgelevel::NVEncFilterEdgelevel() {
    m_sFilterName = _T("edgelevel");
}

NVEncFilterEdgelevel::~NVEncFilterEdgelevel() {
    close();
}

RGY_ERR NVEncFilterEdgelevel::init(shared_ptr<NVEncFilterParam> pParam, shared_ptr<RGYLog> pPrintMes) {
    RGY_ERR sts = RGY_ERR_NONE;
    m_pPrintMes = pPrintMes;
    auto pEdgelevelParam = std::dynamic_pointer_cast<NVEncFilterParamEdgelevel>(pParam);
    if (!pEdgelevelParam) {
        AddMessage(RGY_LOG_ERROR, _T("Invalid parameter type.\n"));
        return RGY_ERR_INVALID_PARAM;
    }
    //パラメータチェック
    if (pEdgelevelParam->frameOut.height <= 0 || pEdgelevelParam->frameOut.width <= 0) {
        AddMessage(RGY_LOG_ERROR, _T("Invalid parameter.\n"));
        return RGY_ERR_INVALID_PARAM;
    }
    if (pEdgelevelParam->edgelevel.strength < -31.0f || 31.0f < pEdgelevelParam->edgelevel.strength) {
        pEdgelevelParam->edgelevel.strength = clamp(pEdgelevelParam->edgelevel.strength, -31.0f, 31.0f);
        AddMessage(RGY_LOG_WARN, _T("strength should be in range of %.1f - %.1f.\n"), -31.0f, 31.0f);
    }
    if (pEdgelevelParam->edgelevel.threshold < 0.0f || 255.0f < pEdgelevelParam->edgelevel.threshold) {
        pEdgelevelParam->edgelevel.threshold = clamp(pEdgelevelParam->edgelevel.threshold, 0.0f, 255.0f);
        AddMessage(RGY_LOG_WARN, _T("threshold should be in range of %.1f - %.1f.\n"), 0.0f, 255.0f);
    }
    if (pEdgelevelParam->edgelevel.black < 0.0f || 31.0f < pEdgelevelParam->edgelevel.black) {
        pEdgelevelParam->edgelevel.black = clamp(pEdgelevelParam->edgelevel.black, 0.0f, 31.0f);
        AddMessage(RGY_LOG_WARN, _T("black should be in range of %.1f - %.1f.\n"), 0.0f, 31.0f);
    }
    if (pEdgelevelParam->edgelevel.white < 0.0f || 31.0f < pEdgelevelParam->edgelevel.white) {
        pEdgelevelParam->edgelevel.white = clamp(pEdgelevelParam->edgelevel.white, 0.0f, 31.0f);
        AddMessage(RGY_LOG_WARN, _T("white should be in range of %.1f - %.1f.\n"), 0.0f, 31.0f);
    }

    auto cudaerr = AllocFrameBuf(pEdgelevelParam->frameOut, 1);
    if (cudaerr != cudaSuccess) {
        AddMessage(RGY_LOG_ERROR, _T("failed to allocate memory: %s.\n"), char_to_tstring(cudaGetErrorName(cudaerr)).c_str());
        return RGY_ERR_MEMORY_ALLOC;
    }
    pEdgelevelParam->frameOut.pitch = m_pFrameBuf[0]->frame.pitch;

    setFilterInfo(pParam->print());
    m_pParam = pEdgelevelParam;
    return sts;
}

tstring NVEncFilterParamEdgelevel::print() const {
    return edgelevel.print();
}

RGY_ERR NVEncFilterEdgelevel::run_filter(const FrameInfo *pInputFrame, FrameInfo **ppOutputFrames, int *pOutputFrameNum, cudaStream_t stream) {
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
    auto pEdgelevelParam = std::dynamic_pointer_cast<NVEncFilterParamEdgelevel>(m_pParam);
    if (!pEdgelevelParam) {
        AddMessage(RGY_LOG_ERROR, _T("Invalid parameter type.\n"));
        return RGY_ERR_INVALID_PARAM;
    }

    static const std::map<RGY_CSP, decltype(edgelevel_frame<uint8_t, 8>)*> denoise_list = {
        { RGY_CSP_YV12,      edgelevel_frame<uint8_t,   8> },
        { RGY_CSP_YV12_16,   edgelevel_frame<uint16_t, 16> },
        { RGY_CSP_YUV444,    edgelevel_frame<uint8_t,   8> },
        { RGY_CSP_YUV444_16, edgelevel_frame<uint16_t, 16> }
    };
    if (denoise_list.count(pInputFrame->csp) == 0) {
        AddMessage(RGY_LOG_ERROR, _T("unsupported csp %s.\n"), RGY_CSP_NAMES[pInputFrame->csp]);
        return RGY_ERR_UNSUPPORTED;
    }
    sts = denoise_list.at(pInputFrame->csp)(ppOutputFrames[0], pInputFrame,
        pEdgelevelParam->edgelevel.strength,
        pEdgelevelParam->edgelevel.threshold,
        pEdgelevelParam->edgelevel.black,
        pEdgelevelParam->edgelevel.white,
        stream);
    if (sts != RGY_ERR_NONE) {
        AddMessage(RGY_LOG_ERROR, _T("error at edgelevel(%s): %s.\n"),
            RGY_CSP_NAMES[pInputFrame->csp],
            get_err_mes(sts));
        return sts;
    }
    return sts;
}

void NVEncFilterEdgelevel::close() {
    m_pFrameBuf.clear();
}
