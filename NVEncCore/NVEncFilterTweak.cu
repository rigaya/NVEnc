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
#include "convert_csp.h"
#include "NVEncFilterTweak.h"
#include "NVEncParam.h"
#pragma warning (push)
#pragma warning (disable: 4819)
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#pragma warning (pop)

static const int TWEAK_BLOCK_X = 64;
static const int TWEAK_BLOCK_Y = 4;

template<typename Type, int bit_depth>
__device__ __inline__
Type apply_basic_tweak_y(Type y, const float contrast, const float brightness, const float gamma_inv) {
    float pixel = (float)y * (1.0f / (1 << bit_depth));
    pixel = contrast * (pixel - 0.5f) + 0.5f + brightness;
    pixel = powf(pixel, gamma_inv);
    return (Type)clamp((int)(pixel * (1 << (bit_depth))), 0, (1 << (bit_depth)) - 1);
}

template<typename Type, typename Type4, int bit_depth>
__global__ void kernel_tweak_y(uint8_t *__restrict__ pFrame, const int pitch,
    const int width, const int height,
    const float contrast, const float brightness, const float gamma_inv) {
    const int ix = blockIdx.x * blockDim.x + threadIdx.x;
    const int iy = blockIdx.y * blockDim.y + threadIdx.y;

    if (ix < width && iy < height) {
        Type4 *ptr = (Type4 *)(pFrame + iy * pitch + ix * sizeof(Type4));
        Type4 src = ptr[0];

        Type4 ret;
        ret.x = apply_basic_tweak_y<Type, bit_depth>(src.x, contrast, brightness, gamma_inv);
        ret.y = apply_basic_tweak_y<Type, bit_depth>(src.y, contrast, brightness, gamma_inv);
        ret.z = apply_basic_tweak_y<Type, bit_depth>(src.z, contrast, brightness, gamma_inv);
        ret.w = apply_basic_tweak_y<Type, bit_depth>(src.w, contrast, brightness, gamma_inv);

        ptr[0] = ret;
    }
}

template<typename Type, int bit_depth>
__device__ __inline__
void apply_basic_tweak_uv(Type& u, Type& v, const float saturation, const float hue_sin, const float hue_cos) {
    float u0 = (float)u * (1.0f / (1 << bit_depth));
    float v0 = (float)v * (1.0f / (1 << bit_depth));
    u0 = saturation * (u0 - 0.5f) + 0.5f;
    v0 = saturation * (v0 - 0.5f) + 0.5f;

    float u1 = ((hue_cos * (u0 - 0.5f)) - (hue_sin * (v0 - 0.5f))) + 0.5f;
    float v1 = ((hue_sin * (u0 - 0.5f)) + (hue_cos * (v0 - 0.5f))) + 0.5f;

    u = (Type)clamp((int)(u1 * (1 << (bit_depth))), 0, (1 << (bit_depth)) - 1);
    v = (Type)clamp((int)(v1 * (1 << (bit_depth))), 0, (1 << (bit_depth)) - 1);
}

template<typename Type, typename Type4, int bit_depth>
__global__ void kernel_tweak_uv(uint8_t *__restrict__ pFrameU, uint8_t *__restrict__ pFrameV, const int pitch,
    const int width, const int height,
    const float saturation, const float hue_sin, const float hue_cos) {
    const int ix = blockIdx.x * blockDim.x + threadIdx.x;
    const int iy = blockIdx.y * blockDim.y + threadIdx.y;

    if (ix < width && iy < height) {
        Type4 *ptrU = (Type4 *)(pFrameU + iy * pitch + ix * sizeof(Type4));
        Type4 *ptrV = (Type4 *)(pFrameV + iy * pitch + ix * sizeof(Type4));

        Type4 pixelU = ptrU[0];
        Type4 pixelV = ptrV[0];

        apply_basic_tweak_uv<Type, bit_depth>(pixelU.x, pixelV.x, saturation, hue_sin, hue_cos);
        apply_basic_tweak_uv<Type, bit_depth>(pixelU.y, pixelV.y, saturation, hue_sin, hue_cos);
        apply_basic_tweak_uv<Type, bit_depth>(pixelU.z, pixelV.z, saturation, hue_sin, hue_cos);
        apply_basic_tweak_uv<Type, bit_depth>(pixelU.w, pixelV.w, saturation, hue_sin, hue_cos);

        ptrU[0] = pixelU;
        ptrV[0] = pixelV;
    }
}

template<typename Type, typename Type4, int bit_depth>
static cudaError_t tweak_yv12(FrameInfo *pFrame,
    float contrast, float brightness, float saturation, float gamma, float hue_degree) {
    dim3 blockSize(TWEAK_BLOCK_X, TWEAK_BLOCK_Y);
    dim3 gridSize(divCeil(pFrame->width, blockSize.x * 4), divCeil(pFrame->height, blockSize.y));

    //Y
    if (   contrast != 1.0f
        || brightness != 0.0f
        || gamma != 1.0f) {
        kernel_tweak_y<Type, Type4, bit_depth><<<gridSize, blockSize>>>(
            (uint8_t *)pFrame->ptr, pFrame->pitch,
            pFrame->width, pFrame->height,
            contrast, brightness, 1.0f / gamma);
        auto cudaerr = cudaGetLastError();
        if (cudaerr != cudaSuccess) {
            return cudaerr;
        }
    }

    //UV
    if (saturation != 1.0f
        || hue_degree != 0.0f) {
        const float hue = hue_degree * (float)M_PI / 180.0f;
        gridSize = dim3(divCeil(pFrame->width >> 1, blockSize.x * 4), divCeil(pFrame->height >> 1, blockSize.y));
        kernel_tweak_uv<Type, Type4, bit_depth><<<gridSize, blockSize>>>(
            (uint8_t *)pFrame->ptr + pFrame->pitch * pFrame->height,
            (uint8_t *)pFrame->ptr + pFrame->pitch * pFrame->height * 3 / 2,
            pFrame->pitch,
            pFrame->width >> 1, pFrame->height >> 1,
            saturation, std::sin(hue) * saturation, std::cos(hue) * saturation);
        auto cudaerr = cudaGetLastError();
        if (cudaerr != cudaSuccess) {
            return cudaerr;
        }
    }
    return cudaSuccess;
}

template<typename Type, typename Type4, int bit_depth>
static cudaError_t tweak_yuv444(FrameInfo *pFrame,
    float contrast, float brightness, float saturation, float gamma, float hue_degree) {
    dim3 blockSize(TWEAK_BLOCK_X, TWEAK_BLOCK_Y);
    dim3 gridSize(divCeil(pFrame->width, blockSize.x * 4), divCeil(pFrame->height, blockSize.y));

    //Y
    if (contrast != 1.0f
        || brightness != 0.0f
        || gamma != 1.0f) {
        kernel_tweak_y<Type, Type4, bit_depth><<<gridSize, blockSize>>>(
            (uint8_t *)pFrame->ptr, pFrame->pitch,
            pFrame->width, pFrame->height,
            contrast, brightness, 1.0f / gamma);
        auto cudaerr = cudaGetLastError();
        if (cudaerr != cudaSuccess) {
            return cudaerr;
        }
    }

    //UV
    if (saturation != 1.0f
        || hue_degree != 0.0f) {
        const float hue = hue_degree * (float)M_PI / 180.0f;
        kernel_tweak_uv<Type, Type4, bit_depth><<<gridSize, blockSize>>>(
            (uint8_t *)pFrame->ptr + pFrame->pitch * pFrame->height,
            (uint8_t *)pFrame->ptr + pFrame->pitch * pFrame->height * 2,
            pFrame->pitch,
            pFrame->width, pFrame->height,
            saturation, std::sin(hue) * saturation, std::cos(hue) * saturation);
        auto cudaerr = cudaGetLastError();
        if (cudaerr != cudaSuccess) {
            return cudaerr;
        }
    }
    return cudaSuccess;
}

NVEncFilterTweak::NVEncFilterTweak() {
    m_sFilterName = _T("tweak");
}

NVEncFilterTweak::~NVEncFilterTweak() {
    close();
}

NVENCSTATUS NVEncFilterTweak::init(shared_ptr<NVEncFilterParam> pParam, shared_ptr<RGYLog> pPrintMes) {
    NVENCSTATUS sts = NV_ENC_SUCCESS;
    m_pPrintMes = pPrintMes;
    auto pTweakParam = std::dynamic_pointer_cast<NVEncFilterParamTweak>(pParam);
    if (!pTweakParam) {
        AddMessage(RGY_LOG_ERROR, _T("Invalid parameter type.\n"));
        return NV_ENC_ERR_INVALID_PARAM;
    }
    //delogoは常に元のフレームを書き換え
    if (!pTweakParam->bOutOverwrite) {
        AddMessage(RGY_LOG_ERROR, _T("Invalid param, delogo will overwrite input frame.\n"));
        return NV_ENC_ERR_INVALID_PARAM;
    }
    pTweakParam->frameOut = pTweakParam->frameIn;

    //パラメータチェック
    if (pTweakParam->frameOut.height <= 0 || pTweakParam->frameOut.width <= 0) {
        AddMessage(RGY_LOG_ERROR, _T("Invalid parameter.\n"));
        return NV_ENC_ERR_INVALID_PARAM;
    }
    if (pTweakParam->tweak.brightness < -1.0f || 1.0f < pTweakParam->tweak.brightness) {
        pTweakParam->tweak.brightness = clamp(pTweakParam->tweak.brightness, -1.0f, 1.0f);
        AddMessage(RGY_LOG_WARN, _T("brightness should be in range of %.1f - %.1f.\n"), -1.0f, 1.0f);
    }
    if (pTweakParam->tweak.contrast < -2.0f || 2.0f < pTweakParam->tweak.contrast) {
        pTweakParam->tweak.contrast = clamp(pTweakParam->tweak.contrast, -2.0f, 2.0f);
        AddMessage(RGY_LOG_WARN, _T("contrast should be in range of %.1f - %.1f.\n"), -2.0f, 2.0f);
    }
    if (pTweakParam->tweak.saturation < 0.0f || 3.0f < pTweakParam->tweak.saturation) {
        pTweakParam->tweak.saturation = clamp(pTweakParam->tweak.saturation, 0.0f, 3.0f);
        AddMessage(RGY_LOG_WARN, _T("saturation should be in range of %.1f - %.1f.\n"), 0.0f, 3.0f);
    }
    if (pTweakParam->tweak.gamma < 0.1f || 10.0f < pTweakParam->tweak.gamma) {
        pTweakParam->tweak.gamma = clamp(pTweakParam->tweak.gamma, 0.1f, 10.0f);
        AddMessage(RGY_LOG_WARN, _T("gamma should be in range of %.1f - %.1f.\n"), 0.1f, 10.0f);
    }

    m_sFilterInfo = strsprintf(_T("tweak: brightness %.1f, contrast %.1f, saturation %.1f, gamma %.1f, hue %.1f"),
        pTweakParam->tweak.brightness, pTweakParam->tweak.contrast, pTweakParam->tweak.saturation, pTweakParam->tweak.gamma, pTweakParam->tweak.hue);

    //コピーを保存
    m_pParam = pTweakParam;
    return sts;
}

NVENCSTATUS NVEncFilterTweak::run_filter(const FrameInfo *pInputFrame, FrameInfo **ppOutputFrames, int *pOutputFrameNum) {
    NVENCSTATUS sts = NV_ENC_SUCCESS;
    if (pInputFrame->ptr == nullptr) {
        return sts;
    }

    *pOutputFrameNum = 1;
    if (ppOutputFrames[0] == nullptr) {
        AddMessage(RGY_LOG_ERROR, _T("ppOutputFrames[0] must be set.\n"));
        return NV_ENC_ERR_UNSUPPORTED_PARAM;
    }
    if (!ppOutputFrames[0]->deivce_mem) {
        AddMessage(RGY_LOG_ERROR, _T("only supported on device memory.\n"));
        return NV_ENC_ERR_UNSUPPORTED_PARAM;
    }
    const auto memcpyKind = getCudaMemcpyKind(ppOutputFrames[0]->deivce_mem, ppOutputFrames[0]->deivce_mem);
    if (memcpyKind != cudaMemcpyDeviceToDevice) {
        AddMessage(RGY_LOG_ERROR, _T("only supported on device memory.\n"));
        return NV_ENC_ERR_UNSUPPORTED_PARAM;
    }
    if (m_pParam->frameOut.csp != m_pParam->frameIn.csp) {
        AddMessage(RGY_LOG_ERROR, _T("csp does not match.\n"));
        return NV_ENC_ERR_UNSUPPORTED_PARAM;
    }
    ppOutputFrames[0]->picstruct = pInputFrame->picstruct;
    auto pTweakParam = std::dynamic_pointer_cast<NVEncFilterParamTweak>(m_pParam);
    if (!pTweakParam) {
        AddMessage(RGY_LOG_ERROR, _T("Invalid parameter type.\n"));
        return NV_ENC_ERR_INVALID_PARAM;
    }

    static const std::map<RGY_CSP, decltype(tweak_yv12<uint8_t, uchar4, 8>)*> tweak_list = {
        { RGY_CSP_YV12,      tweak_yv12<uint8_t,    uchar4,   8> },
        { RGY_CSP_YV12_16,   tweak_yv12<uint16_t,   ushort4, 16> },
        { RGY_CSP_YUV444,    tweak_yuv444<uint8_t,  uchar4,   8> },
        { RGY_CSP_YUV444_16, tweak_yuv444<uint16_t, ushort4, 16> }
    };
    if (tweak_list.count(pInputFrame->csp) == 0) {
        AddMessage(RGY_LOG_ERROR, _T("unsupported csp %s.\n"), RGY_CSP_NAMES[pInputFrame->csp]);
        return NV_ENC_ERR_UNIMPLEMENTED;
    }
    tweak_list.at(pInputFrame->csp)(ppOutputFrames[0],
        pTweakParam->tweak.contrast,
        pTweakParam->tweak.brightness,
        pTweakParam->tweak.saturation,
        pTweakParam->tweak.gamma,
        pTweakParam->tweak.hue);
    auto cudaerr = cudaGetLastError();
    if (cudaerr != cudaSuccess) {
        AddMessage(RGY_LOG_ERROR, _T("error at tweak(%s): %s.\n"),
            RGY_CSP_NAMES[pInputFrame->csp],
            char_to_tstring(cudaGetErrorString(cudaerr)).c_str());
        return NV_ENC_ERR_INVALID_CALL;
    }
    return sts;
}

void NVEncFilterTweak::close() {
    m_pFrameBuf.clear();
}
