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
#include "rgy_prm.h"
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

template<typename Type, int bit_depth>
__device__ __inline__
Type apply_basic_tweak_y_without_gamma(Type y, const float contrast, const float brightness) {
    float pixel = (float)y * (1.0f / (1 << bit_depth));
    pixel = contrast * (pixel - 0.5f) + 0.5f + brightness;
    return (Type)clamp((int)(pixel * (1 << (bit_depth))), 0, (1 << (bit_depth)) - 1);
}

template<typename Type, int bit_depth>
__device__ __inline__
Type apply_basic_tweak_cbcr(Type y, const float contrast, const float brightness) {
    float pixel = (float)y * (1.0f / (1 << bit_depth));
    pixel = contrast * pixel + brightness;
    return (Type)clamp((int)(pixel * (1 << (bit_depth))), 0, (1 << (bit_depth)) - 1);
}

template<typename Type, typename Type4, int bit_depth>
__global__ void kernel_tweak_y(uint8_t *__restrict__ pFrame, const int pitch,
    const int width, const int height,
    const float contrast, const float brightness, const float gamma_inv,
    const bool tweak_y, const float y_gain, const float y_offset) {
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

        if (tweak_y) {
            ret.x = apply_basic_tweak_y_without_gamma<Type, bit_depth>(ret.x, y_gain, y_offset);
            ret.y = apply_basic_tweak_y_without_gamma<Type, bit_depth>(ret.y, y_gain, y_offset);
            ret.z = apply_basic_tweak_y_without_gamma<Type, bit_depth>(ret.z, y_gain, y_offset);
            ret.w = apply_basic_tweak_y_without_gamma<Type, bit_depth>(ret.w, y_gain, y_offset);
        }

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
    const float saturation, const float hue_sin, const float hue_cos, const bool swapuv,
    const bool tweak_cb, const float cb_gain, const float cb_offset,
    const bool tweak_cr, const float cr_gain, const float cr_offset) {
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

        if (tweak_cb) {
            apply_basic_tweak_cbcr<Type, bit_depth>(pixelU.x, cb_gain, cb_offset);
            apply_basic_tweak_cbcr<Type, bit_depth>(pixelU.y, cb_gain, cb_offset);
            apply_basic_tweak_cbcr<Type, bit_depth>(pixelU.z, cb_gain, cb_offset);
            apply_basic_tweak_cbcr<Type, bit_depth>(pixelU.w, cb_gain, cb_offset);
        }
        if (tweak_cr) {
            apply_basic_tweak_cbcr<Type, bit_depth>(pixelV.x, cr_gain, cr_offset);
            apply_basic_tweak_cbcr<Type, bit_depth>(pixelV.y, cr_gain, cr_offset);
            apply_basic_tweak_cbcr<Type, bit_depth>(pixelV.z, cr_gain, cr_offset);
            apply_basic_tweak_cbcr<Type, bit_depth>(pixelV.w, cr_gain, cr_offset);
        }

        ptrU[0] = (swapuv) ? pixelV : pixelU;
        ptrV[0] = (swapuv) ? pixelU : pixelV;
    }
}

template<typename Type, typename Type4, int bit_depth>
static RGY_ERR tweak_frame(RGYFrameInfo *pFrame, const NVEncFilterParamTweak *prm, cudaStream_t stream) {
    auto planeInputY = getPlane(pFrame, RGY_PLANE_Y);
    auto planeInputU = getPlane(pFrame, RGY_PLANE_U);
    auto planeInputV = getPlane(pFrame, RGY_PLANE_V);
    static_assert(sizeof(Type4::x) == sizeof(Type), "sizeof(Type4::x) == sizeof(Type)");
    
    const float contrast   = prm->tweak.contrast;
    const float brightness = prm->tweak.brightness;
    const float saturation = prm->tweak.saturation;
    const float gamma      = prm->tweak.gamma;
    const float hue_degree = prm->tweak.hue;
    const int   swapuv     = prm->tweak.swapuv ? 1 : 0;

    //Y
    if (   contrast != 1.0f
        || brightness != 0.0f
        || gamma != 1.0f
        || prm->tweak.y.enabled()) {
        dim3 blockSize(TWEAK_BLOCK_X, TWEAK_BLOCK_Y);
        dim3 gridSize(divCeil(planeInputY.width, blockSize.x * 4), divCeil(planeInputY.height, blockSize.y));
        kernel_tweak_y<Type, Type4, bit_depth><<<gridSize, blockSize, 0, stream>>>(
            planeInputY.ptr[0], planeInputY.pitch[0], planeInputY.width, planeInputY.height,
            contrast, brightness, 1.0f / gamma,
            prm->tweak.y.enabled(), prm->tweak.y.gain, prm->tweak.y.offset);
        auto sts = err_to_rgy(cudaGetLastError());
        if (sts != RGY_ERR_NONE) {
            return sts;
        }
    }

    //UV
    if (saturation != 1.0f
        || hue_degree != 0.0f
        || swapuv
        || prm->tweak.cb.enabled()
        || prm->tweak.cr.enabled()) {
        if (   planeInputU.width  != planeInputV.width
            || planeInputU.height != planeInputV.height
            || planeInputU.pitch[0] != planeInputV.pitch[0]) {
            return RGY_ERR_UNSUPPORTED;
        }
        dim3 blockSize(TWEAK_BLOCK_X, TWEAK_BLOCK_Y);
        dim3 gridSize(divCeil(planeInputU.width, blockSize.x * 4), divCeil(planeInputU.height, blockSize.y));
        const float hue = hue_degree * (float)M_PI / 180.0f;
        kernel_tweak_uv<Type, Type4, bit_depth><<<gridSize, blockSize, 0, stream>>>(
            planeInputU.ptr[0], planeInputV.ptr[0], planeInputU.pitch[0], planeInputU.width, planeInputU.height,
            saturation, std::sin(hue) * saturation, std::cos(hue) * saturation, swapuv,
            prm->tweak.cb.enabled(), prm->tweak.cb.gain, prm->tweak.cb.offset,
            prm->tweak.cr.enabled(), prm->tweak.cr.gain, prm->tweak.cr.offset);
        auto sts = err_to_rgy(cudaGetLastError());
        if (sts != RGY_ERR_NONE) {
            return sts;
        }
    }
    return RGY_ERR_NONE;
}

template<typename Type, typename Type4, int bit_depth>
static RGY_ERR tweak_frame_rgb(RGYFrameInfo *pFrame, const NVEncFilterParamTweak *prm, cudaStream_t stream) {
    std::vector<std::pair<const VppTweakChannel *, RGYFrameInfo>> targetPlanes;
    if (prm->tweak.r.enabled()) { targetPlanes.push_back(std::make_pair(&prm->tweak.r, getPlane(pFrame, RGY_PLANE_R))); }
    if (prm->tweak.g.enabled()) { targetPlanes.push_back(std::make_pair(&prm->tweak.g, getPlane(pFrame, RGY_PLANE_G))); }
    if (prm->tweak.b.enabled()) { targetPlanes.push_back(std::make_pair(&prm->tweak.b, getPlane(pFrame, RGY_PLANE_B))); }

    for (auto& target : targetPlanes) {
        const auto& plane = target.second;
        dim3 blockSize(TWEAK_BLOCK_X, TWEAK_BLOCK_Y);
        dim3 gridSize(divCeil(plane.width, blockSize.x * 4), divCeil(plane.height, blockSize.y));
        kernel_tweak_y<Type, Type4, bit_depth><<<gridSize, blockSize, 0, stream>>>(
            plane.ptr[0], plane.pitch[0], plane.width, plane.height,
            target.first->gain, target.first->offset, 1.0f / target.first->gamma,
            false, 0.0f, 0.0f);
        auto sts = err_to_rgy(cudaGetLastError());
        if (sts != RGY_ERR_NONE) {
            return sts;
        }
    }
    return RGY_ERR_NONE;
}

NVEncFilterTweak::NVEncFilterTweak() :
    NVEncFilter(),
    m_convSrc(),
    m_convDst() {
    m_name = _T("tweak");
}

NVEncFilterTweak::~NVEncFilterTweak() {
    close();
}

RGY_ERR NVEncFilterTweak::init(shared_ptr<NVEncFilterParam> pParam, shared_ptr<RGYLog> pPrintMes) {
    RGY_ERR sts = RGY_ERR_NONE;
    m_pLog = pPrintMes;
    auto prm = std::dynamic_pointer_cast<NVEncFilterParamTweak>(pParam);
    if (!prm) {
        AddMessage(RGY_LOG_ERROR, _T("Invalid parameter type.\n"));
        return RGY_ERR_INVALID_PARAM;
    }
    //tweakは常に元のフレームを書き換え
    if (!prm->bOutOverwrite) {
        AddMessage(RGY_LOG_ERROR, _T("Invalid param, tweak will overwrite input frame.\n"));
        return RGY_ERR_INVALID_PARAM;
    }
    prm->frameOut = prm->frameIn;

    //パラメータチェック
    if (prm->frameOut.height <= 0 || prm->frameOut.width <= 0) {
        AddMessage(RGY_LOG_ERROR, _T("Invalid parameter.\n"));
        return RGY_ERR_INVALID_PARAM;
    }
    if (prm->tweak.brightness < -1.0f || 1.0f < prm->tweak.brightness) {
        prm->tweak.brightness = clamp(prm->tweak.brightness, -1.0f, 1.0f);
        AddMessage(RGY_LOG_WARN, _T("brightness should be in range of %.1f - %.1f.\n"), -1.0f, 1.0f);
    }
    if (prm->tweak.contrast < -2.0f || 2.0f < prm->tweak.contrast) {
        prm->tweak.contrast = clamp(prm->tweak.contrast, -2.0f, 2.0f);
        AddMessage(RGY_LOG_WARN, _T("contrast should be in range of %.1f - %.1f.\n"), -2.0f, 2.0f);
    }
    if (prm->tweak.saturation < 0.0f || 3.0f < prm->tweak.saturation) {
        prm->tweak.saturation = clamp(prm->tweak.saturation, 0.0f, 3.0f);
        AddMessage(RGY_LOG_WARN, _T("saturation should be in range of %.1f - %.1f.\n"), 0.0f, 3.0f);
    }
    if (prm->tweak.gamma < 0.1f || 10.0f < prm->tweak.gamma) {
        prm->tweak.gamma = clamp(prm->tweak.gamma, 0.1f, 10.0f);
        AddMessage(RGY_LOG_WARN, _T("gamma should be in range of %.1f - %.1f.\n"), 0.1f, 10.0f);
    }
    for (auto prmtweak : { &prm->tweak.r, &prm->tweak.g, &prm->tweak.b, &prm->tweak.y, &prm->tweak.cb, &prm->tweak.cr }) {
        if (prmtweak->offset < -1.0f || 1.0f < prmtweak->offset) {
            prmtweak->offset = clamp(prmtweak->offset, -1.0f, 1.0f);
            AddMessage(RGY_LOG_WARN, _T("offset should be in range of %.1f - %.1f.\n"), -1.0f, 1.0f);
        }
        if (prmtweak->gain < -2.0f || 2.0f < prmtweak->gain) {
            prmtweak->gain = clamp(prmtweak->gain, -2.0f, 2.0f);
            AddMessage(RGY_LOG_WARN, _T("gain should be in range of %.1f - %.1f.\n"), -2.0f, 2.0f);
        }
    }
    for (auto prmtweak : { &prm->tweak.r, &prm->tweak.g, &prm->tweak.b }) {
        if (prmtweak->gamma < 0.1f || 10.0f < prmtweak->gamma) {
            prmtweak->gamma = clamp(prmtweak->gamma, 0.1f, 10.0f);
            AddMessage(RGY_LOG_WARN, _T("gamma should be in range of %.1f - %.1f.\n"), 0.1f, 10.0f);
        }
    }

    if ((prm->tweak.r.enabled() || prm->tweak.g.enabled() || prm->tweak.b.enabled())
        && (!m_convSrc || !m_convDst)) {
        const auto csp_rgb = RGY_CSP_RGB_16;
        VideoVUIInfo vui = prm->vui;
        vui.setIfUnset(VideoVUIInfo().to((CspMatrix)COLOR_VALUE_AUTO_RESOLUTION).to((CspColorprim)COLOR_VALUE_AUTO_RESOLUTION).to((CspTransfer)COLOR_VALUE_AUTO_RESOLUTION));
        vui.apply_auto(VideoVUIInfo(), pParam->frameIn.height);
        {
            unique_ptr<NVEncFilterCspCrop> filterCrop(new NVEncFilterCspCrop());
            shared_ptr<NVEncFilterParamCrop> paramCrop(new NVEncFilterParamCrop());
            paramCrop->frameIn = pParam->frameIn;
            paramCrop->frameOut = pParam->frameIn;
            paramCrop->frameOut.csp = csp_rgb;
            paramCrop->baseFps = pParam->baseFps;
            paramCrop->matrix = vui.matrix;
            paramCrop->frameOut.mem_type = RGY_MEM_TYPE_GPU;
            paramCrop->bOutOverwrite = false;
            sts = filterCrop->init(paramCrop, m_pLog);
            if (sts != RGY_ERR_NONE) {
                return sts;
            }
            m_convSrc = std::move(filterCrop);
        }
        {
            unique_ptr<NVEncFilterCspCrop> filterCrop(new NVEncFilterCspCrop());
            shared_ptr<NVEncFilterParamCrop> paramCrop(new NVEncFilterParamCrop());
            paramCrop->frameIn = pParam->frameIn;
            paramCrop->frameIn.csp = csp_rgb;
            paramCrop->frameOut = pParam->frameIn;
            paramCrop->baseFps = pParam->baseFps;
            paramCrop->matrix = vui.matrix;
            paramCrop->frameOut.mem_type = RGY_MEM_TYPE_GPU;
            paramCrop->bOutOverwrite = false;
            sts = filterCrop->init(paramCrop, m_pLog);
            if (sts != RGY_ERR_NONE) {
                return sts;
            }
            m_convDst = std::move(filterCrop);
        }
    }

    //コピーを保存
    if (prm->tweak.r.enabled() || prm->tweak.g.enabled() || prm->tweak.b.enabled()) {
        tstring str = prm->tweak.print(false) + _T("\n");
        const tstring indent = _T("                ");
        str += indent + m_convSrc->GetInputMessage() + _T("\n");
        if (prm->tweak.r.enabled()) str += indent + _T("r: ") + prm->tweak.r.print() + _T("\n");
        if (prm->tweak.g.enabled()) str += indent + _T("g: ") + prm->tweak.g.print() + _T("\n");
        if (prm->tweak.b.enabled()) str += indent + _T("b: ") + prm->tweak.b.print() + _T("\n");
        str += indent + m_convDst->GetInputMessage();
        setFilterInfo(str);
    } else {
        setFilterInfo(prm->print());
    }
    m_param = prm;
    return sts;
}

tstring NVEncFilterParamTweak::print() const {
    return tweak.print();
}

RGY_ERR NVEncFilterTweak::run_filter(const RGYFrameInfo *pInputFrame, RGYFrameInfo **ppOutputFrames, int *pOutputFrameNum, cudaStream_t stream) {
    RGY_ERR sts = RGY_ERR_NONE;
    if (pInputFrame->ptr[0] == nullptr) {
        return sts;
    }

    *pOutputFrameNum = 1;
    if (ppOutputFrames[0] == nullptr) {
        AddMessage(RGY_LOG_ERROR, _T("ppOutputFrames[0] must be set.\n"));
        return RGY_ERR_INVALID_PARAM;
    }
    if (!ppOutputFrames[0]->mem_type) {
        AddMessage(RGY_LOG_ERROR, _T("only supported on device memory.\n"));
        return RGY_ERR_INVALID_PARAM;
    }
    const auto memcpyKind = getCudaMemcpyKind(ppOutputFrames[0]->mem_type, ppOutputFrames[0]->mem_type);
    if (memcpyKind != cudaMemcpyDeviceToDevice) {
        AddMessage(RGY_LOG_ERROR, _T("only supported on device memory.\n"));
        return RGY_ERR_INVALID_PARAM;
    }
    if (m_param->frameOut.csp != m_param->frameIn.csp) {
        AddMessage(RGY_LOG_ERROR, _T("csp does not match.\n"));
        return RGY_ERR_INVALID_PARAM;
    }
    ppOutputFrames[0]->picstruct = pInputFrame->picstruct;
    auto prm = std::dynamic_pointer_cast<NVEncFilterParamTweak>(m_param);
    if (!prm) {
        AddMessage(RGY_LOG_ERROR, _T("Invalid parameter type.\n"));
        return RGY_ERR_INVALID_PARAM;
    }

    static const std::map<RGY_CSP, decltype(tweak_frame<uint8_t, uchar4, 8>)*> tweak_list = {
        { RGY_CSP_YV12,      tweak_frame<uint8_t,  uchar4,   8> },
        { RGY_CSP_YV12_16,   tweak_frame<uint16_t, ushort4, 16> },
        { RGY_CSP_YUV444,    tweak_frame<uint8_t,  uchar4,   8> },
        { RGY_CSP_YUV444_16, tweak_frame<uint16_t, ushort4, 16> }
    };
    if (tweak_list.count(pInputFrame->csp) == 0) {
        AddMessage(RGY_LOG_ERROR, _T("unsupported csp %s.\n"), RGY_CSP_NAMES[pInputFrame->csp]);
        return RGY_ERR_UNSUPPORTED;
    }
    sts = tweak_list.at(pInputFrame->csp)(ppOutputFrames[0], prm.get(), stream);
    if (sts != RGY_ERR_NONE) {
        AddMessage(RGY_LOG_ERROR, _T("error at tweak(%s): %s.\n"),
            RGY_CSP_NAMES[pInputFrame->csp],
            get_err_mes(sts));
        return sts;
    }
    if (prm->tweak.r.enabled() || prm->tweak.g.enabled() || prm->tweak.b.enabled()) {
        RGYFrameInfo *targetFrame = nullptr;
        { // YUV -> RGB
            int cropFilterOutputNum = 0;
            RGYFrameInfo *pCropFilterOutput[1] = { nullptr };
            RGYFrameInfo inFrame = *(ppOutputFrames[0]);
            auto sts_filter = m_convSrc->filter(&inFrame, (RGYFrameInfo **)&pCropFilterOutput, &cropFilterOutputNum, stream);
            if (pCropFilterOutput[0] == nullptr || cropFilterOutputNum != 1) {
                AddMessage(RGY_LOG_ERROR, _T("Unknown behavior \"%s\".\n"), m_convSrc->name().c_str());
                return sts_filter;
            }
            if (sts_filter != RGY_ERR_NONE || cropFilterOutputNum != 1) {
                AddMessage(RGY_LOG_ERROR, _T("Error while running filter \"%s\".\n"), m_convSrc->name().c_str());
                return sts_filter;
            }
            targetFrame = pCropFilterOutput[0];
        }
        // RGBでの処理を実行
        sts = tweak_frame_rgb<uint16_t, ushort4, 16>(targetFrame, prm.get(), stream);
        if (sts != RGY_ERR_NONE) {
            AddMessage(RGY_LOG_ERROR, _T("error at tweak_rgb(%s): %s.\n"),
                RGY_CSP_NAMES[pInputFrame->csp],
                get_err_mes(sts));
            return sts;
        }
        { // RGB -> YUV
            int cropFilterOutputNum = 0;
            RGYFrameInfo *pCropFilterOutput[1] = { nullptr };
            RGYFrameInfo inFrame = *targetFrame;
            auto sts_filter = m_convDst->filter(&inFrame, (RGYFrameInfo **)&pCropFilterOutput, &cropFilterOutputNum, stream);
            if (pCropFilterOutput[0] == nullptr || cropFilterOutputNum != 1) {
                AddMessage(RGY_LOG_ERROR, _T("Unknown behavior \"%s\".\n"), m_convDst->name().c_str());
                return sts_filter;
            }
            if (sts_filter != RGY_ERR_NONE || cropFilterOutputNum != 1) {
                AddMessage(RGY_LOG_ERROR, _T("Error while running filter \"%s\".\n"), m_convDst->name().c_str());
                return sts_filter;
            }
            ppOutputFrames[0] = pCropFilterOutput[0];
        }
    }
    return RGY_ERR_NONE;
}

void NVEncFilterTweak::close() {
    m_convSrc.reset();
    m_convDst.reset();
    m_frameBuf.clear();
}
