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
#include "convert_csp.h"
#include "NVEncFilterCurves.h"
#include "NVEncParam.h"
#include "rgy_avutil.h"
#include "rgy_util.h"
#include "rgy_aspect_ratio.h"
#include "cpu_info.h"

tstring NVEncFilterParamCurves::print() const {
    return curves.print();
}

NVEncFilterCurves::NVEncFilterCurves() :
    m_convIn(),
    m_convOut(),
    m_lut(),
    m_bInterlacedWarn(false) {
    m_sFilterName = _T("curves");
}

NVEncFilterCurves::~NVEncFilterCurves() {
    close();
}

RGY_ERR NVEncFilterCurves::checkParam(const NVEncFilterParam *param) {
    auto prm = dynamic_cast<const NVEncFilterParamCurves *>(param);
    if (!prm) {
        AddMessage(RGY_LOG_ERROR, _T("Invalid parameter type.\n"));
        return RGY_ERR_INVALID_PARAM;
    }
    //パラメータチェック
    if (prm->frameOut.height <= 0 || prm->frameOut.width <= 0) {
        AddMessage(RGY_LOG_ERROR, _T("Invalid parameter.\n"));
        return RGY_ERR_INVALID_PARAM;
    }
    return RGY_ERR_NONE;
}

std::vector<std::pair<double, double>> NVEncFilterCurves::parsePoints(const tstring& str) {
    std::vector<std::pair<double, double>> points;
    // "0/0 0.5/0.58 1/1"
    for (const auto& point : split(str, _T(" "))) {
        double a = 0.0, b = 0.0;
        if (_stscanf_s(point.c_str(), _T("%lf/%lf"), &a, &b) != 2) {
            AddMessage(RGY_LOG_ERROR, _T("Failed to parse cruves value: %s.\n"), point.c_str());
            return {};
        }
        points.push_back(std::make_pair(a, b));
    }
    return points;
}

template<typename Type>
std::vector<Type> NVEncFilterCurves::createLUT(const std::vector<std::pair<double, double>>& vec, const int scale) {
    const double scale_inv = 1.0 / scale;
    std::vector<Type> table(scale, 0);

    const int n = (int)vec.size();
    std::vector<vec3> coef(n, vec3(0.0, 0.0, 0.0));
    std::vector<double> h(n - 1, 0.0);
    for (int i = 0; i < n - 1; i++) {
        h[i] = vec[i + 1].first - vec[i].first;
    }

    std::vector<double> tmp(n, 0.0);
    for (int i = 1; i < n - 1; i++) {
        tmp[i] = (vec[i + 1].second - vec[i].second) / h[i] - (vec[i].second - vec[i - 1].second) / h[i - 1];
    }

    coef[0](0) = 1.0;
    for (int i = 1; i < n - 1; i++) {
        coef[i](0) = h[i - 1];
        coef[i](1) = (h[i] + h[i - 1]) * 2.0;
        coef[i](2) = h[i];
    }
    coef[n - 1](1) = 1.0;


    for (int i = 1; i < n; i++) {
        double d = coef[i](1) - coef[i](0) * coef[i - 1](2);
        if (d != 0.0) {
            d = 1.0 / d;
        }
        coef[i](2) *= d;
        tmp[i] = (tmp[i] - coef[i](0) * tmp[i - 1]) * d;
    }

    for (int i = n - 2; i >= 0; i--) {
        tmp[i] -= coef[i](2) * tmp[i + 1];
    }


    {
        const int x1 = (int)(vec.front().first * scale);
        const Type val = (Type)clamp((int)(vec.front().second * scale), 0, scale - 1);
        for (int ix = 0; ix <= x1; ix++) {
            table[ix] = val;
        }
    }
    for (int i = 0; i < n - 1; i++) {
        const double y0 = vec[i + 0].second;
        const double y1 = vec[i + 1].second;

        const double a0 = y0;
        const double a1 = (y1 - y0) / h[i] - h[i] * tmp[1] * 0.5 - h[i] * (tmp[i + 1] - tmp[i]) * (1.0 / 6.0);
        const double a2 = tmp[i] * 0.5;
        const double a3 = (tmp[i + 1] - tmp[i]) / (h[i] * 6.0);

        auto interp = [a0, a1, a2, a3](const double x) {
            return ((a3 * x + a2) * x + a1) * x + a0;
        };

        const int x0 = clamp((int)(vec[i + 0].first * scale), 0, scale - 1);
        const int x1 = clamp((int)(vec[i + 1].first * scale), 0, scale - 1);
        for (int ix = x0; ix <= x1; ix++) {
            table[ix] = (Type)clamp((int)(interp((ix - x0) * scale_inv) * scale), 0, scale - 1);
        }
    }
    {
        const int x0 = (int)(vec.back().first * scale);
        const Type val = (Type)clamp((int)(vec.back().second * scale), 0, scale - 1);
        for (int ix = x0; ix < scale; ix++) {
            table[ix] = val;
        }
    }
    return table;
}

VppCurveParams NVEncFilterCurves::getPreset(const VppCurvesPreset preset) {
    switch (preset) {
    case VppCurvesPreset::COLOR_NEGATIVE:
        return VppCurveParams(
            _T("0.129/1 0.466/0.498 0.725/0"),
            _T("0.109/1 0.301/0.498 0.517/0"),
            _T("0.098/1 0.235/0.498 0.423/0"), _T(""));
    case VppCurvesPreset::PROCESS:
        return VppCurveParams(
            _T("0/0 0.25/0.156 0.501/0.501 0.686/0.745 1/1"),
            _T("0/0 0.25/0.188 0.38/0.501 0.745/0.815 1/0.815"),
            _T("0/0 0.231/0.094 0.709/0.874 1/1"), _T(""));
    case VppCurvesPreset::DARKER:
        return VppCurveParams(_T(""), _T(""), _T(""), _T("0/0 0.5/0.4 1/1"));
    case VppCurvesPreset::LIGHTER:
        return VppCurveParams(_T(""), _T(""), _T(""), _T("0/0 0.4/0.5 1/1"));
    case VppCurvesPreset::INCREASE_CONTRAST:
        return VppCurveParams(_T(""), _T(""), _T(""), _T("0/0 0.149/0.066 0.831/0.905 0.905/0.98 1/1"));
    case VppCurvesPreset::LINEAR_CONTRAST:
        return VppCurveParams(_T(""), _T(""), _T(""), _T("0/0 0.305/0.286 0.694/0.713 1/1"));
    case VppCurvesPreset::MEDIUM_CONTRAST:
        return VppCurveParams(_T(""), _T(""), _T(""), _T("0/0 0.286/0.219 0.639/0.643 1/1"));
    case VppCurvesPreset::STRONG_CONTRAST:
        return VppCurveParams(_T(""), _T(""), _T(""), _T("0/0 0.301/0.196 0.592/0.6 0.686/0.737 1/1"));
    case VppCurvesPreset::NEGATIVE:
        return VppCurveParams(_T(""), _T(""), _T(""), _T("0/1 1/0"));
    case VppCurvesPreset::VINTAGE:
        return VppCurveParams(
            _T("0/0.11 0.42/0.51 1/0.95"),
            _T("0/0 0.50/0.48 1/1"),
            _T("0/0.22 0.49/0.44 1/0.8"), _T(""));
    case VppCurvesPreset::NONE:
    default:
        break;
    }
    return VppCurveParams();
}

template<typename Type>
RGY_ERR NVEncFilterCurves::createLUTFromParam(std::vector<Type>& lut, const tstring& str, const RGY_CSP csp, const std::vector<Type> *master) {
    if (str.length() == 0 && master == nullptr) {
        return RGY_ERR_NONE;
    }
    auto points = parsePoints((str.length() == 0) ? _T("0/0 1/1") : str);
    if (points.size() == 0) {
        return RGY_ERR_INVALID_PARAM;
    } else if (points.size() == 1) {
        return RGY_ERR_INVALID_PARAM;
    }
    lut = createLUT<Type>(points, 1 << RGY_CSP_BIT_DEPTH[csp]);
    if (master && master->size() > 0) {
        for (size_t i = 0; i < lut.size(); i++) {
            lut[i] = (*master)[lut[i]];
        }
    }
    return RGY_ERR_NONE;
}

template<typename Type>
RGY_ERR NVEncFilterCurves::sendLUTToGPU(std::unique_ptr<CUMemBuf>& mem, const std::vector<Type>& lut) {
    if (lut.size() > 0) {
        auto sts = RGY_ERR_NONE;
        mem = std::make_unique<CUMemBuf>(lut.size() * sizeof(lut[0]));
        if ((sts = err_to_rgy(mem->alloc())) != RGY_ERR_NONE) {
            AddMessage(RGY_LOG_ERROR, _T("Failed to allocate CUDA memory for lut: %s.\n"), get_err_mes(sts));
            return sts;
        }
        if ((sts = err_to_rgy(cudaMemcpy(mem->ptr, lut.data(), lut.size() * sizeof(lut[0]), cudaMemcpyHostToDevice))) != RGY_ERR_NONE) {
            AddMessage(RGY_LOG_ERROR, _T("Failed to send lut to CUDA memory: %s.\n"), get_err_mes(sts));
            return sts;
        }
    }
    return RGY_ERR_NONE;
}

template<typename Type>
RGY_ERR NVEncFilterCurves::createLUT(const VppCurveParams& prm, const RGY_CSP csp) {
    std::vector<Type> lutR, lutG, lutB, lutM;
    auto sts = RGY_ERR_NONE;
    if ((sts = createLUTFromParam<Type>(lutM, prm.m, csp, nullptr)) != RGY_ERR_NONE) {
        AddMessage(RGY_LOG_ERROR, _T("Failed to create LUT(m): %s.\n"), get_err_mes(sts));
        return sts;
    }
    if ((sts = createLUTFromParam<Type>(lutR, prm.r, csp, &lutM)) != RGY_ERR_NONE) {
        AddMessage(RGY_LOG_ERROR, _T("Failed to create LUT(r): %s.\n"), get_err_mes(sts));
        return sts;
    }
    if ((sts = createLUTFromParam<Type>(lutG, prm.g, csp, &lutM)) != RGY_ERR_NONE) {
        AddMessage(RGY_LOG_ERROR, _T("Failed to create LUT(g): %s.\n"), get_err_mes(sts));
        return sts;
    }
    if ((sts = createLUTFromParam<Type>(lutB, prm.b, csp, &lutM)) != RGY_ERR_NONE) {
        AddMessage(RGY_LOG_ERROR, _T("Failed to create LUT(b): %s.\n"), get_err_mes(sts));
        return sts;
    }
    if ((sts = sendLUTToGPU<Type>(m_lut.r, lutR)) != RGY_ERR_NONE) {
        AddMessage(RGY_LOG_ERROR, _T("Failed to send LUT(r) to GPU: %s.\n"), get_err_mes(sts));
        return sts;
    }
    if ((sts = sendLUTToGPU<Type>(m_lut.g, lutG)) != RGY_ERR_NONE) {
        AddMessage(RGY_LOG_ERROR, _T("Failed to send LUT(g) to GPU: %s.\n"), get_err_mes(sts));
        return sts;
    }
    if ((sts = sendLUTToGPU<Type>(m_lut.b, lutB)) != RGY_ERR_NONE) {
        AddMessage(RGY_LOG_ERROR, _T("Failed to send LUT(b) to GPU: %s.\n"), get_err_mes(sts));
        return sts;
    }
    return sts;
}

RGY_ERR NVEncFilterCurves::createLUT(const NVEncFilterParamCurves *prm) {
    VppCurveParams p = getPreset(prm->curves.preset);
    if (prm->curves.prm.r.length() > 0) p.r = prm->curves.prm.r;
    if (prm->curves.prm.g.length() > 0) p.g = prm->curves.prm.g;
    if (prm->curves.prm.b.length() > 0) p.b = prm->curves.prm.b;
    if (prm->curves.prm.m.length() > 0) p.m = prm->curves.prm.m;
    if (p.r.length() == 0) p.r = prm->curves.all;
    if (p.g.length() == 0) p.g = prm->curves.all;
    if (p.b.length() == 0) p.b = prm->curves.all;

    return (RGY_CSP_BIT_DEPTH[prm->frameIn.csp] > 8)
        ? createLUT<uint16_t>(p, prm->frameIn.csp)
        : createLUT<uint8_t>( p, prm->frameIn.csp);
}

RGY_ERR NVEncFilterCurves::init(shared_ptr<NVEncFilterParam> pParam, shared_ptr<RGYLog> pPrintMes) {
    RGY_ERR sts = RGY_ERR_NONE;
    m_pPrintMes = pPrintMes;
    auto prm = std::dynamic_pointer_cast<NVEncFilterParamCurves>(pParam);
    if (!prm) {
        AddMessage(RGY_LOG_ERROR, _T("Invalid parameter type.\n"));
        return RGY_ERR_INVALID_PARAM;
    }
    //パラメータチェック
    sts = checkParam(pParam.get());
    if (sts != RGY_ERR_NONE) {
        return sts;
    }
    if (!m_pParam ||
        prm->curves != std::dynamic_pointer_cast<NVEncFilterParamCurves>(m_pParam)->curves) {
        sts = createLUT(prm.get());
        if (sts != RGY_ERR_NONE) {
            return sts;
        }

        if (RGY_CSP_CHROMA_FORMAT[pParam->frameIn.csp] != RGY_CHROMAFMT_RGB) {
            const RGY_CSP rgb_csp = RGY_CSP_BIT_DEPTH[pParam->frameIn.csp] > 8 ? RGY_CSP_RGB_16 : RGY_CSP_RGB;
            if (prm->vuiInfo.matrix == RGY_MATRIX_UNSPECIFIED) {
                prm->vuiInfo.matrix = (CspMatrix)COLOR_VALUE_AUTO_RESOLUTION;
            }
            prm->vuiInfo.apply_auto(prm->vuiInfo, pParam->frameIn.height);
            {
                AddMessage(RGY_LOG_DEBUG, _T("Create input csp conversion filter.\n"));
                unique_ptr<NVEncFilterCspCrop> filter(new NVEncFilterCspCrop());
                shared_ptr<NVEncFilterParamCrop> paramCrop(new NVEncFilterParamCrop());
                paramCrop->frameIn = pParam->frameIn;
                paramCrop->frameOut = paramCrop->frameIn;
                paramCrop->frameOut.csp = rgb_csp;
                paramCrop->matrix = prm->vuiInfo.matrix;
                paramCrop->baseFps = pParam->baseFps;
                paramCrop->frameIn.deivce_mem = true;
                paramCrop->frameOut.deivce_mem = true;
                paramCrop->bOutOverwrite = false;
                sts = filter->init(paramCrop, m_pPrintMes);
                if (sts != RGY_ERR_NONE) {
                    return sts;
                }
                m_convIn = std::move(filter);
                AddMessage(RGY_LOG_DEBUG, _T("created %s.\n"), m_convIn->GetInputMessage().c_str());
            }
            {
                AddMessage(RGY_LOG_DEBUG, _T("Create output csp conversion filter.\n"));
                unique_ptr<NVEncFilterCspCrop> filter(new NVEncFilterCspCrop());
                shared_ptr<NVEncFilterParamCrop> paramCrop(new NVEncFilterParamCrop());
                paramCrop->frameIn = pParam->frameOut;
                paramCrop->frameIn.csp = rgb_csp;
                paramCrop->matrix = prm->vuiInfo.matrix;
                paramCrop->frameOut = pParam->frameOut;
                paramCrop->baseFps = pParam->baseFps;
                paramCrop->frameIn.deivce_mem = true;
                paramCrop->frameOut.deivce_mem = true;
                paramCrop->bOutOverwrite = false;
                sts = filter->init(paramCrop, m_pPrintMes);
                if (sts != RGY_ERR_NONE) {
                    return sts;
                }
                m_convOut = std::move(filter);
                AddMessage(RGY_LOG_DEBUG, _T("created %s.\n"), m_convOut->GetInputMessage().c_str());
            }
        }

        auto cudaerr = AllocFrameBuf(prm->frameOut, 1);
        if (cudaerr != cudaSuccess) {
            AddMessage(RGY_LOG_ERROR, _T("failed to allocate memory: %s.\n"), char_to_tstring(cudaGetErrorName(cudaerr)).c_str());
            return RGY_ERR_MEMORY_ALLOC;
        }
        prm->frameOut.pitch = m_pFrameBuf[0]->frame.pitch;
    }

    tstring info = _T("curves: ");
    if (m_convIn) {
        info += m_convIn->GetInputMessage() + _T("\n");
    }
    const auto extraIndent = tstring(_tcslen(_T("curves: ")), _T(' '));
    info += tstring(INFO_INDENT) + extraIndent + pParam->print();
    if (m_convOut) {
        info += tstring(_T("\n")) + tstring(INFO_INDENT) + extraIndent + m_convOut->GetInputMessage();
    }
    setFilterInfo(info);
    m_pParam = pParam;
    return sts;
}

RGY_ERR NVEncFilterCurves::run_filter(const RGYFrameInfo *pInputFrame, RGYFrameInfo **ppOutputFrames, int *pOutputFrameNum, cudaStream_t stream) {
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
    auto prm = std::dynamic_pointer_cast<NVEncFilterParamCurves>(m_pParam);
    if (!prm) {
        AddMessage(RGY_LOG_ERROR, _T("Invalid parameter type.\n"));
        return RGY_ERR_INVALID_PARAM;
    }
    RGYFrameInfo targetFrame = *pInputFrame;
    if (m_convIn) {
        int cropFilterOutputNum = 0;
        RGYFrameInfo *outInfo[1] = { 0 };
        auto sts_filter = m_convIn->filter(&targetFrame, (RGYFrameInfo **)&outInfo, &cropFilterOutputNum, stream);
        if (outInfo[0] == nullptr || cropFilterOutputNum != 1) {
            AddMessage(RGY_LOG_ERROR, _T("Unknown behavior \"%s\".\n"), m_convIn->name().c_str());
            return sts_filter;
        }
        if (sts_filter != RGY_ERR_NONE || cropFilterOutputNum != 1) {
            AddMessage(RGY_LOG_ERROR, _T("Error while running filter \"%s\".\n"), m_convIn->name().c_str());
            return sts_filter;
        }
        targetFrame = *outInfo[0];
    }

    if ((sts = procFrame(&targetFrame, stream)) != RGY_ERR_NONE) {
        return sts;
    }

    if (m_convOut) {
        auto sts_filter = m_convOut->filter(&targetFrame, ppOutputFrames, pOutputFrameNum, stream);
        if (ppOutputFrames[0] == nullptr || *pOutputFrameNum != 1) {
            AddMessage(RGY_LOG_ERROR, _T("Unknown behavior \"%s\".\n"), m_convOut->name().c_str());
            return sts_filter;
        }
        if (sts_filter != RGY_ERR_NONE || *pOutputFrameNum != 1) {
            AddMessage(RGY_LOG_ERROR, _T("Error while running filter \"%s\".\n"), m_convOut->name().c_str());
            return sts_filter;
        }
    }
    return sts;
}

void NVEncFilterCurves::close() {
    m_convIn.reset();
    m_convOut.reset();
    m_pFrameBuf.clear();
    m_bInterlacedWarn = false;
}
