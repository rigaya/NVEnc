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

#pragma once

#include <stdint.h>
#include <memory>
#include <vector>
#include "rgy_frame.h"
#include "rgy_osdep.h"
#include "rgy_tchar.h"
#include "rgy_log.h"
#include "rgy_prm.h"
#include "convert_csp.h"
#include "rgy_filter.h"
#include "rgy_cuda_util.h"

#pragma comment(lib, "cudart_static.lib")

struct AVPacket;

extern const TCHAR *NPPI_DLL_NAME_TSTR;
extern const TCHAR *NVRTC_DLL_NAME_TSTR;
extern const TCHAR *NVRTC_BUILTIN_DLL_NAME_TSTR;

bool check_if_nppi_dll_available();
#if ENABLE_NVRTC
bool check_if_nvrtc_dll_available();
bool check_if_nvrtc_builtin_dll_available();
#endif

using std::vector;

RGY_ERR copyPlane(RGYFrameInfo *pOutputFrame, const RGYFrameInfo *pInputFrame, cudaStream_t stream);
RGY_ERR copyFrame(RGYFrameInfo *pOutputFrame, const RGYFrameInfo *pInputFrame, cudaStream_t stream);

static NppiSize nppisize(const RGYFrameInfo *pFrame) {
    NppiSize size;
    size.width = pFrame->width;
    size.height = pFrame->height;
    return size;
}

static NppiRect nppiroi(const RGYFrameInfo *pFrame) {
    NppiRect rect;
    rect.x = 0;
    rect.y = 0;
    rect.width = pFrame->width;
    rect.height = pFrame->height;
    return rect;
}

class RGYFilterPerfNV : public RGYFilterPerf {
public:
    RGYFilterPerfNV() : RGYFilterPerf() {};
    virtual ~RGYFilterPerfNV() { };

    virtual RGY_ERR checkPerformace(void *event_start, void *event_fin) override;
protected:
};

using NVEncFilterParam = RGYFilterParam;

class NVEncFilter : public RGYFilterBase {
public:
    NVEncFilter();
    virtual ~NVEncFilter();
    RGY_ERR filter(RGYFrameInfo *pInputFrame, RGYFrameInfo **ppOutputFrames, int *pOutputFrameNum, cudaStream_t stream);
    virtual void setCheckPerformance(const bool check) override;
protected:
    virtual RGY_ERR AllocFrameBuf(const RGYFrameInfo &frame, int frames) override;
    RGY_ERR filter_as_interlaced_pair(const RGYFrameInfo *pInputFrame, RGYFrameInfo *pOutputFrame, cudaStream_t stream);
    virtual RGY_ERR run_filter(const RGYFrameInfo *pInputFrame, RGYFrameInfo **ppOutputFrames, int *pOutputFrameNum, cudaStream_t stream) = 0;

    static const TCHAR *INFO_INDENT;
    std::vector<unique_ptr<CUFrameBuf>> m_frameBuf;
    int m_nFrameIdx;
    std::unique_ptr<CUFrameBuf> m_pFieldPairIn;
    std::unique_ptr<CUFrameBuf> m_pFieldPairOut;
    std::unique_ptr<cudaEvent_t, cudaevent_deleter> m_peFilterStart;
    std::unique_ptr<cudaEvent_t, cudaevent_deleter> m_peFilterFin;
};

class NVEncFilterParamCrop : public NVEncFilterParam {
public:
    sInputCrop crop;
    CspMatrix matrix;

    NVEncFilterParamCrop();
    virtual ~NVEncFilterParamCrop();
    virtual tstring print() const override;
};

class NVEncFilterCspCrop : public NVEncFilter {
public:
    NVEncFilterCspCrop();
    virtual ~NVEncFilterCspCrop();
    virtual RGY_ERR init(shared_ptr<NVEncFilterParam> pParam, shared_ptr<RGYLog> pPrintMes) override;
protected:
    virtual RGY_ERR run_filter(const RGYFrameInfo *pInputFrame, RGYFrameInfo **ppOutputFrames, int *pOutputFrameNum, cudaStream_t stream) override;
    RGY_ERR convertYBitDepth(RGYFrameInfo *pOutputFrame, const RGYFrameInfo *pInputFrame, cudaStream_t stream);
    RGY_ERR convertCspFromNV12(RGYFrameInfo *pOutputFrame, const RGYFrameInfo *pInputFrame, cudaStream_t stream);
    RGY_ERR convertCspFromYV12(RGYFrameInfo *pOutputFrame, const RGYFrameInfo *pInputFrame, cudaStream_t stream);
    RGY_ERR convertCspFromNV16(RGYFrameInfo *pOutputFrame, const RGYFrameInfo *pInputFrame, cudaStream_t stream);
    RGY_ERR convertCspFromRGB(RGYFrameInfo *pOutputFrame, const RGYFrameInfo *pInputFrame, cudaStream_t stream);
    RGY_ERR convertCspFromYUV444(RGYFrameInfo *pOutputFrame, const RGYFrameInfo *pInputFrame, cudaStream_t stream);
    virtual void close() override;
};

class NVEncFilterParamNvvfxSuperRes;

class NVEncFilterParamResize : public NVEncFilterParam {
public:
    RGY_VPP_RESIZE_ALGO interp;
    RGY_VPP_RESIZE_ALGO nvvfxSubAlgo;
    std::shared_ptr<NVEncFilterParamNvvfxSuperRes> nvvfxSuperRes;
    NVEncFilterParamResize();
    virtual ~NVEncFilterParamResize();
    virtual tstring print() const override;
};

class NVEncFilterNvvfxSuperRes;

class NVEncFilterResize : public NVEncFilter {
public:
    NVEncFilterResize();
    virtual ~NVEncFilterResize();
    virtual RGY_ERR init(shared_ptr<NVEncFilterParam> pParam, shared_ptr<RGYLog> pPrintMes) override;
protected:
    virtual RGY_ERR run_filter(const RGYFrameInfo *pInputFrame, RGYFrameInfo **ppOutputFrames, int *pOutputFrameNum, cudaStream_t stream) override;
    RGY_ERR resizeNppiYV12(RGYFrameInfo *pOutputFrame, const RGYFrameInfo *pInputFrame);
    RGY_ERR resizeNppiYUV444(RGYFrameInfo *pOutputFrame, const RGYFrameInfo *pInputFrame);
    RGY_ERR initNvvfxFilter(NVEncFilterParamResize *param);
    RGY_ERR resizeNvvfxSuperRes(RGYFrameInfo *pOutputFrame, const RGYFrameInfo *pInputFrame);
    virtual void close() override;

    bool m_bInterlacedWarn;
    std::unique_ptr<CUMemBuf> m_weightSpline;
    RGY_VPP_RESIZE_ALGO m_weightSplineAlgo;
    std::unique_ptr<NVEncFilterNvvfxSuperRes> m_nvvfxSuperRes;
};


class NVEncFilterParamGaussDenoise : public NVEncFilterParam {
public:
    NppiMaskSize masksize;
    NVEncFilterParamGaussDenoise() : masksize(NPP_MASK_SIZE_3_X_3) {};
    virtual ~NVEncFilterParamGaussDenoise() {};
    virtual tstring print() const override;
};

class NVEncFilterDenoiseGauss : public NVEncFilter {
public:
    NVEncFilterDenoiseGauss();
    virtual ~NVEncFilterDenoiseGauss();
    virtual RGY_ERR init(shared_ptr<NVEncFilterParam> pParam, shared_ptr<RGYLog> pPrintMes) override;
protected:
    virtual RGY_ERR run_filter(const RGYFrameInfo *pInputFrame, RGYFrameInfo **ppOutputFrames, int *pOutputFrameNum, cudaStream_t stream) override;
    RGY_ERR denoisePlane(RGYFrameInfo *pOutputFrame, const RGYFrameInfo *pInputFrame);
    RGY_ERR denoiseFrame(RGYFrameInfo *pOutputFrame, const RGYFrameInfo *pInputFrame);
    virtual void close() override;
    bool m_bInterlacedWarn;
};

class NVEncFilterParamPad : public NVEncFilterParam {
public:
    VppPad pad;
    RGY_CSP encoderCsp;
    NVEncFilterParamPad() : pad(), encoderCsp(RGY_CSP_NA) {};
    virtual ~NVEncFilterParamPad() {};
    virtual tstring print() const override;
};

class NVEncFilterPad : public NVEncFilter {
public:
    NVEncFilterPad();
    virtual ~NVEncFilterPad();
    virtual RGY_ERR init(shared_ptr<NVEncFilterParam> pParam, shared_ptr<RGYLog> pPrintMes) override;
protected:
    virtual RGY_ERR run_filter(const RGYFrameInfo *pInputFrame, RGYFrameInfo **ppOutputFrames, int *pOutputFrameNum, cudaStream_t stream) override;

    RGY_ERR padPlane(RGYFrameInfo *pOutputFrame, const RGYFrameInfo *pInputFrame, int pad_color, const VppPad *pad, cudaStream_t stream);
    virtual void close() override;
};
