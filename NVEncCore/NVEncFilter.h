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
#include "rgy_cuda_util.h"
#include "rgy_frame.h"
#include "NVEncUtil.h"
#include "NVEncParam.h"
#include "NVEncFrameInfo.h"
#include "rgy_osdep.h"
#include "rgy_tchar.h"
#include "rgy_log.h"
#include "convert_csp.h"

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

static NppiSize nppisize(const FrameInfo *pFrame) {
    NppiSize size;
    size.width = pFrame->width;
    size.height = pFrame->height;
    return size;
}

static NppiRect nppiroi(const FrameInfo *pFrame) {
    NppiRect rect;
    rect.x = 0;
    rect.y = 0;
    rect.width = pFrame->width;
    rect.height = pFrame->height;
    return rect;
}

class NVEncFilterParam {
public:
    FrameInfo frameIn;
    FrameInfo frameOut;
    rgy_rational<int> baseFps;
    bool bOutOverwrite;

    NVEncFilterParam() : frameIn(), frameOut(), baseFps(), bOutOverwrite(false) {};
    virtual tstring print() const = 0;
    virtual ~NVEncFilterParam() {};
};

enum FILTER_PATHTHROUGH_FRAMEINFO : uint32_t {
    FILTER_PATHTHROUGH_NONE      = 0x00u,
    FILTER_PATHTHROUGH_TIMESTAMP = 0x01u,
    FILTER_PATHTHROUGH_FLAGS     = 0x02u,
    FILTER_PATHTHROUGH_PICSTRUCT = 0x04u,
    FILTER_PATHTHROUGH_DATA      = 0x08u,

    FILTER_PATHTHROUGH_ALL       = 0x0fu,
};

static FILTER_PATHTHROUGH_FRAMEINFO operator|(FILTER_PATHTHROUGH_FRAMEINFO a, FILTER_PATHTHROUGH_FRAMEINFO b) {
    return (FILTER_PATHTHROUGH_FRAMEINFO)((uint32_t)a | (uint32_t)b);
}

static FILTER_PATHTHROUGH_FRAMEINFO operator|=(FILTER_PATHTHROUGH_FRAMEINFO& a, FILTER_PATHTHROUGH_FRAMEINFO b) {
    a = a | b;
    return a;
}

static FILTER_PATHTHROUGH_FRAMEINFO operator&(FILTER_PATHTHROUGH_FRAMEINFO a, FILTER_PATHTHROUGH_FRAMEINFO b) {
    return (FILTER_PATHTHROUGH_FRAMEINFO)((uint32_t)a & (uint32_t)b);
}

static FILTER_PATHTHROUGH_FRAMEINFO operator&=(FILTER_PATHTHROUGH_FRAMEINFO& a, FILTER_PATHTHROUGH_FRAMEINFO b) {
    a = a & b;
    return a;
}

static FILTER_PATHTHROUGH_FRAMEINFO operator~(FILTER_PATHTHROUGH_FRAMEINFO a) {
    return (FILTER_PATHTHROUGH_FRAMEINFO)(~((uint32_t)a));
}

class NVEncFilter {
public:
    NVEncFilter();
    virtual ~NVEncFilter();
    tstring name() {
        return m_sFilterName;
    }
    virtual RGY_ERR init(shared_ptr<NVEncFilterParam> pParam, shared_ptr<RGYLog> pPrintMes) = 0;
    RGY_ERR filter(FrameInfo *pInputFrame, FrameInfo **ppOutputFrames, int *pOutputFrameNum, cudaStream_t stream);
    const tstring GetInputMessage() {
        return m_sFilterInfo;
    }
    const NVEncFilterParam *GetFilterParam() {
        return m_pParam.get();
    }
    void CheckPerformance(bool flag);
    double GetAvgTimeElapsed();
    virtual RGY_ERR addStreamPacket(AVPacket *pkt) { UNREFERENCED_PARAMETER(pkt); return RGY_ERR_UNSUPPORTED; };
    virtual int targetTrackIdx() { return 0; };
protected:
    RGY_ERR filter_as_interlaced_pair(const FrameInfo *pInputFrame, FrameInfo *pOutputFrame, cudaStream_t stream);
    virtual RGY_ERR run_filter(const FrameInfo *pInputFrame, FrameInfo **ppOutputFrames, int *pOutputFrameNum, cudaStream_t stream) = 0;
    virtual void close() = 0;

    void setFilterInfo(const tstring &info) {
        m_sFilterInfo = info;
        AddMessage(RGY_LOG_DEBUG, info);
    }
    void AddMessage(int log_level, const tstring& str) {
        if (m_pPrintMes == nullptr || log_level < m_pPrintMes->getLogLevel()) {
            return;
        }
        auto lines = split(str, _T("\n"));
        for (const auto& line : lines) {
            if (line[0] != _T('\0')) {
                m_pPrintMes->write(log_level, (m_sFilterName + _T(": ") + line + _T("\n")).c_str());
            }
        }
    }
    void AddMessage(int log_level, const TCHAR *format, ... ) {
        if (m_pPrintMes == nullptr || log_level < m_pPrintMes->getLogLevel()) {
            return;
        }

        va_list args;
        va_start(args, format);
        int len = _vsctprintf(format, args) + 1; // _vscprintf doesn't count terminating '\0'
        tstring buffer;
        buffer.resize(len, _T('\0'));
        _vstprintf_s(&buffer[0], len, format, args);
        va_end(args);
        AddMessage(log_level, buffer);
    }
    cudaError_t AllocFrameBuf(const FrameInfo& frame, int frames);

    tstring m_sFilterName;
    tstring m_sFilterInfo;
    shared_ptr<RGYLog> m_pPrintMes;  //ログ出力
    vector<unique_ptr<CUFrameBuf>> m_pFrameBuf;
    int m_nFrameIdx;
    unique_ptr<CUFrameBuf> m_pFieldPairIn;
    unique_ptr<CUFrameBuf> m_pFieldPairOut;
    shared_ptr<NVEncFilterParam> m_pParam;
    FILTER_PATHTHROUGH_FRAMEINFO m_nPathThrough;
private:
    bool m_bCheckPerformance;
    unique_ptr<cudaEvent_t, cudaevent_deleter> m_peFilterStart;
    unique_ptr<cudaEvent_t, cudaevent_deleter> m_peFilterFin;
    double m_dFilterTimeMs;
    int m_nFilterRunCount;
};

class NVEncFilterParamCrop : public NVEncFilterParam {
public:
    sInputCrop crop;

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
    virtual RGY_ERR run_filter(const FrameInfo *pInputFrame, FrameInfo **ppOutputFrames, int *pOutputFrameNum, cudaStream_t stream) override;
    RGY_ERR convertYBitDepth(FrameInfo *pOutputFrame, const FrameInfo *pInputFrame, cudaStream_t stream);
    RGY_ERR convertCspFromNV12(FrameInfo *pOutputFrame, const FrameInfo *pInputFrame, cudaStream_t stream);
    RGY_ERR convertCspFromYV12(FrameInfo *pOutputFrame, const FrameInfo *pInputFrame, cudaStream_t stream);
    RGY_ERR convertCspFromNV16(FrameInfo *pOutputFrame, const FrameInfo *pInputFrame, cudaStream_t stream);
    RGY_ERR convertCspFromRGB(FrameInfo *pOutputFrame, const FrameInfo *pInputFrame, cudaStream_t stream);
    RGY_ERR convertCspFromYUV444(FrameInfo *pOutputFrame, const FrameInfo *pInputFrame, cudaStream_t stream);
    virtual void close() override;
};

class NVEncFilterParamResize : public NVEncFilterParam {
public:
    int interp;
    NVEncFilterParamResize() : interp(RESIZE_CUDA_SPLINE36) {}
    virtual ~NVEncFilterParamResize() {};
    virtual tstring print() const override;
};

class NVEncFilterResize : public NVEncFilter {
public:
    NVEncFilterResize();
    virtual ~NVEncFilterResize();
    virtual RGY_ERR init(shared_ptr<NVEncFilterParam> pParam, shared_ptr<RGYLog> pPrintMes) override;
protected:
    virtual RGY_ERR run_filter(const FrameInfo *pInputFrame, FrameInfo **ppOutputFrames, int *pOutputFrameNum, cudaStream_t stream) override;
    RGY_ERR resizeNppiYV12(FrameInfo *pOutputFrame, const FrameInfo *pInputFrame);
    RGY_ERR resizeNppiYUV444(FrameInfo *pOutputFrame, const FrameInfo *pInputFrame);
    virtual void close() override;

    bool m_bInterlacedWarn;
    CUMemBuf m_weightSpline;
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
    virtual RGY_ERR run_filter(const FrameInfo *pInputFrame, FrameInfo **ppOutputFrames, int *pOutputFrameNum, cudaStream_t stream) override;
    RGY_ERR denoiseYV12(FrameInfo *pOutputFrame, const FrameInfo *pInputFrame);
    RGY_ERR denoiseYUV444(FrameInfo *pOutputFrame, const FrameInfo *pInputFrame);
    virtual void close() override;
    bool m_bInterlacedWarn;
};

class NVEncFilterParamPad : public NVEncFilterParam {
public:
    VppPad pad;
    NVEncFilterParamPad() : pad() {};
    virtual ~NVEncFilterParamPad() {};
    virtual tstring print() const override;
};

class NVEncFilterPad : public NVEncFilter {
public:
    NVEncFilterPad();
    virtual ~NVEncFilterPad();
    virtual RGY_ERR init(shared_ptr<NVEncFilterParam> pParam, shared_ptr<RGYLog> pPrintMes) override;
protected:
    virtual RGY_ERR run_filter(const FrameInfo *pInputFrame, FrameInfo **ppOutputFrames, int *pOutputFrameNum, cudaStream_t stream) override;

    RGY_ERR padPlane(FrameInfo *pOutputFrame, const FrameInfo *pInputFrame, int pad_color, const VppPad *pad);
    virtual void close() override;
};
