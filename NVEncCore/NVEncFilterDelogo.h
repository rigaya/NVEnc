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

#include "NVEncFilter.h"
#include "logo.h"
#include "NVEncParam.h"

#define DELOGO_BLOCK_X  (32)
#define DELOGO_BLOCK_Y  (8)
#define DELOGO_BLOCK_LOOP_Y (4)

#define DELOGO_PARALLEL_FADE (33)
#define DELOGO_PRE_DIV_COUNT (4)
#define DELOGO_ADJMASK_DIV_COUNT (32)
#define DELOGO_ADJMASK_POW_BASE (1.1)

#define DELOGO_MASK_THRESHOLD_DEFAULT (1024)

#define LOGO_NR_MAX (4)
#define LOGO_FADE_AD_MAX	(10)
#define LOGO_FADE_AD_DEF	(7)

static const int LOGO_MULTI_ALIGN = 16;
static const int LOGO_MULTI_PADDING = 16;

struct LOGO_MULTI_DATA {
    int block_width;
    int block_height;
    int block_x;
    int block_y;
    int block_offset_x;
    int block_offset_y;
};

static int logo_multi_align(int n) {
    return (n + (LOGO_MULTI_ALIGN - 1)) & (~(LOGO_MULTI_ALIGN - 1));
}

static LOGO_MULTI_DATA get_logo_multi_data(int logo_width, int logo_height, int width, int height) {
    LOGO_MULTI_DATA data;
    data.block_width = logo_multi_align(logo_width + LOGO_MULTI_PADDING * 2);
    data.block_height = logo_multi_align(logo_height + LOGO_MULTI_PADDING * 2);
    if (data.block_width > data.block_height) {
        data.block_offset_x = 0;
        data.block_offset_y = (data.block_width - data.block_height) / 2;
        data.block_height = data.block_width;
    } else {
        data.block_offset_x = (data.block_height - data.block_width) / 2;
        data.block_offset_y = 0;
        data.block_width = data.block_height;
    }
    data.block_x = width / data.block_width;
    data.block_y = height / data.block_height;
    return data;
}

struct ProcessDataDelogo {
    unique_ptr<int16_t, aligned_malloc_deleter> pLogoPtr;
    unique_ptr<CUFrameBuf> pDevLogo;
    int    width;
    int    i_start;
    int    height;
    int    j_start;
    int    depth;
    short  offset[2];
    int    fade;
    shared_ptr<CUMemBuf> pBlockDepth;

    ~ProcessDataDelogo() {
        pLogoPtr.reset();
        pDevLogo.reset();
        pBlockDepth.reset();
    }
};

enum {
    LOGO_AUTO_SELECT_NOHIT   = -2,
    LOGO_AUTO_SELECT_INVALID = -1,
};

enum {
    LOGO__Y,
    LOGO_UV,
    LOGO__U,
    LOGO__V
};

typedef struct {
    int16_t x, y;
} int16x2_t;

struct LogoData {
    LOGO_HEADER header;
    vector<LOGO_PIXEL> logoPixel;

    LogoData() : header(), logoPixel() { memset(&header, 0, sizeof(header)); };
    ~LogoData() {};
};

typedef struct LOGO_SELECT_KEY {
    std::string key;
    char logoname[LOGO_MAX_NAME];
} LOGO_SELECT_KEY;

struct FadeArrayElem {
    int frameId;
    float fade;
    float adjFade;
    int nNR;
};

struct FadeArrayCache {
private:
    std::array<FadeArrayElem, 8> m_array;
public:
    FadeArrayElem& operator[](int iframe) {
        return m_array[std::max(iframe, 0) & 7];
    }
};

class DelogoSrcBuffer {
private:
    std::array<CUFrameBuf, 4> m_src;
public:
    DelogoSrcBuffer() : m_src() {};
    ~DelogoSrcBuffer() {
        clear();
    }
    cudaError_t alloc(const RGYFrameInfo& frame) {
        for (size_t i = 0; i < m_src.size(); i++) {
            m_src[i].frame = frame;
            auto sts = m_src[i].alloc();
            if (sts != cudaSuccess) {
                return sts;
            }
        }
        return cudaSuccess;
    }
    void clear() {
        for (size_t i = 0; i < m_src.size(); i++) {
            m_src[i].clear();
        }
    }
    CUFrameBuf& operator[](int iframe) {
        return m_src[std::max(iframe, 0) & 3];
    }
};

struct DelogoEvalStreams {
public:
    int evalBlocks;
    unique_ptr<cudaStream_t, cudastream_deleter> stEval;
    unique_ptr<cudaStream_t, cudastream_deleter> stEvalSub;
    unique_ptr<cudaEvent_t, cudaevent_deleter> heEval;
    unique_ptr<cudaEvent_t, cudaevent_deleter> heEvalCopyFin;
    DelogoEvalStreams() : evalBlocks(0), stEval(), stEvalSub(), heEval(), heEvalCopyFin() { }
    ~DelogoEvalStreams() {
        stEval.reset();
        stEvalSub.reset();
        heEval.reset();
        heEvalCopyFin.reset();
    }
    cudaError_t init(CUctx_flags cudaSchedule, bool noSubStream = false) {
        cudaError_t cudaerr = cudaSuccess;
        stEval = std::unique_ptr<cudaStream_t, cudastream_deleter>(new cudaStream_t(), cudastream_deleter());
        cudaerr = cudaStreamCreateWithFlags(stEval.get(), cudaStreamNonBlocking);
        if (cudaerr != cudaSuccess) return cudaerr;

        if (!noSubStream) {
            stEvalSub = std::unique_ptr<cudaStream_t, cudastream_deleter>(new cudaStream_t(), cudastream_deleter());
            cudaerr = cudaStreamCreateWithFlags(stEvalSub.get(), cudaStreamNonBlocking);
            if (cudaerr != cudaSuccess) return cudaerr;
        }
        const uint32_t cudaEventFlags = (cudaSchedule & CU_CTX_SCHED_BLOCKING_SYNC) ? cudaEventBlockingSync : 0;
        heEval = std::unique_ptr<cudaEvent_t, cudaevent_deleter>(new cudaEvent_t(), cudaevent_deleter());
        cudaerr = cudaEventCreateWithFlags(heEval.get(), cudaEventFlags | cudaEventDisableTiming);
        if (cudaerr != cudaSuccess) return cudaerr;

        heEvalCopyFin = std::unique_ptr<cudaEvent_t, cudaevent_deleter>(new cudaEvent_t(), cudaevent_deleter());
        cudaerr = cudaEventCreateWithFlags(heEvalCopyFin.get(), cudaEventFlags | cudaEventDisableTiming);
        if (cudaerr != cudaSuccess) return cudaerr;

        return cudaerr;
    }
};

class NVEncFilterParamDelogo : public NVEncFilterParam {
public:
    const TCHAR *inputFileName; //入力ファイル名
    const TCHAR *outputFileName; //出力ファイル名
    CUctx_flags cudaSchedule;
    VppDelogo delogo;

    NVEncFilterParamDelogo() : inputFileName(nullptr), outputFileName(nullptr), cudaSchedule(CU_CTX_SCHED_AUTO), delogo() {

    };
    virtual ~NVEncFilterParamDelogo() {};
    virtual tstring print() const override;
};

class NVEncFilterDelogo : public NVEncFilter {
public:
    NVEncFilterDelogo();
    virtual ~NVEncFilterDelogo();
    virtual RGY_ERR init(shared_ptr<NVEncFilterParam> pParam, shared_ptr<RGYLog> pPrintMes) override;
protected:
    virtual RGY_ERR run_filter(const RGYFrameInfo *pInputFrame, RGYFrameInfo **ppOutputFrames, int *pOutputFrameNum, cudaStream_t stream) override;
    virtual void close() override;

    int readLogoFile(const std::shared_ptr<NVEncFilterParamDelogo> pDelogoParam);
    int getLogoIdx(const std::string& logoName);
    int selectLogo(const tstring& selectStr, const tstring& inputFilename);
    std::string logoNameList();

    RGY_ERR delogoY(RGYFrameInfo *pFrame, float fade);
    RGY_ERR delogoUV(RGYFrameInfo *pFrame, float fade);
    RGY_ERR logoNR(RGYFrameInfo *pFrame, int nr_value);

    RGY_ERR createLogoMask();
    RGY_ERR createLogoMask(int maskThreshold);
    RGY_ERR createNRMask(CUFrameBuf *ptr_mask_nr, const CUFrameBuf *ptr_mask, int nr_value);
    RGY_ERR calcAutoFadeNR(int& auto_nr, float& auto_fade, const RGYFrameInfo *pFrame);
    RGY_ERR calcAutoFadeNRFrame(int& auto_nr, float& auto_fade, const RGYFrameInfo *pFrame);
    RGY_ERR createAdjustedMask(const RGYFrameInfo *frame_logo);
    RGY_ERR runDelogoYMultiFade(const RGYFrameInfo *frame_logo, const bool multi_src, const int nr_value, const float *fade, const int fade_n, cudaStream_t stream);
    RGY_ERR runSmooth(const int smooth_n, const int nr_value, const int nr_area, cudaStream_t stream);
    RGY_ERR prewittEvaluateRun(const bool store_pixel_result, const CUFrameBuf *target, const CUFrameBuf *mask, const int nr_value, const int eval_n, DelogoEvalStreams& evalst);
    RGY_ERR autoFadeCoef2Run(const bool store_pixel_result, const RGYFrameInfo *frame_logo, const int nr_value, const int nr_area, const float *ptrDevFadeDepth, int calc_n, DelogoEvalStreams& evalst);
    RGY_ERR autoFadeCoef2Collect(std::vector<float>& eval, const int nr_value, cudaEvent_t eventCopyFin);
    RGY_ERR autoFadeLS2(float& auto_fade, const int nr_value);
    RGY_ERR logAutoFadeNR();

    tstring m_LogoFilePath;
    int m_nLogoIdx;
    vector<LogoData> m_sLogoDataList;
    ProcessDataDelogo m_sProcessData[4];

    DelogoSrcBuffer m_src;
    unique_ptr<CUFrameBuf> m_mask;           //評価用Mask(Original)
    unique_ptr<CUFrameBuf> m_maskAdjusted;   //評価用Mask(Frame毎の調整後)
    unique_ptr<CUFrameBuf> m_maskNR;         //NR用Mask
    unique_ptr<CUFrameBuf> m_maskNRAdjusted; //NR用Mask(Frame毎の調整後)
    int m_maskValidCount;
    int m_maskThreshold;

    std::array<unique_ptr<CUFrameBuf>, LOGO_NR_MAX+1> m_bufDelogo;   // ロゴ除去したもの
    std::array<unique_ptr<CUFrameBuf>, LOGO_NR_MAX+1> m_bufDelogoNR; // m_bufDelogoについてロゴ除去したもの
    std::array<unique_ptr<CUFrameBuf>, LOGO_NR_MAX+1> m_bufEval;  // 評価用のバッファ
    std::array<DelogoEvalStreams, LOGO_NR_MAX+1> m_evalStream;  // 評価用のバッファ
    unique_ptr<CUFrameBuf> m_adjMaskMinIndex;
    unique_ptr<CUFrameBuf> m_adjMaskThresholdTest;
    unique_ptr<CUFrameBuf> m_NRProcTemp;
    std::array<CUMemBufPair, LOGO_NR_MAX+1> m_evalCounter;
    CUMemBufPair m_createLogoMaskValidMaskCount;
    CUMemBufPair m_adjMaskEachFadeCount;
    CUMemBufPair m_adjMaskMinResAndValidMaskCount;
    CUMemBufPair m_adjMask2ValidMaskCount;
    unique_ptr<void, cudadevice_deleter> m_adjMask2TargetCount;
    DelogoEvalStreams m_adjMaskStream;
    unique_ptr<void, cudadevice_deleter> m_smoothKernel;
    CUMemBufPair m_fadeValueAdjust;
    CUMemBufPair m_fadeValueParallel;
    CUMemBufPair m_fadeValueTemp;

    FadeArrayCache m_fadeArray; // 近傍のFrameのFade値
    int m_frameIn;
    int m_frameOut;
    int m_yDepth;
    bool m_EnableAutoNR;
    tstring m_logPath;

    shared_ptr<CUMemBuf> m_Depth;
    vector<float> m_DepthHost;
    unique_ptr<FILE, fp_deleter> m_fpDepth;
    int m_nFramesProcessed;
};
