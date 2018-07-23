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

#define DELOGO_PARALLEL_FADE (11)
#define DELOGO_PARALLEL_FADE_MUL (1.25f)
#define DELOGO_PRE_DIV_COUNT (4)
#define DELOGO_ADJMASK_DIV_COUNT (32)
#define DELOGO_ADJMASK_POW_BASE (1.1)

#define DELOGO_MASK_THRESHOLD_DEFAULT (1024)

#define LOGO_NR_MAX (4)
#define LOGO_FADE_AD_MAX	(10)
#define LOGO_FADE_AD_DEF	(7)

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

    ~ProcessDataDelogo() {
        pLogoPtr.reset();
        pDevLogo.reset();
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

typedef struct LogoData {
    LOGO_HEADER header;
    vector<LOGO_PIXEL> logoPixel;
} LogoData;

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

class NVEncFilterParamDelogo : public NVEncFilterParam {
public:
    const TCHAR *inputFileName; //入力ファイル名
    VppDelogo delogo;

    NVEncFilterParamDelogo() : inputFileName(nullptr), delogo() {

    };
    virtual ~NVEncFilterParamDelogo() {};
};

class NVEncFilterDelogo : public NVEncFilter {
public:
    NVEncFilterDelogo();
    virtual ~NVEncFilterDelogo();
    virtual NVENCSTATUS init(shared_ptr<NVEncFilterParam> pParam, shared_ptr<RGYLog> pPrintMes) override;
protected:
    virtual NVENCSTATUS run_filter(const FrameInfo *pInputFrame, FrameInfo **ppOutputFrames, int *pOutputFrameNum) override;
    virtual void close() override;

    int readLogoFile(const std::shared_ptr<NVEncFilterParamDelogo> pDelogoParam);
    int getLogoIdx(const std::string& logoName);
    int selectLogo(const tstring& selectStr, const tstring& inputFilename);
    std::string logoNameList();

    NVENCSTATUS delogoY(FrameInfo *pFrame, float fade);
    NVENCSTATUS delogoUV(FrameInfo *pFrame, float fade);
    NVENCSTATUS logoNR(FrameInfo *pFrame, int nr_value);

    NVENCSTATUS createLogoMask();
    NVENCSTATUS createLogoMask(int maskThreshold);
    NVENCSTATUS createNRMask(CUFrameBuf *ptr_mask_nr, const CUFrameBuf *ptr_mask, int nr_value);
    NVENCSTATUS calcAutoFadeNR(int& auto_nr, float& auto_fade, const FrameInfo *pFrame);
    NVENCSTATUS calcAutoFadeNRFrame(int& auto_nr, float& auto_fade, const FrameInfo *pFrame);
    NVENCSTATUS createAdjustedMask(const FrameInfo *frame_logo);
    NVENCSTATUS runDelogoYMultiFade(const FrameInfo *frame_logo, const bool multi_src, const int nr_value, const float *fade, const int fade_n);
    NVENCSTATUS runSmooth(const int smooth_n, const int nr_value, const int nr_area);
    NVENCSTATUS prewittEvaluate(std::vector<float>& eval, const bool store_pixel_result, const CUFrameBuf *target, const CUFrameBuf *mask, const int nr_value);
    NVENCSTATUS calcAutoFadeCoef2(std::vector<float>& eval, const bool store_pixel_result, const FrameInfo *frame_logo, const int nr_value, const int nr_area, const float *ptrDevFadeDepth);
    NVENCSTATUS calcAutoFadeLS2(float& auto_fade, const FrameInfo *frame_logo, const int nr_value, const int nr_area);
    NVENCSTATUS calcAutoFade4(float& auto_fade, const FrameInfo *frame_logo, const int nr_value, const int nr_area);
    NVENCSTATUS logAutoFadeNR();

    tstring m_LogoFilePath;
    int m_nLogoIdx;
    vector<LogoData> m_sLogoDataList;
    ProcessDataDelogo m_sProcessData[4];

    unique_ptr<CUFrameBuf> m_mask;           //評価用Mask(Original)
    unique_ptr<CUFrameBuf> m_maskAdjusted;   //評価用Mask(Frame毎の調整後)
    unique_ptr<CUFrameBuf> m_maskNR;         //NR用Mask
    unique_ptr<CUFrameBuf> m_maskNRAdjusted; //NR用Mask(Frame毎の調整後)
    int m_maskValidCount;
    int m_maskThreshold;

    std::array<unique_ptr<CUFrameBuf>, LOGO_NR_MAX+1> m_bufDelogo;   // ロゴ除去したもの
    std::array<unique_ptr<CUFrameBuf>, LOGO_NR_MAX+1> m_bufDelogoNR; // m_bufDelogoについてロゴ除去したもの
    std::array<unique_ptr<CUFrameBuf>, LOGO_NR_MAX+1> m_bufEval;  // 評価用のバッファ
    unique_ptr<CUFrameBuf> m_adjMaskMinIndex;
    unique_ptr<CUFrameBuf> m_adjMaskThresholdTest;
    unique_ptr<CUFrameBuf> m_NRProcTemp;
    std::array<CUMemBufPair, LOGO_NR_MAX+1> m_evalCounter;
    CUMemBufPair m_createLogoMaskValidMaskCount;
    CUMemBufPair m_adjMaskEachFadeCount;
    CUMemBufPair m_adjMaskMinResAndValidMaskCount;
    CUMemBufPair m_adjMask2ValidMaskCount;
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
};
