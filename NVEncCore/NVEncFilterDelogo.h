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

typedef struct LogoData {
    LOGO_HEADER header;
    vector<LOGO_PIXEL> logoPixel;
} LogoData;

typedef struct LOGO_SELECT_KEY {
    std::string key;
    char logoname[LOGO_MAX_NAME];
} LOGO_SELECT_KEY;

class NVEncFilterParamDelogo : public NVEncFilterParam {
public:
    const TCHAR *inputFileName; //入力ファイル名
    const TCHAR *logoFilePath;  //ロゴファイル名
    const TCHAR *logoSelect;    //ロゴの名前
    short posX, posY; //位置オフセット
    short depth;      //透明度深度
    short Y, Cb, Cr;  //(輝度・色差)オフセット
    int mode;

    NVEncFilterParamDelogo() : inputFileName(nullptr), logoFilePath(nullptr), logoSelect(nullptr),
        posX(0), posY(0), depth(128), Y(0), Cb(0), Cr(0), mode(DELOGO_MODE_REMOVE) {

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

    int readLogoFile();
    int getLogoIdx(const std::string& logoName);
    int selectLogo(const TCHAR *selectStr);
    std::string logoNameList();

    NVENCSTATUS delogoY(FrameInfo *pFrame);
    NVENCSTATUS delogoUV(FrameInfo *pFrame);

    int m_nLogoIdx;
    vector<LogoData> m_sLogoDataList;
    ProcessDataDelogo m_sProcessData[4];
};
