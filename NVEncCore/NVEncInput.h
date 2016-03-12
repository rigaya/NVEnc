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

#include <stdio.h>
#include <tchar.h>
#include <string>
#include <vector>
#include <chrono>
#include "NVEncUtil.h"
#include "NVEncStatus.h"
#include "NVEncVersion.h"
#include "ConvertCsp.h"
#include "NVEncLog.h"
#include "NVEncParam.h"
#include <nvcuvid.h>

enum {
    NV_ENC_INPUT_UNKNWON = 0,
    NV_ENC_INPUT_AUO = 0,
    NV_ENC_INPUT_RAW,
    NV_ENC_INPUT_Y4M,
    NV_ENC_INPUT_AVI,
    NV_ENC_INPUT_AVS,
    NV_ENC_INPUT_VPY,
    NV_ENC_INPUT_VPY_MT,
    NV_ENC_INPUT_AVCUVID,
};

typedef struct InputVideoInfo {
    int type;             //種類 (NV_ENC_INPUT_xxx)
    uint32_t width;       //横解像度
    uint32_t height;      //縦解像度
    uint32_t src_pitch;
    uint32_t codedWidth;
    uint32_t codedHeight;
    uint32_t dstWidth;    //出力解像度
    uint32_t dstHeight;   //出力解像度
    int scale;            //フレームレート (分子)
    int rate;             //フレームレート (分母)
    sInputCrop crop;      //入力時切り落とし
    int sar[2];           //par
    NV_ENC_CSP csp;       //入力色空間 (NV_ENC_CSP_xxx)
    TCHAR *filename;      //入力ファイル名
    cudaVideoCodec codec; //入力コーデック (デコード時使用)
    void *codecExtra;
    uint32_t codecExtraSize;
    void *otherPrm; //その他入力情報
} InputVideoInfo;

class NVEncBasicInput {
public:
    NVEncBasicInput();
    ~NVEncBasicInput();

    virtual void SetNVEncLogPtr(shared_ptr<CNVEncLog> pNVLog) {
        m_pPrintMes = pNVLog;
    }

    virtual int Init(InputVideoInfo *inputPrm, shared_ptr<EncodeStatus> pStatus);
    virtual int LoadNextFrame(void *dst, int dst_pitch);

#pragma warning(push)
#pragma warning(disable: 4100)
    //動画ストリームの1フレーム分のデータをbitstreamに追加する
    virtual int GetNextBitstream(vector<uint8_t>& bitstream, int64_t *dts) { return 0; };

    //ストリームのヘッダ部分を取得する
    virtual int GetHeader(vector<uint8_t>& bitstream) { return 0; };
#pragma warning(pop)

    virtual void Close();

    tstring getInputMes() {
        return m_strInputInfo;
    }
    void AddMessage(int log_level, const tstring& str) {
        if (m_pPrintMes == nullptr || log_level < m_pPrintMes->getLogLevel()) {
            return;
        }
        auto lines = split(str, _T("\n"));
        for (const auto& line : lines) {
            if (line[0] != _T('\0')) {
                m_pPrintMes->write(log_level, (m_strReaderName + _T(": ") + line + _T("\n")).c_str());
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

protected:
    virtual void CreateInputInfo(const TCHAR *inputTypeName, const TCHAR *inputCSpName, const TCHAR *convSIMD, const TCHAR *outputCSpName, const InputVideoInfo *inputPrm);

    shared_ptr<EncodeStatus> m_pEncSatusInfo;
    FILE *m_fp = NULL;
    tstring m_strInputInfo;
    const ConvertCSP *m_pConvCSPInfo = nullptr;
    shared_ptr<CNVEncLog> m_pPrintMes;  //ログ出力
    tstring m_strReaderName;    //読み込みの名前
    InputVideoInfo m_sDecParam; //デコード情報

    sTrimParam m_sTrimParam;
};
