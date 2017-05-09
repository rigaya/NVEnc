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
#include "rgy_err.h"

enum RGY_INPUT_FMT {
    RGY_INPUT_FMT_UNKNWON = 0,
    RGY_INPUT_FMT_AUO = 0,
    RGY_INPUT_FMT_RAW,
    RGY_INPUT_FMT_Y4M,
    RGY_INPUT_FMT_AVI,
    RGY_INPUT_FMT_AVS,
    RGY_INPUT_FMT_VPY,
    RGY_INPUT_FMT_VPY_MT,
    RGY_INPUT_FMT_AVHW,
    RGY_INPUT_FMT_AVSW,
    RGY_INPUT_FMT_AVANY,
};

typedef struct InputVideoInfo {
    RGY_INPUT_FMT type;   //種類 (RGY_INPUT_FMT_xxx)
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
    RGY_CSP csp;       //入力色空間 (RGY_CSP_xxx)
    TCHAR *filename;         //入力ファイル名
    cudaVideoCodec codec;    //入力コーデック (デコード時使用)
    void *codecExtra;        //入力コーデックのヘッダー
    uint32_t codecExtraSize; //入力コーデックのヘッダーの大きさ
    int cuvidType;           //avcuvidリーダーを使用する際のモード (NV_ENC_AVCUVID_xxx)
    void *otherPrm;          //その他入力情報
} InputVideoInfo;

static_assert(std::is_pod<InputVideoInfo>::value == true, "InputVideoInfo is POD");

class NVEncBasicInput {
public:
    NVEncBasicInput();
    ~NVEncBasicInput();

    virtual void SetNVEncLogPtr(shared_ptr<CNVEncLog> pNVLog) {
        m_pPrintMes = pNVLog;
    }

    virtual RGY_ERR Init(InputVideoInfo *inputPrm, shared_ptr<EncodeStatus> pStatus);
    virtual RGY_ERR LoadNextFrame(void *dst, int dst_pitch);

#pragma warning(push)
#pragma warning(disable: 4100)
    //動画ストリームの1フレーム分のデータをbitstreamに追加する
    virtual RGY_ERR GetNextBitstream(vector<uint8_t>& bitstream, int64_t *dts) { return RGY_ERR_NONE; };

    //ストリームのヘッダ部分を取得する
    virtual RGY_ERR GetHeader(vector<uint8_t>& bitstream) { return RGY_ERR_NONE; };
#pragma warning(pop)

    virtual void Close();

    void SetTrimParam(const sTrimParam& trim) {
        m_sTrimParam = trim;
    }

    const sTrimParam *GetTrimParam() {
        return &m_sTrimParam;
    }

    void GetInputCropInfo(sInputCrop *cropInfo) {
        memcpy(cropInfo, &m_sInputCrop, sizeof(m_sInputCrop));
    }

    //入力ファイルに存在する音声のトラック数を返す
    virtual int GetAudioTrackCount() {
        return 0;
    }

    //入力ファイルに存在する字幕のトラック数を返す
    virtual int GetSubtitleTrackCount() {
        return 0;
    }

    //デコードするストリームの情報を取得する
    void GetDecParam(InputVideoInfo *decParam) {
        memcpy(decParam, &m_sDecParam, sizeof(m_sDecParam));
    }

    const TCHAR *GetInputMessage() {
        const TCHAR *mes = m_strInputInfo.c_str();
        return (mes) ? mes : _T("");
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

    //デコードを行うかどうかを返す
    bool inputCodecIsValid() {
        return m_nInputCodec != cudaVideoCodec_NumCodecs;
    }
protected:
    virtual void CreateInputInfo(const TCHAR *inputTypeName, const TCHAR *inputCSpName, const TCHAR *convSIMD, const TCHAR *outputCSpName, const InputVideoInfo *inputPrm);

    //trim listを参照し、動画の最大フレームインデックスを取得する
    int getVideoTrimMaxFramIdx() {
        if (m_sTrimParam.list.size() == 0) {
            return INT_MAX;
        }
        return m_sTrimParam.list[m_sTrimParam.list.size()-1].fin;
    }

    shared_ptr<EncodeStatus> m_pEncSatusInfo;
    FILE *m_fp = NULL;
    tstring m_strInputInfo;
    const ConvertCSP *m_sConvert = nullptr;
    shared_ptr<CNVEncLog> m_pPrintMes;  //ログ出力
    tstring m_strReaderName;    //読み込みの名前
    InputVideoInfo m_sDecParam; //デコード情報
    sInputCrop m_sInputCrop;

    sTrimParam m_sTrimParam;
    int m_nInputCodec;
};
