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

#include <limits.h>
#include "nvEncodeAPI.h"
#include "NVEncoderPerf.h"
#include "NVEncCore.h"
#include "NVEncParam.h"

class NVEncFeature : public NVEncCore
{
public:
    NVEncFeature();
    ~NVEncFeature();

    //featureリストの作成を開始 (非同期)
    int createCacheAsync(int deviceID);

    //featureリストを取得 (取得できるまで待機)
    const std::vector<NVEncCodecFeature>& GetCachedNVEncCapability();
    
    //featureリストからHEVCのリストを取得 (HEVC非対応ならnullptr)
    static const NVEncCodecFeature *GetHEVCFeatures(const std::vector<NVEncCodecFeature>& codecFeatures);
    //featureリストからHEVCのリストを取得 (H.264対応ならnullptr)
    static const NVEncCodecFeature *GetH264Features(const std::vector<NVEncCodecFeature>& codecFeatures);

    //H.264が使用可能かどうかを取得 (取得できるまで待機)
    bool H264Available();
    //HEVCが使用可能かどうかを取得 (取得できるまで待機)
    bool HEVCAvailable();
protected:
    //createCacheを非同期実行するスレッド用
    static unsigned int __stdcall createCacheLoader(void *prm);
    //featureの取得を実行
    int createCache(int deviceID);

    int m_nTargetDeviceID;   //対象デバイスID
    NVEncCore *m_pNVEncCore; //NVEncCoreのインスタンス (スレッド終了時にdelete)

    HANDLE m_hThCreateCache;      //featureリスト作成用スレッドのハンドル
    HANDLE m_hEvCreateCache;      //featureリストの作成終了のイベント (ManualReset)
    HANDLE m_hEvCreateCodecCache; //codecのみのリスト作成終了のイベント (ManualReset)
    bool m_bH264; //H.264が使用可能かどうか (m_hEvCreateCodecCache後に有効)
    bool m_bHEVC; //HEVCが使用可能かどうか (m_hEvCreateCodecCache後に有効)

    //featureリスト
    //コーデックの有無はm_hEvCreateCodecCache後に有効
    //フルリストはm_hEvCreateCodecCache後に有効
    std::vector<NVEncCodecFeature> m_EncodeFeatures;
};
