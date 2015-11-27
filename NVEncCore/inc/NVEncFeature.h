//  -----------------------------------------------------------------------------------------
//    NVEnc by rigaya
//  -----------------------------------------------------------------------------------------
//   ソースコードについて
//   ・無保証です。
//   ・本ソースコードを使用したことによるいかなる損害・トラブルについてrigayaは責任を負いません。
//   以上に了解して頂ける場合、本ソースコードの使用、複製、改変、再頒布を行って頂いて構いません。
//  -----------------------------------------------------------------------------------------

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
