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
#pragma warning (push)
#pragma warning (disable: 4819)
#pragma warning (disable: 4201)
#include "nvEncodeAPI.h"
#pragma warning (pop)
#include <vector>
#include <list>
#include <string>
#include "CuvidDecode.h"
#include "NVEncDevice.h"
#include "NVEncUtil.h"
#include "NVEncParam.h"
#include "NVEncFilter.h"
#include "NVEncFilterSsim.h"
#include "NvHWEncoder.h"
#include "rgy_input.h"
#include "rgy_output.h"
#include "rgy_osdep.h"
#include "rgy_tchar.h"
#include "rgy_status.h"
#include "rgy_log.h"
#include "rgy_bitstream.h"
#include "rgy_frame_info.h"
#include "rgy_hdr10plus.h"
#include "rgy_device_usage.h"

class RGYTimecode;

using std::vector;

struct InputFrameBufInfo {
    std::unique_ptr<CUFrameBuf> cubuf; //入力フレームへのポインタと情報
    std::unique_ptr<void, handle_deleter> heTransferFin; //入力フレームに関連付けられたイベント、このフレームが不要になったらSetする
};

tstring get_codec_profile_name_from_guid(RGY_CODEC codec, const GUID& codecProfileGUID);
tstring get_codec_level_name(RGY_CODEC codec, int level);

class NVEncCore : public NVEncCtrl {
public:
    NVEncCore();
    virtual ~NVEncCore();

    virtual NVENCSTATUS Init(InEncodeVideoParam *inputParam);

    //エンコードを実行
    virtual NVENCSTATUS Encode();

    //エンコーダのClose・リソース開放
    virtual NVENCSTATUS Deinitialize();

    //エンコードの設定を取得
    virtual tstring GetEncodingParamsInfo(int output_level);

    //エンコードの設定を表示
    virtual void PrintEncodingParamsInfo(int output_level);

    //ユーザーからの中断を知らせるフラグへのポインタをセット
    void SetAbortFlagPointer(bool *abortFlag);
protected:
    bool encodeIsHighBitDepth(const InEncodeVideoParam *inputParam);

    //メインメソッド
    RGY_ERR CheckDynamicRCParams(std::vector<NVEncRCParam> &dynamicRC);

    //エンコーダが出力使用する色空間を入力パラメータをもとに取得
    RGY_CSP GetEncoderCSP(const InEncodeVideoParam *inputParam);
    RGY_CSP GetRawOutCSP(const InEncodeVideoParam *inputParam);

    //hwデコーダが出力する色空間を取得
    DeviceCodecCsp GetHWDecCodecCsp(const bool skipHWDecodeCheck, std::vector<std::unique_ptr<NVGPUInfo>>& gpuList);

    //チャプターファイルを読み込み
    NVENCSTATUS readChapterFile(const tstring& chapfile);

    //ログを初期化
    virtual NVENCSTATUS InitLog(const InEncodeVideoParam *inputParam);

    //エンコーダへの入力を初期化
    virtual NVENCSTATUS InitInput(InEncodeVideoParam *inputParam, DeviceCodecCsp& HWDecCodecCsp);

    //エンコーダへの入力を初期化
    virtual NVENCSTATUS InitOutput(InEncodeVideoParam *inputParam, NV_ENC_BUFFER_FORMAT encBufferFormat);

    //perfMonitorの初期化
    virtual NVENCSTATUS InitPerfMonitor(const InEncodeVideoParam *inputParam);

    //nvvfxを使用するかチェック
    bool useNVVFX(const InEncodeVideoParam *inputParam);

    //ngxを使用するかチェック
    bool useNVNGX(const InEncodeVideoParam *inputParam);

    //GPUListのGPUが必要なエンコードを行えるかチェック
    NVENCSTATUS CheckGPUListByEncoder(std::vector<std::unique_ptr<NVGPUInfo>> &gpuList, const InEncodeVideoParam *inputParam);

    //GPUを自動的に選択する
    NVENCSTATUS GPUAutoSelect(std::vector<std::unique_ptr<NVGPUInfo>> &gpuList, const InEncodeVideoParam *inputParam, const RGYDeviceUsageLockManager *devUsageLock);

    //デバイスの初期化
    virtual NVENCSTATUS InitDevice(std::vector<std::unique_ptr<NVGPUInfo>> &gpuList, const InEncodeVideoParam *inputParam);

    //inputParamからエンコーダに渡すパラメータを設定
    NVENCSTATUS SetInputParam(InEncodeVideoParam *inputParam);

    //デコーダインスタンスを作成
    NVENCSTATUS InitDecoder(const InEncodeVideoParam *inputParam);

    //デコーダインスタンスを作成
    RGY_ERR InitFilters(const InEncodeVideoParam *inputParam);

    //power throttoling設定の自動設定
    RGY_ERR InitPowerThrottoling(InEncodeVideoParam *inputParam);

    //チャプター読み込み等
    NVENCSTATUS InitChapters(const InEncodeVideoParam *inputParam);

    //入出力用バッファを確保
    RGY_ERR AllocateBufferInputHost(const VideoInfo *pInputInfo);
    RGY_ERR AllocateBufferEncoder(const uint32_t uInputWidth, const uint32_t uInputHeight, const NV_ENC_BUFFER_FORMAT inputFormat, const bool alphaChannel);
    RGY_ERR AllocateBufferRawOutput(const uint32_t uInputWidth, const uint32_t uInputHeight, const RGY_CSP csp);

    NVENCSTATUS NvEncEncodeFrame(EncodeBuffer *pEncodeBuffer, const int id, const int64_t timestamp, const int64_t duration, const int inputFrameId, const std::vector<std::shared_ptr<RGYFrameData>>& frameDataList);

    //エンコーダをフラッシュしてストリームを最後まで取り出す
    NVENCSTATUS FlushEncoder();

    //入出力バッファを解放
    NVENCSTATUS ReleaseIOBuffers();

    //フレームの出力と集計
    NVENCSTATUS ProcessOutput(const EncodeBuffer *pEncodeBuffer);

    //cuvidでのリサイズを有効にするか
    bool enableCuvidResize(const InEncodeVideoParam *inputParam);

    //vpp-rffが使用されているか
    bool VppRffEnabled();

    //vpp-afsのrffが使用されているか
    bool VppAfsRffAware();

    std::unique_ptr<NVGPUInfo>   m_dev;
#if ENABLE_AVSW_READER
    unique_ptr<CuvidDecode>      m_cuvidDec;              //デコード
#endif //#if ENABLE_AVSW_READER
    std::unique_ptr<RGYDeviceUsage> m_deviceUsage;

    bool                        *m_pAbortByUser;          //ユーザーからの中断指令

    CUctx_flags                  m_cudaSchedule;          //CUDAのスケジュール

    NV_ENC_INITIALIZE_PARAMS     m_stCreateEncodeParams;  //エンコーダの初期化パラメータ
    std::vector<NVEncRCParam>    m_dynamicRC;             //動的に変更するエンコーダのパラメータ
    int                          m_appliedDynamicRC;      //今適用されているパラメータ(未適用なら-1)

    int                          m_pipelineDepth;
    vector<InputFrameBufInfo>    m_inputHostBuffer;
    std::unique_ptr<CUFrameBuf>  m_outputFrameHostRaw;

    sTrimParam                    m_trimParam;
    std::unique_ptr<RGYPoolAVPacket> m_poolPkt;
    std::unique_ptr<RGYPoolAVFrame> m_poolFrame;
    shared_ptr<RGYInput>          m_pFileReader;           //動画読み込み
    vector<shared_ptr<RGYInput>>  m_AudioReaders;
    shared_ptr<RGYOutput>         m_pFileWriter;           //動画書き出し
    vector<shared_ptr<RGYOutput>> m_pFileWriterListAudio;
    shared_ptr<EncodeStatus>      m_pStatus;               //エンコードステータス管理
    shared_ptr<CPerfMonitor>      m_pPerfMonitor;
    NV_ENC_PIC_STRUCT             m_stPicStruct;           //エンコードフレーム情報(プログレッシブ/インタレ)
    NV_ENC_CONFIG                 m_stEncConfig;           //エンコード設定
#if ENABLE_AVSW_READER
    bool                          m_keyOnChapter;        //チャプター上にキーフレームを配置する
    vector<int>                   m_keyFile;             //キーフレームの指定
    vector<unique_ptr<AVChapter>> m_Chapters;            //ファイルから読み込んだチャプター
    bool                          m_hdr10plusMetadataCopy;
#endif //#if ENABLE_AVSW_READER
    unique_ptr<RGYTimecode>       m_timecode;
    unique_ptr<RGYHDR10Plus>      m_hdr10plus;
    unique_ptr<RGYHDRMetadata>    m_hdrseiIn;
    unique_ptr<RGYHDRMetadata>    m_hdrseiOut;
    unique_ptr<DOVIRpu>           m_dovirpu;
    bool                          m_dovirpuMetadataCopy;
    RGYDOVIProfile                m_doviProfile;
    std::unique_ptr<RGYTimestamp> m_encTimestamp;
    int64_t                       m_encodeFrameID;
    int                           m_videoIgnoreTimestampError;

    vector<unique_ptr<NVEncFilter>> m_vpFilters;
    shared_ptr<NVEncFilterParam>    m_pLastFilterParam;
#if ENABLE_SSIM
    unique_ptr<NVEncFilterSsim>  m_ssim;
#endif //#if ENABLE_SSIM

    unique_ptr<RGYListRef<RGYFrameDataQP>> m_qpTable;

    GUID                         m_stCodecGUID;           //出力コーデック
    int                          m_uEncWidth;             //出力縦解像度
    int                          m_uEncHeight;            //出力横解像度
    rgy_rational<int>            m_sar;                   //出力のsar比
    VideoVUIInfo                 m_encVUI;                //出力のVUI情報
    bool                         m_rgbAsYUV444;           //YUV444でRGB出力を行う

    int                          m_nProcSpeedLimit;       //処理速度制限 (0で制限なし)
    RGYAVSync                    m_nAVSyncMode;           //映像音声同期設定
    bool                         m_timestampPassThrough;  //timestampをそのまま転送する
    rgy_rational<int>            m_inputFps;              //入力フレームレート
    rgy_rational<int>            m_outputTimebase;        //出力のtimebase
    rgy_rational<int>            m_encFps;                //エンコードのフレームレート

    int                          m_encodeBufferCount;                 //入力バッファ数 (16以上、MAX_ENCODE_QUEUE以下)
    CNvQueue<EncodeBuffer>       m_EncodeBufferQueue;                 //エンコーダへのフレーム投入キュー
    EncodeOutputBuffer           m_stEOSOutputBfr;                    //エンコーダからの出力バッファ
    EncodeBuffer                 m_stEncodeBuffer[MAX_ENCODE_QUEUE];  //エンコーダへのフレームバッファ
};
