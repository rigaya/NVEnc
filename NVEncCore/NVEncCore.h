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

#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <Windows.h>
#include <stdint.h>
#include "nvEncodeAPI.h"
#include <tchar.h>
#include <vector>
#include <list>
#include <string>
#include "rgy_input.h"
#include "rgy_output.h"
#include "rgy_status.h"
#include "rgy_log.h"
#include "rgy_bitstream.h"
#include "NVEncUtil.h"
#include "NVEncParam.h"
#include "CuvidDecode.h"
#include "NVEncFilter.h"
#include "NVEncFrameInfo.h"

using std::vector;

static const int MAX_DECODE_FRAMES = 16;

static const int BITSTREAM_BUFFER_SIZE =  4 * 1024 * 1024;
static const int OUTPUT_BUF_SIZE       = 16 * 1024 * 1024;

static const int DEFAULT_GOP_LENGTH  = 0;
static const int DEFAULT_B_FRAMES_H264 = 3;
static const int DEFAULT_B_FRAMES_HEVC = 0;
static const int DEFAULT_REF_FRAMES  = 3;
static const int DEFAULT_NUM_SLICES  = 1;
static const int DEFAUTL_QP_I        = 20;
static const int DEFAULT_QP_P        = 23;
static const int DEFAULT_QP_B        = 25;
static const int DEFAULT_AVG_BITRATE = 7500000;
static const int DEFAULT_MAX_BITRATE = 17500000;
static const int DEFAULT_OUTPUT_BUF  = 8;
static const int DEFAULT_LOOKAHEAD   = 16;
static const int DEFAULT_IGNORE_DECODE_ERROR = 10;

static const int DEFAULT_CUDA_SCHEDULE = CU_CTX_SCHED_AUTO;

static const uint32_t PIPELINE_DEPTH = 4;
static const int MAX_FILTER_OUTPUT = 2;

#ifdef _M_IX86
static const TCHAR *NVENCODE_API_DLL = _T("nvEncodeAPI.dll");
#else
static const TCHAR *NVENCODE_API_DLL = _T("nvEncodeAPI64.dll");
#endif

#define INIT_CONFIG(configStruct, type) { memset(&(configStruct), 0, sizeof(configStruct)); (configStruct).version = type##_VER;}
#ifndef SET_VER
#define SET_VER(configStruct, type) { (configStruct).version = type##_VER; }
#endif

typedef NVENCSTATUS (NVENCAPI *MYPROC)(NV_ENCODE_API_FUNCTION_LIST*);

bool check_if_nvcuda_dll_available();

struct InputFrameBufInfo {
    FrameInfo frameInfo; //入力フレームへのポインタと情報
    std::unique_ptr<void, handle_deleter> heTransferFin; //入力フレームに関連付けられたイベント、このフレームが不要になったらSetする
};

class NVEncoderGPUInfo
{
public:
    NVEncoderGPUInfo(int deviceId, bool getFeatures);
    ~NVEncoderGPUInfo();
    const std::list<NVGPUInfo> getGPUList() {
        return GPUList;
    }
private:
    std::list<NVGPUInfo> GPUList;
};

struct InEncodeVideoParam {
    VideoInfo input;              //入力する動画の情報
    tstring inputFilename;        //入力ファイル名
    tstring outputFilename;       //出力ファイル名
    tstring sAVMuxOutputFormat;   //出力フォーマット
    int preset;                   //出力プリセット
    int deviceID;                 //使用するGPUのID
    int nHWDecType;               //
    int par[2];                   //使用されていません
    NV_ENC_CONFIG encConfig;      //エンコード設定
    int codec;                    //出力コーデック
    int bluray;                   //bluray出力
    int yuv444;                   //YUV444出力
    int lossless;                 //ロスレス出力
    std::string sMaxCll;
    std::string sMasterDisplay;
    tstring logfile;              //ログ出力先
    int loglevel;                 //ログ出力レベル
    int nOutputBufSizeMB;         //出力バッファサイズ
    tstring sFramePosListLog;     //framePosList出力先
    float fSeekSec;               //指定された秒数分先頭を飛ばす
    int nSubtitleSelectCount;
    int *pSubtitleSelect;
    int nAudioSourceCount;
    TCHAR **ppAudioSourceList;
    int nAudioSelectCount; //pAudioSelectの数
    sAudioSelect **ppAudioSelectList;
    int nAudioResampler;
    int nAVDemuxAnalyzeSec;
    int nAVMux;                       //RGY_MUX_xxx
    int nVideoTrack;
    int nVideoStreamId;
    int nTrimCount;
    sTrim *pTrimList;
    bool bCopyChapter;
    int nOutputThread;
    int nAudioThread;
    int nInputThread;
    int nAudioIgnoreDecodeError;
    muxOptList *pMuxOpt;
    tstring sChapterFile;
    TCHAR *pMuxVidTsLogFile;
    TCHAR *pAVInputFormat;
    RGYAVSync nAVSyncMode;     //avsyncの方法 (NV_AVSYNC_xxx)
    int nProcSpeedLimit;      //処理速度制限 (0で制限なし)
    VppParam vpp;                 //vpp
    int nWeightP;
    int64_t nPerfMonitorSelect;
    int64_t nPerfMonitorSelectMatplot;
    int     nPerfMonitorInterval;
    int     nCudaSchedule;
    void *pPrivatePrm;

    InEncodeVideoParam();
};

class NVEncCore {
public:
    NVEncCore();
    virtual ~NVEncCore();

    //デフォルトのエンコード設定を取得 (H.264用に設定済み)
    static NV_ENC_CONFIG DefaultParam();

    //デフォルトのH.264/AVC用の設定
    static NV_ENC_CODEC_CONFIG DefaultParamH264();

    //デフォルトのH.265/EVC用の設定
    static NV_ENC_CODEC_CONFIG DefaultParamHEVC();

    //CUDAインターフェース・デバイスの初期化
    virtual NVENCSTATUS Initialize(InEncodeVideoParam *inputParam);

    //デバイスの初期化
    virtual NVENCSTATUS InitDevice(const InEncodeVideoParam *inputParam);

    //エンコードの初期化 (デバイスの初期化(Initialize())後に行うこと)
    virtual NVENCSTATUS InitEncode(InEncodeVideoParam *inputParam);

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
    //メインメソッド
    NVENCSTATUS SetEncodeCodecList(void *encode);

    //エンコーダが出力使用する色空間を入力パラメータをもとに取得
    RGY_CSP GetEncoderCSP(const InEncodeVideoParam *inputParam);
    
    //既定の出力先に情報をメッセージを出力
    virtual void PrintMes(int logLevel, const TCHAR *format, ...);
    
    //特定の関数でのエラーを表示
    void NVPrintFuncError(const TCHAR *funcName, NVENCSTATUS nvStatus);

    //特定の関数でのエラーを表示
    void NVPrintFuncError(const TCHAR *funcName, CUresult code);

    //チャプターファイルを読み込み
    NVENCSTATUS readChapterFile(const tstring& chapfile);

    //エンコーダへの入力を初期化
    virtual NVENCSTATUS InitInput(InEncodeVideoParam *inputParam);

    //エンコーダへの入力を初期化
    virtual NVENCSTATUS InitOutput(InEncodeVideoParam *inputParam, NV_ENC_BUFFER_FORMAT encBufferFormat);

    //ログを初期化
    virtual NVENCSTATUS InitLog(const InEncodeVideoParam *inputParam);

    //GPUListのGPUが必要なエンコードを行えるかチェック
    NVENCSTATUS CheckGPUListByEncoder(const InEncodeVideoParam *inputParam);

    //GPUを自動的に選択する
    NVENCSTATUS GPUAutoSelect(const InEncodeVideoParam *inputParam);

    //CUDAインターフェースを初期化
    NVENCSTATUS InitCuda(int cudaSchedule);

    //inputParamからエンコーダに渡すパラメータを設定
    NVENCSTATUS SetInputParam(const InEncodeVideoParam *inputParam);

    //デコーダインスタンスを作成
    NVENCSTATUS InitDecoder(const InEncodeVideoParam *inputParam);

    //デコーダインスタンスを作成
    NVENCSTATUS InitFilters(const InEncodeVideoParam *inputParam);

    //エンコーダインスタンスを作成
    NVENCSTATUS CreateEncoder(const InEncodeVideoParam *inputParam);

    //入出力用バッファを確保
    NVENCSTATUS AllocateIOBuffers(uint32_t uInputWidth, uint32_t uInputHeight, NV_ENC_BUFFER_FORMAT inputFormat, const VideoInfo *pInputInfo);

    //フレームを1枚エンコーダに投入(非同期)
    //NVENCSTATUS EncodeFrame(uint64_t timestamp);

    //フレームを1枚エンコーダに投入(非同期、トランスコード中継用)
    NVENCSTATUS EncodeFrame(EncodeFrameConfig *pEncodeFrame, uint64_t timestamp, uint64_t duration);

    NVENCSTATUS NvEncEncodeFrame(EncodeBuffer *pEncodeBuffer, uint64_t timestamp, uint64_t duration);

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

    std::list<NVGPUInfo>         m_GPUList;               //GPUのリスト

    bool                        *m_pAbortByUser;          //ユーザーからの中断指令
    shared_ptr<RGYLog>           m_pNVLog;                //ログ出力管理

    CUctx_flags                  m_cudaSchedule;          //CUDAのスケジュール
    CUdevice                     m_device;                //CUDAデバイスインスタンス
    CUcontext                    m_cuContextCurr;         //CUDAコンテキスト
    CUvideoctxlock               m_ctxLock;               //CUDAロック
    int                          m_nDeviceId;             //DeviceId
    void                        *m_pDevice;               //デバイスインスタンス
    NV_ENCODE_API_FUNCTION_LIST *m_pEncodeAPI;            //NVEnc APIの関数リスト
    HINSTANCE                    m_hinstLib;              //nvEncodeAPI.dllのモジュールハンドル
    void                        *m_hEncoder;              //エンコーダのインスタンス
    NV_ENC_INITIALIZE_PARAMS     m_stCreateEncodeParams;  //エンコーダの初期化パラメータ

    vector<InputFrameBufInfo>    m_inputHostBuffer;

    const sTrimParam             *m_pTrimParam;
    shared_ptr<RGYInput>          m_pFileReader;           //動画読み込み
    vector<shared_ptr<RGYInput>>  m_AudioReaders;
    shared_ptr<RGYOutput>         m_pFileWriter;           //動画書き出し
    vector<shared_ptr<RGYOutput>> m_pFileWriterListAudio;
    shared_ptr<EncodeStatus>      m_pStatus;               //エンコードステータス管理
    shared_ptr<CPerfMonitor>      m_pPerfMonitor;
    NV_ENC_PIC_STRUCT             m_stPicStruct;           //エンコードフレーム情報(プログレッシブ/インタレ)
    NV_ENC_CONFIG                 m_stEncConfig;           //エンコード設定
#if ENABLE_AVSW_READER
    vector<unique_ptr<AVChapter>> m_AVChapterFromFile;   //ファイルから読み込んだチャプター
#endif //#if ENABLE_AVSW_READER

    vector<unique_ptr<NVEncFilter>> m_vpFilters;
    shared_ptr<NVEncFilterParam>    m_pLastFilterParam;

    GUID                         m_stCodecGUID;           //出力コーデック
    uint32_t                     m_uEncWidth;             //出力縦解像度
    uint32_t                     m_uEncHeight;            //出力横解像度
    vector<uint8_t>              m_HEVCHDRSeiMaxCll;
    vector<uint8_t>              m_HEVCHDRSeiMasterDisplay;
    vector<NV_ENC_SEI_PAYLOAD>   m_HEVCHDRSeiArray;       //HDR情報
    bool                         m_HEVCHDRSeiAppended;    //HDR情報を付加した

    int                          m_nProcSpeedLimit;       //処理速度制限 (0で制限なし)
    RGYAVSync                    m_nAVSyncMode;           //映像音声同期設定
    rgy_rational<int>            m_inputFps;              //入力フレームレート
    rgy_rational<int>            m_outputTimebase;        //出力のtimebase
#if ENABLE_AVSW_READER
    unique_ptr<CuvidDecode>      m_cuvidDec;              //デコード
#endif //#if ENABLE_AVSW_READER
    //サブメソッド
    NVENCSTATUS NvEncOpenEncodeSessionEx(void *device, NV_ENC_DEVICE_TYPE deviceType);
    NVENCSTATUS NvEncCreateInputBuffer(uint32_t width, uint32_t height, void **inputBuffer, NV_ENC_BUFFER_FORMAT inputFormat);
    NVENCSTATUS NvEncDestroyInputBuffer(NV_ENC_INPUT_PTR inputBuffer);
    NVENCSTATUS NvEncCreateBitstreamBuffer(uint32_t size, void **bitstreamBuffer);
    NVENCSTATUS NvEncDestroyBitstreamBuffer(NV_ENC_OUTPUT_PTR bitstreamBuffer);
    NVENCSTATUS NvEncLockBitstream(NV_ENC_LOCK_BITSTREAM *lockBitstreamBufferParams);
    NVENCSTATUS NvEncUnlockBitstream(NV_ENC_OUTPUT_PTR bitstreamBuffer);
    NVENCSTATUS NvEncLockInputBuffer(void *inputBuffer, void **bufferDataPtr, uint32_t *pitch);
    NVENCSTATUS NvEncUnlockInputBuffer(NV_ENC_INPUT_PTR inputBuffer);
    NVENCSTATUS NvEncGetEncodeStats(NV_ENC_STAT *encodeStats);
    NVENCSTATUS NvEncGetSequenceParams(NV_ENC_SEQUENCE_PARAM_PAYLOAD *sequenceParamPayload);
    NVENCSTATUS NvEncRegisterAsyncEvent(void **completionEvent);
    NVENCSTATUS NvEncUnregisterAsyncEvent(void *completionEvent);
    NVENCSTATUS NvEncRegisterResource(NV_ENC_INPUT_RESOURCE_TYPE resourceType, void* resourceToRegister, uint32_t width, uint32_t height, uint32_t pitch, NV_ENC_BUFFER_FORMAT inputFormat, void** registeredResource);
    NVENCSTATUS NvEncUnregisterResource(NV_ENC_REGISTERED_PTR registeredRes);
    NVENCSTATUS NvEncMapInputResource(void *registeredResource, void **mappedResource);
    NVENCSTATUS NvEncUnmapInputResource(NV_ENC_INPUT_PTR mappedInputBuffer);
    NVENCSTATUS NvEncFlushEncoderQueue(void *hEOSEvent);
    NVENCSTATUS NvEncDestroyEncoder();

    uint32_t                     m_uEncodeBufferCount;                //入力バッファ数 (16以上、MAX_ENCODE_QUEUE以下)
    CNvQueue<EncodeBuffer>       m_EncodeBufferQueue;                 //エンコーダへのフレーム投入キュー
    EncodeOutputBuffer           m_stEOSOutputBfr;                    //エンコーダからの出力バッファ
    EncodeBuffer                 m_stEncodeBuffer[MAX_ENCODE_QUEUE];  //エンコーダへのフレームバッファ

    //feature情報用
public:
    //コーデックのリストをm_EncodeFeaturesに作成
    //(これが終了した時点では、Codec GUIDのみが存在する)
    virtual NVENCSTATUS createDeviceCodecList();

    //Profile, Preset, Featureなどの情報を作成し、m_EncodeFeaturesを完成させる
    virtual NVENCSTATUS createDeviceFeatureList(bool getPresetConfig = true);

    //コーデックのFeature情報のリストの作成・取得
    virtual const std::vector<NVEncCodecFeature>& GetNVEncCapability();
protected:
    //指定したcodecFeatureのプロファイルリストをcodecFeatureに作成
    NVENCSTATUS setCodecProfileList(void *m_hEncoder, NVEncCodecFeature& codecFeature);

    //指定したcodecFeatureのプリセットリストをcodecFeatureに作成
    NVENCSTATUS setCodecPresetList(void *m_hEncoder, NVEncCodecFeature& codecFeature, bool getPresetConfig = true);

    //指定したcodecFeatureの対応入力フォーマットリストをcodecFeatureに作成
    NVENCSTATUS setInputFormatList(void *m_hEncoder, NVEncCodecFeature& codecFeature);

    //指定したcodecFeatureのfeatureリストをcodecFeatureに作成
    NVENCSTATUS GetCurrentDeviceNVEncCapability(void *m_hEncoder, NVEncCodecFeature& codecFeature);

    //m_EncodeFeaturesから指定したコーデックのデータへのポインタを取得 (なければnullptr)
    const NVEncCodecFeature *getCodecFeature(const GUID& codec);

    //指定したcodecFeatureで、指定したプロファイルに対応しているか
    bool checkProfileSupported(GUID profile, const NVEncCodecFeature *codecFeature = nullptr);

    //指定したcodecFeatureで、指定したプリセットに対応しているか
    bool checkPresetSupported(GUID profile, const NVEncCodecFeature *codecFeature = nullptr);

    //指定したcodecFeatureで、指定した入力フォーマットに対応しているか
    bool checkSurfaceFmtSupported(NV_ENC_BUFFER_FORMAT surfaceFormat, const NVEncCodecFeature *codecFeature = nullptr);

    //指定したcodecFeatureで、指定したfeatureの値を取得
    int getCapLimit(NV_ENC_CAPS flag, const NVEncCodecFeature *codecFeature = nullptr); 

    //コーデックのFeature情報のリスト (コーデックごとのリスト)
    std::vector<NVEncCodecFeature> m_EncodeFeatures;
};
