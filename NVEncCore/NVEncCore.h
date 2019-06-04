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
#pragma warning (push)
#pragma warning (disable: 4819)
#pragma warning (disable: 4201)
#include "nvEncodeAPI.h"
#pragma warning (pop)
#include <tchar.h>
#include <vector>
#include <list>
#include <string>
#include "rgy_input.h"
#include "rgy_output.h"
#include "rgy_status.h"
#include "rgy_log.h"
#include "rgy_bitstream.h"
#include "rgy_hdr10plus.h"
#include "NVEncUtil.h"
#include "NVEncParam.h"
#include "CuvidDecode.h"
#include "NVEncFilter.h"
#include "NVEncFrameInfo.h"

using std::vector;

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

class NVEncCodecFeature {
public:
    GUID codec;                                       //CodecのGUID
    std::vector<GUID> profiles;                       //ProfileのGUIDリスト
    std::vector<GUID> presets;                        //PresetのGUIDリスト
    std::vector<NV_ENC_PRESET_CONFIG> presetConfigs;  //Presetの設定リスト
    std::vector<NV_ENC_BUFFER_FORMAT> surfaceFmt;     //対応フォーマットのリスト
    std::vector<NVEncCap> caps;                       //対応Featureデータ

    NVEncCodecFeature(GUID codec ={ 0 }) {
        this->codec = codec;
    }
};

struct NVGPUInfo {
    int id;                 //CUDA device id
    std::string pciBusId;   //PCI Bus ID
    tstring name;           //GPU名
    std::pair<int, int> compute_capability;
    int nv_driver_version;   //1000倍
    int cuda_driver_version; //1000倍
    int cuda_cores;          //CUDAコア数
    int clock_rate;          //基本動作周波数(Hz)
    int pcie_gen, pcie_link; //PCIe接続情報
    CodecCsp cuvid_csp;      //デコード機能
    vector<NVEncCodecFeature> nvenc_codec_features; //エンコード機能
};

typedef void* nvfeature_t;
nvfeature_t nvfeature_create();
int nvfeature_createCacheAsync(nvfeature_t obj, int deviceID);
const std::vector<NVEncCodecFeature>& nvfeature_GetCachedNVEncCapability(nvfeature_t obj);

tstring get_codec_profile_name_from_guid(RGY_CODEC codec, const GUID& codecProfileGUID);
tstring get_codec_level_name(RGY_CODEC codec, int level);

//featureリストからHEVCのリストを取得 (HEVC非対応ならnullptr)
const NVEncCodecFeature *nvfeature_GetHEVCFeatures(const std::vector<NVEncCodecFeature>& codecFeatures);
//featureリストからHEVCのリストを取得 (H.264対応ならnullptr)
const NVEncCodecFeature *nvfeature_GetH264Features(const std::vector<NVEncCodecFeature>& codecFeatures);

//H.264が使用可能かどうかを取得 (取得できるまで待機)
bool nvfeature_H264Available(nvfeature_t obj);
//HEVCが使用可能かどうかを取得 (取得できるまで待機)
bool nvfeature_HEVCAvailable(nvfeature_t obj);

void nvfeature_close(nvfeature_t obj);


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

class NVEncCore {
public:
    NVEncCore();
    virtual ~NVEncCore();

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

    RGY_ERR CheckDynamicRCParams(std::vector<DynamicRCParam> &dynamicRC);

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
    RGY_ERR InitFilters(const InEncodeVideoParam *inputParam);

    //チャプター読み込み等
    NVENCSTATUS InitChapters(const InEncodeVideoParam *inputParam);

    //エンコーダインスタンスを作成
    NVENCSTATUS CreateEncoder(const InEncodeVideoParam *inputParam);

    //入出力用バッファを確保
    NVENCSTATUS AllocateIOBuffers(uint32_t uInputWidth, uint32_t uInputHeight, NV_ENC_BUFFER_FORMAT inputFormat, const VideoInfo *pInputInfo);

    NVENCSTATUS NvEncEncodeFrame(EncodeBuffer *pEncodeBuffer, int id, uint64_t timestamp, uint64_t duration, int inputFrameId);

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
    std::vector<DynamicRCParam>  m_dynamicRC;             //動的に変更するエンコーダのパラメータ
    int                          m_appliedDynamicRC;      //今適用されているパラメータ(未適用なら-1)

    vector<InputFrameBufInfo>    m_inputHostBuffer;

    sTrimParam                    m_trimParam;
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
#endif //#if ENABLE_AVSW_READER
    unique_ptr<RGYHDR10Plus>      m_hdr10plus;

    vector<unique_ptr<NVEncFilter>> m_vpFilters;
    shared_ptr<NVEncFilterParam>    m_pLastFilterParam;

    GUID                         m_stCodecGUID;           //出力コーデック
    uint32_t                     m_uEncWidth;             //出力縦解像度
    uint32_t                     m_uEncHeight;            //出力横解像度
    rgy_rational<int>            m_sar;                   //出力のsar比

    int                          m_nProcSpeedLimit;       //処理速度制限 (0で制限なし)
    RGYAVSync                    m_nAVSyncMode;           //映像音声同期設定
    rgy_rational<int>            m_inputFps;              //入力フレームレート
    rgy_rational<int>            m_outputTimebase;        //出力のtimebase
    rgy_rational<int>            m_encFps;                //エンコードのフレームレート
#if ENABLE_AVSW_READER
    unique_ptr<CuvidDecode>      m_cuvidDec;              //デコード
#endif //#if ENABLE_AVSW_READER
    //サブメソッド
    NVENCSTATUS NvEncOpenEncodeSessionEx(void *device, NV_ENC_DEVICE_TYPE deviceType, const int sessionRetry);
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

    int                          m_encodeBufferCount;                 //入力バッファ数 (16以上、MAX_ENCODE_QUEUE以下)
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
