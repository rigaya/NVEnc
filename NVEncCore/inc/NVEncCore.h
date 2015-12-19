//  -----------------------------------------------------------------------------------------
//    NVEnc by rigaya
//  -----------------------------------------------------------------------------------------
//   ソースコードについて
//   ・無保証です。
//   ・本ソースコードを使用したことによるいかなる損害・トラブルについてrigayaは責任を負いません。
//   以上に了解して頂ける場合、本ソースコードの使用、複製、改変、再頒布を行って頂いて構いません。
//  -----------------------------------------------------------------------------------------

#pragma once

#include <Windows.h>
#include <stdint.h>
#include <nvcuvid.h>
#include "nvEncodeAPI.h"
#include <tchar.h>
#include <vector>
#include <string>
#include "NVEncInput.h"
#include "NVEncUtil.h"
#include "NVEncStatus.h"
#include "NVEncLog.h"
#include "NVEncParam.h"
#include "FrameQueue.h"
#include "CuvidDecode.h"

static const int MAX_DECODE_FRAMES = 16;

static const int BITSTREAM_BUFFER_SIZE =  4 * 1024 * 1024;
static const int OUTPUT_BUF_SIZE       = 16 * 1024 * 1024;

static const int DEFAULT_GOP_LENGTH  = 0;
static const int DEFAULT_B_FRAMES    = 3;
static const int DEFAULT_REF_FRAMES  = 3;
static const int DEFAULT_NUM_SLICES  = 1;
static const int DEFAUTL_QP_I        = 20;
static const int DEFAULT_QP_P        = 23;
static const int DEFAULT_QP_B        = 25;
static const int DEFAULT_AVG_BITRATE = 7500000;
static const int DEFAULT_MAX_BITRATE = 17500000;

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

class NVEncoderGPUInfo
{
public:
    NVEncoderGPUInfo();
    ~NVEncoderGPUInfo();
    const std::vector<std::pair<uint32_t, tstring>> getGPUList() {
        return GPUList;
    }
private:
    std::vector<std::pair<uint32_t, tstring>> GPUList;
};

typedef struct VppParam {
    cudaVideoDeinterlaceMode deinterlace;
} VppParam;

typedef struct InEncodeVideoParam {
    InputVideoInfo input;         //入力する動画の情報
    tstring outputFilename;       //出力ファイル名
    int preset;                   //出力プリセット
    int deviceID;                 //使用するGPUのID
    int inputBuffer;              //使用されていません
    int par[2];                   //使用されていません
    NV_ENC_PIC_STRUCT picStruct;  //フレーム情報(プログレッシブ/インタレ)
    NV_ENC_CONFIG encConfig;      //エンコード設定
    int codec;                    //出力コーデック
    int bluray;                   //bluray出力
    int yuv444;                   //YUV444出力
    int lossless;                 //ロスレス出力
    tstring logfile;              //ログ出力先
    int loglevel;                 //ログ出力レベル
    VppParam vpp;                 //vpp
} InEncodeVideoParam;

class NVEncCore {
public:
    NVEncCore();
    ~NVEncCore();

    //デフォルトのエンコード設定を取得 (H.264用に設定済み)
    static NV_ENC_CONFIG DefaultParam();

    //デフォルトのH.264/AVC用の設定
    static NV_ENC_CODEC_CONFIG DefaultParamH264();

    //デフォルトのH.265/EVC用の設定
    static NV_ENC_CODEC_CONFIG DefaultParamHEVC();

    //CUDAインターフェース・デバイスの初期化
    virtual NVENCSTATUS Initialize(InEncodeVideoParam *inputParam);

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

protected:
    //メインメソッド
    NVENCSTATUS SetEncodeCodecList(void *encode);
    
    //既定の出力先に情報をメッセージを出力
    virtual void NVPrintf(FILE *fp, int logLevel, const TCHAR *format, ...);
    
    //特定の関数でのエラーを表示
    void NVPrintFuncError(const TCHAR *funcName, NVENCSTATUS nvStatus);

    //特定の関数でのエラーを表示
    void NVPrintFuncError(const TCHAR *funcName, CUresult code);

    //エンコーダへの入力を初期化
    virtual NVENCSTATUS InitInput(InEncodeVideoParam *inputParam);

    //ログを初期化
    virtual NVENCSTATUS InitLog(const InEncodeVideoParam *inputParam);

    //CUDAインターフェースを初期化
    NVENCSTATUS InitCuda(uint32_t deviceID);

    //エンコードデバイスを初期化
    NVENCSTATUS InitDevice(const InEncodeVideoParam *inputParam);

    //inputParamからエンコーダに渡すパラメータを設定
    NVENCSTATUS SetInputParam(const InEncodeVideoParam *inputParam);

    //デコーダインスタンスを作成
    NVENCSTATUS CreateDecoder(const InEncodeVideoParam *inputParam);

    //エンコーダインスタンスを作成
    NVENCSTATUS CreateEncoder(const InEncodeVideoParam *inputParam);

    //入出力用バッファを確保
    NVENCSTATUS AllocateIOBuffers(uint32_t uInputWidth, uint32_t uInputHeight, bool bYUV444);

    //フレームを1枚エンコーダに投入(非同期)
    NVENCSTATUS EncodeFrame(uint64_t timestamp);

    //フレームを1枚エンコーダに投入(非同期、トランスコード中継用)
    NVENCSTATUS EncodeFrame(EncodeFrameConfig *pEncodeFrame, uint64_t timestamp);

    NVENCSTATUS NvEncEncodeFrame(EncodeBuffer *pEncodeBuffer, uint64_t timestamp);

    //エンコーダをフラッシュしてストリームを最後まで取り出す
    NVENCSTATUS FlushEncoder();

    //入出力バッファを解放
    NVENCSTATUS ReleaseIOBuffers();

protected:
    //フレームの出力と集計
    NVENCSTATUS ProcessOutput(const EncodeBuffer *pEncodeBuffer);

    shared_ptr<CNVEncLog>        m_pNVLog;                //ログ出力管理

    CUdevice                     m_device;                //CUDAデバイスインスタンス
    CUcontext                    m_cuContextCurr;         //CUDAコンテキスト
    CUvideoctxlock               m_ctxLock;               //CUDAロック
    void                        *m_pDevice;               //デバイスインスタンス
    NV_ENCODE_API_FUNCTION_LIST *m_pEncodeAPI;            //NVEnc APIの関数リスト
    HINSTANCE                    m_hinstLib;              //nvEncodeAPI.dllのモジュールハンドル
    void                        *m_hEncoder;              //エンコーダのインスタンス
    NV_ENC_INITIALIZE_PARAMS     m_stCreateEncodeParams;  //エンコーダの初期化パラメータ
    FILE                        *m_fOutput;               //出力先ファイルポインタ
    char                        *m_pOutputBuf;            //出力ファイルバッファ

    NVEncBasicInput             *m_pInput;                //動画入力インスタンス
    shared_ptr<EncodeStatus>     m_pStatus;               //エンコードステータス管理
    NV_ENC_PIC_STRUCT            m_stPicStruct;           //エンコードフレーム情報(プログレッシブ/インタレ)
    NV_ENC_CONFIG                m_stEncConfig;           //エンコード設定

    GUID                         m_stCodecGUID;           //出力コーデック
    uint32_t                     m_uEncWidth;             //出力縦解像度
    uint32_t                     m_uEncHeight;            //出力横解像度
#if ENABLE_AVCUVID_READER
    unique_ptr<CuvidDecode>      m_cuvidDec;              //デコード
#endif //#if ENABLE_AVCUVID_READER
    //サブメソッド
    NVENCSTATUS NvEncOpenEncodeSessionEx(void *device, NV_ENC_DEVICE_TYPE deviceType);
    NVENCSTATUS NvEncCreateInputBuffer(uint32_t width, uint32_t height, void **inputBuffer, uint32_t isYuv444);
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
    NVENCSTATUS NvEncRegisterResource(NV_ENC_INPUT_RESOURCE_TYPE resourceType, void* resourceToRegister, uint32_t width, uint32_t height, uint32_t pitch, void** registeredResource);
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
