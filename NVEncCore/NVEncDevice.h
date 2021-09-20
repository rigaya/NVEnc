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

#include "rgy_osdep.h"
#pragma warning(push)
#pragma warning(disable: 4819)
//ファイルは、現在のコード ページ (932) で表示できない文字を含んでいます。
//データの損失を防ぐために、ファイルを Unicode 形式で保存してください。
#include <cuda.h>
#include <cuda_runtime.h>
#include "nvEncodeAPI.h"
#include "CuvidDecode.h"
#pragma warning(pop)
#include "NVEncParam.h"
#include <vector>
#include <list>
#include <memory>

#include "rgy_version.h"
#include "rgy_log.h"

#if defined(_WIN32) || defined(_WIN64)
#ifdef _M_IX86
static const TCHAR *NVENCODE_API_DLL = _T("nvEncodeAPI.dll");
#else
static const TCHAR *NVENCODE_API_DLL = _T("nvEncodeAPI64.dll");
#endif
#else
static const TCHAR *NVENCODE_API_DLL = _T("libnvidia-encode.so");
#endif


#if defined(_WIN32) || defined(_WIN64)
#define ENABLE_ASYNC 1
#else
#define ENABLE_ASYNC 0
#endif

#define INIT_CONFIG(configStruct, type, apiver) { \
    memset(&(configStruct), 0, sizeof(configStruct)); \
    (configStruct).version = ((type##_VER & 0xf0ff0000) | apiver); \
}
#define SET_VER(configStruct, type, apiver) { \
    (configStruct).version = ((type##_VER & 0xf0ff0000) | apiver); \
}

#define MAX_ENCODE_QUEUE 64

typedef NVENCSTATUS(NVENCAPI *MYPROC)(NV_ENCODE_API_FUNCTION_LIST *);

bool check_if_nvcuda_dll_available();

class NVEncCodecFeature {
public:
    GUID codec;                                       //CodecのGUID
    std::vector<GUID> profiles;                       //ProfileのGUIDリスト
    std::vector<GUID> presets;                        //PresetのGUIDリスト
    std::vector<NV_ENC_PRESET_CONFIG> presetConfigs;  //Presetの設定リスト
    std::vector<NV_ENC_BUFFER_FORMAT> surfaceFmt;     //対応フォーマットのリスト
    std::vector<NVEncCap> caps;                       //対応Featureデータ

    NVEncCodecFeature(GUID codec = { 0 });

    int getCapLimit(NV_ENC_CAPS flag) const;

    //指定したプロファイルに対応しているか
    bool checkProfileSupported(GUID profile) const;

    //指定したプリセットに対応しているか
    bool checkPresetSupported(GUID preset) const;

    //指定した入力フォーマットに対応しているか
    bool checkSurfaceFmtSupported(NV_ENC_BUFFER_FORMAT surfaceFormat) const;
};

template<class T>
class CNvQueue {
    T** m_pBuffer;
    unsigned int m_uSize;
    unsigned int m_uPendingCount;
    unsigned int m_uAvailableIdx;
    unsigned int m_uPendingndex;
public:
    CNvQueue(): m_pBuffer(NULL), m_uSize(0), m_uPendingCount(0), m_uAvailableIdx(0),
                m_uPendingndex(0)
    {
    }

    ~CNvQueue()
    {
        delete[] m_pBuffer;
    }

    bool Initialize(T *pItems, unsigned int uSize)
    {
        m_uSize = uSize;
        m_uPendingCount = 0;
        m_uAvailableIdx = 0;
        m_uPendingndex = 0;
        m_pBuffer = new T *[m_uSize];
        for (unsigned int i = 0; i < m_uSize; i++)
        {
            m_pBuffer[i] = &pItems[i];
        }
        return true;
    }


    T * GetAvailable()
    {
        T *pItem = NULL;
        if (m_uPendingCount == m_uSize)
        {
            return NULL;
        }
        pItem = m_pBuffer[m_uAvailableIdx];
        m_uAvailableIdx = (m_uAvailableIdx+1)%m_uSize;
        m_uPendingCount += 1;
        return pItem;
    }

    T* GetPending()
    {
        if (m_uPendingCount == 0)
        {
            return NULL;
        }

        T *pItem = m_pBuffer[m_uPendingndex];
        m_uPendingndex = (m_uPendingndex+1)%m_uSize;
        m_uPendingCount -= 1;
        return pItem;
    }
};

class NVEncoder {
public:
    NVEncoder(void *device, shared_ptr<RGYLog> log);
    virtual ~NVEncoder();

    NVENCSTATUS DestroyEncoder();

    //
    NVENCSTATUS loadNVEncAPIDLL();
    NVENCSTATUS InitSession();

    //エンコーダインスタンスを作成
    NVENCSTATUS CreateEncoder(NV_ENC_INITIALIZE_PARAMS *initParams);

    //サブメソッド
    NVENCSTATUS NvEncOpenEncodeSessionEx(void *device, NV_ENC_DEVICE_TYPE deviceType, const int sessionRetry = 0);
    NVENCSTATUS NvEncCreateInputBuffer(uint32_t width, uint32_t height, void **inputBuffer, NV_ENC_BUFFER_FORMAT inputFormat);
    NVENCSTATUS NvEncDestroyInputBuffer(NV_ENC_INPUT_PTR inputBuffer);
    NVENCSTATUS NvEncCreateBitstreamBuffer(uint32_t size, void **bitstreamBuffer);
    NVENCSTATUS NvEncDestroyBitstreamBuffer(NV_ENC_OUTPUT_PTR bitstreamBuffer);
    NVENCSTATUS NvEncLockBitstream(NV_ENC_LOCK_BITSTREAM *lockBitstreamBufferParams);
    NVENCSTATUS NvEncUnlockBitstream(NV_ENC_OUTPUT_PTR bitstreamBuffer);
    NVENCSTATUS NvEncLockInputBuffer(void *inputBuffer, void **bufferDataPtr, uint32_t *pitch);
    NVENCSTATUS NvEncUnlockInputBuffer(NV_ENC_INPUT_PTR inputBuffer);
    NVENCSTATUS NvEncReconfigureEncoder(NV_ENC_RECONFIGURE_PARAMS *reconf_params);
    NVENCSTATUS NvEncEncodePicture(NV_ENC_PIC_PARAMS *picParams);
    NVENCSTATUS NvEncGetEncodeStats(NV_ENC_STAT *encodeStats);
    NVENCSTATUS NvEncGetSequenceParams(NV_ENC_SEQUENCE_PARAM_PAYLOAD *sequenceParamPayload);
    NVENCSTATUS NvEncRegisterAsyncEvent(void **completionEvent);
    NVENCSTATUS NvEncUnregisterAsyncEvent(void *completionEvent);
    NVENCSTATUS NvEncRegisterResource(NV_ENC_INPUT_RESOURCE_TYPE resourceType, void *resourceToRegister, uint32_t width, uint32_t height, uint32_t pitch, NV_ENC_BUFFER_FORMAT inputFormat, void **registeredResource);
    NVENCSTATUS NvEncUnregisterResource(NV_ENC_REGISTERED_PTR registeredRes);
    NVENCSTATUS NvEncMapInputResource(void *registeredResource, void **mappedResource);
    NVENCSTATUS NvEncUnmapInputResource(NV_ENC_INPUT_PTR mappedInputBuffer);
    NVENCSTATUS NvEncFlushEncoderQueue(void *hEOSEvent);
    NVENCSTATUS NvEncDestroyEncoder();

    NVENCSTATUS SetEncodeCodecList();

    //指定したcodecFeatureのプロファイルリストをcodecFeatureに作成
    NVENCSTATUS setCodecProfileList(NVEncCodecFeature &codecFeature);

    //指定したcodecFeatureのプリセットリストをcodecFeatureに作成
    NVENCSTATUS setCodecPresetList(NVEncCodecFeature &codecFeature, bool getPresetConfig = true);

    //指定したcodecFeatureの対応入力フォーマットリストをcodecFeatureに作成
    NVENCSTATUS setInputFormatList(NVEncCodecFeature &codecFeature);

    //指定したcodecFeatureのfeatureリストをcodecFeatureに作成
    NVENCSTATUS GetCurrentDeviceNVEncCapability(NVEncCodecFeature &codecFeature);

    //m_EncodeFeaturesから指定したコーデックのデータへのポインタを取得 (なければnullptr)
    const NVEncCodecFeature *getCodecFeature(const GUID &codec);

    //feature情報用
    //コーデックのリストをm_EncodeFeaturesに作成
    //(これが終了した時点では、Codec GUIDのみが存在する)
    NVENCSTATUS createDeviceCodecList();

    //Profile, Preset, Featureなどの情報を作成し、m_EncodeFeaturesを完成させる
    NVENCSTATUS createDeviceFeatureList(bool getPresetConfig = false);

    //コーデックのFeature情報のリストの作成・取得
    const std::vector<NVEncCodecFeature> &GetNVEncCapability();

    bool checkAPIver(uint32_t major, uint8_t minor) const;
    uint32_t getAPIver() const { return m_apiVer; }
protected:
    //既定の出力先に情報をメッセージを出力
    void PrintMes(RGYLogLevel log_level, const tstring &str);
    void PrintMes(RGYLogLevel logLevel, const TCHAR *format, ...);

    //特定の関数でのエラーを表示
    void NVPrintFuncError(const TCHAR *funcName, NVENCSTATUS nvStatus);

    //特定の関数でのエラーを表示
    void NVPrintFuncError(const TCHAR *funcName, CUresult code);

    void *m_device;
    std::unique_ptr<NV_ENCODE_API_FUNCTION_LIST> m_pEncodeAPI;            //NVEnc APIの関数リスト
    HINSTANCE                    m_hinstLib;              //nvEncodeAPI.dllのモジュールハンドル
    void *m_hEncoder;              //エンコーダのインスタンス
    uint32_t m_apiVer;
    std::vector<NVEncCodecFeature> m_EncodeFeatures;
    std::shared_ptr<RGYLog>           m_log;
};

using unique_cuCtx = std::unique_ptr<std::remove_pointer<CUcontext>::type, decltype(&cuCtxDestroy)>;
using unique_vidCtxLock = std::unique_ptr<std::remove_pointer<CUvideoctxlock>::type, decltype(cuvidCtxLockDestroy)>;

class NVGPUInfo {
protected:
    int m_id;                 //CUDA device id
    std::string m_pciBusId;   //PCI Bus ID
    tstring m_name;           //GPU名
    std::pair<int, int> m_compute_capability;
    int m_nv_driver_version;   //1000倍
    int m_cuda_driver_version; //1000倍
    int m_cuda_cores;          //CUDAコア数
    int m_clock_rate;          //基本動作周波数(Hz)
    int m_pcie_gen, m_pcie_link; //PCIe接続情報
    CodecCsp m_cuvid_csp;      //デコード機能
    std::vector<NVEncCodecFeature> m_nvenc_codec_features; //エンコード機能
    CUdevice m_cudevice;
    unique_cuCtx m_cuCtx;
    unique_vidCtxLock m_vidCtxLock;
    std::unique_ptr<NVEncoder> m_encoder;
    std::shared_ptr<RGYLog> m_log;
public:
    NVGPUInfo(std::shared_ptr<RGYLog> log) :
        m_id(-1),
        m_pciBusId(),
        m_name(),
        m_compute_capability({ 0,0 }),
        m_nv_driver_version(0),
        m_cuda_driver_version(0),
        m_cuda_cores(0),
        m_clock_rate(0),
        m_pcie_gen(0),
        m_pcie_link(0),
        m_cuvid_csp(),
        m_nvenc_codec_features(),
        m_cudevice(),
        m_cuCtx(unique_cuCtx(nullptr, cuCtxDestroy)),
        m_vidCtxLock(unique_vidCtxLock(nullptr, cuvidCtxLockDestroy)),
        m_encoder(),
        m_log(log) {}

    ~NVGPUInfo() {
        close_device();
    }
    int id() const { return m_id; }
    const std::string& pciBusId() const { return m_pciBusId; }
    const tstring& name() const { return m_name; }
    const std::pair<int, int>& cc() const { return m_compute_capability; }
    int nv_driver_version() const { return m_nv_driver_version; }   //1000倍
    int cuda_driver_version() const { return m_cuda_driver_version; } //1000倍
    int cuda_cores() const { return m_cuda_cores; }         //CUDAコア数
    int clock_rate() const { return m_clock_rate; }          //基本動作周波数(Hz)
    int pcie_gen() const { return m_pcie_gen; }
    int pcie_link() const { return m_pcie_link; } //PCIe接続情報
    CodecCsp cuvid_csp() const { return m_cuvid_csp; }     //デコード機能
    const std::vector<NVEncCodecFeature> &nvenc_codec_features() const { return m_nvenc_codec_features; }//エンコード機能

    CUdevice cudevicetx() const { return m_cudevice; }
    CUcontext cuCtx() const { return m_cuCtx.get(); }
    CUvideoctxlock vidCtxLock() const { return m_vidCtxLock.get(); }
    NVEncoder *encoder() const { return m_encoder.get(); }

    void close_device();

    RGY_ERR initDevice(int deviceID, CUctx_flags ctxFlags, bool error_if_fail, bool skipHWDecodeCheck);
    RGY_ERR initEncoder();
    tstring infostr() const;
protected:
    void writeLog(RGYLogLevel log_level, const tstring &str) {
        if (!m_log || log_level < m_log->getLogLevel(RGY_LOGT_DEV)) {
            return;
        }
        auto lines = split(str, _T("\n"));
        for (const auto &line : lines) {
            if (line[0] != _T('\0')) {
                m_log->write(log_level, RGY_LOGT_DEV, (+_T("gpuinfo: ") + line + _T("\n")).c_str());
            }
        }
    }
    void writeLog(RGYLogLevel log_level, const TCHAR *format, ...) {
        if (!m_log || log_level < m_log->getLogLevel(RGY_LOGT_DEV)) {
            return;
        }

        va_list args;
        va_start(args, format);
        int len = _vsctprintf(format, args) + 1; // _vscprintf doesn't count terminating '\0'
        tstring buffer;
        buffer.resize(len, _T('\0'));
        _vstprintf_s(&buffer[0], len, format, args);
        va_end(args);
        writeLog(log_level, buffer);
    }
};

