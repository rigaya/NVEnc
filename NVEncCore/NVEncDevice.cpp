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

#include "rgy_osdep.h"

#include "cpu_info.h"
#include "gpu_info.h"
#include "rgy_version.h"
#include "rgy_log.h"
#include "rgy_util.h"
#include "NVEncDevice.h"
#pragma warning(push)
#pragma warning(disable: 4819)
#include "helper_cuda.h"
#include "helper_nvenc.h"
#pragma warning(pop)
#include "rgy_perf_monitor.h"

bool check_if_nvcuda_dll_available() {
    //check for nvcuda.dll
    HMODULE hModule = LoadLibrary(_T("nvcuda.dll"));
    if (hModule == NULL)
        return false;
    FreeLibrary(hModule);
    return true;
}

NVEncCodecFeature::NVEncCodecFeature(GUID codec_) :
    codec(codec_),
    profiles(),
    presets(),
    presetConfigs(),
    surfaceFmt(),
    caps() {
}

int NVEncCodecFeature::getCapLimit(NV_ENC_CAPS flag) const {
    return get_value(flag, caps);
}

//指定したプロファイルに対応しているか
bool NVEncCodecFeature::checkProfileSupported(GUID profile) const {
    for (auto &codecProf : profiles) {
        if (0 == memcmp(&profile, &codecProf, sizeof(codecProf))) {
            return true;
        }
    }
    return false;
}

//指定したプリセットに対応しているか
bool NVEncCodecFeature::checkPresetSupported(GUID preset) const {
    for (auto &codecPreset : presets) {
        if (0 == memcmp(&preset, &codecPreset, sizeof(codecPreset))) {
            return true;
        }
    }
    return false;
}

//指定した入力フォーマットに対応しているか
bool NVEncCodecFeature::checkSurfaceFmtSupported(NV_ENC_BUFFER_FORMAT surfaceFormat) const {
    for (auto codecFmt : surfaceFmt) {
        if (0 == memcmp(&surfaceFormat, &codecFmt, sizeof(surfaceFormat))) {
            return true;
        }
    }
    return false;
}

NVEncoder::NVEncoder(void *device, shared_ptr<RGYLog> log) :
    m_device(device),
    m_pEncodeAPI(nullptr),
    m_hinstLib(nullptr),
    m_hEncoder(nullptr),
    m_EncodeFeatures(),
    m_log(log) {

}

NVEncoder::~NVEncoder() {
    DestroyEncoder();
}

NVENCSTATUS NVEncoder::DestroyEncoder() {
    auto sts = NvEncDestroyEncoder();

    m_pEncodeAPI.reset();

    if (m_hinstLib) {
        FreeLibrary(m_hinstLib);
        m_hinstLib = nullptr;
    }

    return sts;
}

void NVEncoder::NVPrintFuncError(const TCHAR *funcName, NVENCSTATUS nvStatus) {
    PrintMes(RGY_LOG_ERROR, _T("Error on %s: %d (%s)\n"), funcName, nvStatus, char_to_tstring(_nvencGetErrorEnum(nvStatus)).c_str());
}

void NVEncoder::NVPrintFuncError(const TCHAR *funcName, CUresult code) {
    PrintMes(RGY_LOG_ERROR, _T("Error on %s: %d (%s)\n"), funcName, (int)code, char_to_tstring(_cudaGetErrorEnum(code)).c_str());
}

void NVEncoder::PrintMes(int log_level, const tstring &str) {
    if (m_log == nullptr || log_level < m_log->getLogLevel()) {
        return;
    }
    auto lines = split(str, _T("\n"));
    for (const auto &line : lines) {
        if (line[0] != _T('\0')) {
            m_log->write(log_level, (_T("nvenc : ") + line + _T("\n")).c_str());
        }
    }
}
void NVEncoder::PrintMes(int log_level, const TCHAR *format, ...) {
    if (m_log == nullptr || log_level < m_log->getLogLevel()) {
        return;
    }

    va_list args;
    va_start(args, format);
    int len = _vsctprintf(format, args) + 1; // _vscprintf doesn't count terminating '\0'
    tstring buffer;
    buffer.resize(len, _T('\0'));
    _vstprintf_s(&buffer[0], len, format, args);
    va_end(args);
    PrintMes(log_level, buffer);
}

NVENCSTATUS NVEncoder::NvEncCreateInputBuffer(uint32_t width, uint32_t height, void **inputBuffer, NV_ENC_BUFFER_FORMAT inputFormat) {
    NV_ENC_CREATE_INPUT_BUFFER createInputBufferParams;
    INIT_CONFIG(createInputBufferParams, NV_ENC_CREATE_INPUT_BUFFER);

    createInputBufferParams.width = width;
    createInputBufferParams.height = height;
    createInputBufferParams.memoryHeap = NV_ENC_MEMORY_HEAP_SYSMEM_CACHED;
    createInputBufferParams.bufferFmt = inputFormat;

    NVENCSTATUS nvStatus = m_pEncodeAPI->nvEncCreateInputBuffer(m_hEncoder, &createInputBufferParams);
    if (nvStatus != NV_ENC_SUCCESS) {
        NVPrintFuncError(_T("nvEncCreateInputBuffer"), nvStatus);
        return nvStatus;
    }

    *inputBuffer = createInputBufferParams.inputBuffer;

    return nvStatus;
}

NVENCSTATUS NVEncoder::NvEncDestroyInputBuffer(NV_ENC_INPUT_PTR inputBuffer) {
    NVENCSTATUS nvStatus = NV_ENC_SUCCESS;
    if (inputBuffer) {
        nvStatus = m_pEncodeAPI->nvEncDestroyInputBuffer(m_hEncoder, inputBuffer);
        if (nvStatus != NV_ENC_SUCCESS) {
            NVPrintFuncError(_T("nvEncDestroyInputBuffer"), nvStatus);
            return nvStatus;
        }
    }
    return nvStatus;
}

NVENCSTATUS NVEncoder::NvEncCreateBitstreamBuffer(uint32_t size, void **bitstreamBuffer) {
    UNREFERENCED_PARAMETER(size);
    NV_ENC_CREATE_BITSTREAM_BUFFER createBitstreamBufferParams;
    INIT_CONFIG(createBitstreamBufferParams, NV_ENC_CREATE_BITSTREAM_BUFFER);

    //ここでは特に指定せず、ドライバにバッファサイズを決めさせる
    //createBitstreamBufferParams.size = size;
    //createBitstreamBufferParams.memoryHeap = NV_ENC_MEMORY_HEAP_SYSMEM_CACHED;

    NVENCSTATUS nvStatus = m_pEncodeAPI->nvEncCreateBitstreamBuffer(m_hEncoder, &createBitstreamBufferParams);
    if (nvStatus != NV_ENC_SUCCESS) {
        NVPrintFuncError(_T("nvEncCreateBitstreamBuffer"), nvStatus);
        return nvStatus;
    }

    *bitstreamBuffer = createBitstreamBufferParams.bitstreamBuffer;

    return nvStatus;
}

NVENCSTATUS NVEncoder::NvEncDestroyBitstreamBuffer(NV_ENC_OUTPUT_PTR bitstreamBuffer) {
    NVENCSTATUS nvStatus = NV_ENC_SUCCESS;
    if (bitstreamBuffer) {
        nvStatus = m_pEncodeAPI->nvEncDestroyBitstreamBuffer(m_hEncoder, bitstreamBuffer);
        if (nvStatus != NV_ENC_SUCCESS) {
            NVPrintFuncError(_T("nvEncDestroyBitstreamBuffer"), nvStatus);
            return nvStatus;
        }
    }
    return nvStatus;
}

NVENCSTATUS NVEncoder::NvEncLockBitstream(NV_ENC_LOCK_BITSTREAM *lockBitstreamBufferParams) {
    NVENCSTATUS nvStatus = m_pEncodeAPI->nvEncLockBitstream(m_hEncoder, lockBitstreamBufferParams);
    if (nvStatus != NV_ENC_SUCCESS) {
        NVPrintFuncError(_T("nvEncLockBitstream"), nvStatus);
        return nvStatus;
    }
    return nvStatus;
}

NVENCSTATUS NVEncoder::NvEncUnlockBitstream(NV_ENC_OUTPUT_PTR bitstreamBuffer) {
    NVENCSTATUS nvStatus = m_pEncodeAPI->nvEncUnlockBitstream(m_hEncoder, bitstreamBuffer);
    if (nvStatus != NV_ENC_SUCCESS) {
        NVPrintFuncError(_T("nvEncUnlockBitstream"), nvStatus);
        return nvStatus;
    }
    return nvStatus;
}

NVENCSTATUS NVEncoder::NvEncLockInputBuffer(void *inputBuffer, void **bufferDataPtr, uint32_t *pitch) {
    NV_ENC_LOCK_INPUT_BUFFER lockInputBufferParams;
    INIT_CONFIG(lockInputBufferParams, NV_ENC_LOCK_INPUT_BUFFER);

    lockInputBufferParams.inputBuffer = inputBuffer;
    NVENCSTATUS nvStatus = m_pEncodeAPI->nvEncLockInputBuffer(m_hEncoder, &lockInputBufferParams);
    if (nvStatus != NV_ENC_SUCCESS) {
        NVPrintFuncError(_T("nvEncLockInputBuffer"), nvStatus);
        return nvStatus;
    }

    *bufferDataPtr = lockInputBufferParams.bufferDataPtr;
    *pitch = lockInputBufferParams.pitch;

    return nvStatus;
}

NVENCSTATUS NVEncoder::NvEncUnlockInputBuffer(NV_ENC_INPUT_PTR inputBuffer) {
    NVENCSTATUS nvStatus = m_pEncodeAPI->nvEncUnlockInputBuffer(m_hEncoder, inputBuffer);
    if (nvStatus != NV_ENC_SUCCESS) {
        NVPrintFuncError(_T("nvEncUnlockInputBuffer"), nvStatus);
        return nvStatus;
    }
    return nvStatus;
}

NVENCSTATUS NVEncoder::NvEncReconfigureEncoder(NV_ENC_RECONFIGURE_PARAMS *reconf_params) {
    NVENCSTATUS nvStatus = m_pEncodeAPI->nvEncReconfigureEncoder(m_hEncoder, reconf_params);
    if (nvStatus != NV_ENC_SUCCESS) {
        NVPrintFuncError(_T("nvEncReconfigureEncoder"), nvStatus);
        return nvStatus;
    }
    return nvStatus;
}

NVENCSTATUS NVEncoder::NvEncEncodePicture(NV_ENC_PIC_PARAMS *picParams) {
    NVENCSTATUS nvStatus = m_pEncodeAPI->nvEncEncodePicture(m_hEncoder, picParams);
    if (nvStatus != NV_ENC_SUCCESS && nvStatus != NV_ENC_ERR_NEED_MORE_INPUT) {
        NVPrintFuncError(_T("NvEncEncodePicture"), nvStatus);
        return nvStatus;
    }
    return nvStatus;
}

NVENCSTATUS NVEncoder::NvEncGetEncodeStats(NV_ENC_STAT *encodeStats) {
    NVENCSTATUS nvStatus = m_pEncodeAPI->nvEncGetEncodeStats(m_hEncoder, encodeStats);
    if (nvStatus != NV_ENC_SUCCESS) {
        NVPrintFuncError(_T("nvEncGetEncodeStats"), nvStatus);
        return nvStatus;
    }

    return nvStatus;
}

NVENCSTATUS NVEncoder::NvEncGetSequenceParams(NV_ENC_SEQUENCE_PARAM_PAYLOAD *sequenceParamPayload) {
    NVENCSTATUS nvStatus = m_pEncodeAPI->nvEncGetSequenceParams(m_hEncoder, sequenceParamPayload);
    if (nvStatus != NV_ENC_SUCCESS) {
        NVPrintFuncError(_T("nvEncGetSequenceParams"), nvStatus);
        return nvStatus;
    }

    return nvStatus;
}

NVENCSTATUS NVEncoder::NvEncRegisterAsyncEvent(void **completionEvent) {
    NV_ENC_EVENT_PARAMS eventParams;
    INIT_CONFIG(eventParams, NV_ENC_EVENT_PARAMS);

    eventParams.completionEvent = CreateEvent(NULL, FALSE, FALSE, NULL);
    NVENCSTATUS nvStatus = m_pEncodeAPI->nvEncRegisterAsyncEvent(m_hEncoder, &eventParams);
    if (nvStatus != NV_ENC_SUCCESS) {
        NVPrintFuncError(_T("nvEncRegisterAsyncEvent"), nvStatus);
        return nvStatus;
    }

    *completionEvent = eventParams.completionEvent;

    return nvStatus;
}

NVENCSTATUS NVEncoder::NvEncUnregisterAsyncEvent(void *completionEvent) {
    if (completionEvent) {
        NV_ENC_EVENT_PARAMS eventParams;
        INIT_CONFIG(eventParams, NV_ENC_EVENT_PARAMS);

        eventParams.completionEvent = completionEvent;

        NVENCSTATUS nvStatus = m_pEncodeAPI->nvEncUnregisterAsyncEvent(m_hEncoder, &eventParams);
        if (nvStatus != NV_ENC_SUCCESS) {
            NVPrintFuncError(_T("nvEncUnregisterAsyncEvent"), nvStatus);
            return nvStatus;
        }
    }

    return NV_ENC_SUCCESS;
}

NVENCSTATUS NVEncoder::NvEncRegisterResource(NV_ENC_INPUT_RESOURCE_TYPE resourceType, void* resourceToRegister, uint32_t width, uint32_t height, uint32_t pitch, NV_ENC_BUFFER_FORMAT inputFormat, void** registeredResource) {
    NVENCSTATUS nvStatus = NV_ENC_SUCCESS;
    NV_ENC_REGISTER_RESOURCE registerResParams;

    INIT_CONFIG(registerResParams, NV_ENC_REGISTER_RESOURCE);

    registerResParams.resourceType = resourceType;
    registerResParams.resourceToRegister = resourceToRegister;
    registerResParams.width = width;
    registerResParams.height = height;
    registerResParams.pitch = pitch;
    registerResParams.bufferFormat = inputFormat;
    registerResParams.bufferUsage = NV_ENC_INPUT_IMAGE;

    nvStatus = m_pEncodeAPI->nvEncRegisterResource(m_hEncoder, &registerResParams);
    if (nvStatus != NV_ENC_SUCCESS) {
        NVPrintFuncError(_T("nvEncRegisterResource"), nvStatus);
    }

    *registeredResource = registerResParams.registeredResource;

    return nvStatus;
}

NVENCSTATUS NVEncoder::NvEncUnregisterResource(NV_ENC_REGISTERED_PTR registeredRes) {
    NVENCSTATUS nvStatus = NV_ENC_SUCCESS;

    nvStatus = m_pEncodeAPI->nvEncUnregisterResource(m_hEncoder, registeredRes);
    if (nvStatus != NV_ENC_SUCCESS) {
        NVPrintFuncError(_T("nvEncUnregisterResource"), nvStatus);
    }
    return nvStatus;
}

NVENCSTATUS NVEncoder::NvEncMapInputResource(void *registeredResource, void **mappedResource) {
    NV_ENC_MAP_INPUT_RESOURCE mapInputResParams;
    INIT_CONFIG(mapInputResParams, NV_ENC_MAP_INPUT_RESOURCE);

    mapInputResParams.registeredResource = registeredResource;

    NVENCSTATUS nvStatus = m_pEncodeAPI->nvEncMapInputResource(m_hEncoder, &mapInputResParams);
    if (nvStatus != NV_ENC_SUCCESS) {
        NVPrintFuncError(_T("nvEncMapInputResource"), nvStatus);
        return nvStatus;
    }

    *mappedResource = mapInputResParams.mappedResource;

    return nvStatus;
}

NVENCSTATUS NVEncoder::NvEncUnmapInputResource(NV_ENC_INPUT_PTR mappedInputBuffer) {
    if (mappedInputBuffer) {
        NVENCSTATUS nvStatus = m_pEncodeAPI->nvEncUnmapInputResource(m_hEncoder, mappedInputBuffer);
        if (nvStatus != NV_ENC_SUCCESS) {
            NVPrintFuncError(_T("nvEncUnmapInputResource"), nvStatus);
            return nvStatus;
        }
    }

    return NV_ENC_SUCCESS;
}

NVENCSTATUS NVEncoder::NvEncDestroyEncoder() {
    NVENCSTATUS nvStatus = NV_ENC_SUCCESS;

    if (m_hEncoder && m_pEncodeAPI) {
        PrintMes(RGY_LOG_DEBUG, _T("nvEncDestroyEncoder...\n"));
        nvStatus = m_pEncodeAPI->nvEncDestroyEncoder(m_hEncoder);
        m_pEncodeAPI.reset();
        m_hEncoder = NULL;
        PrintMes(RGY_LOG_DEBUG, _T("nvEncDestroyEncoder: success.\n"));
    }

    return nvStatus;
}

NVENCSTATUS NVEncoder::NvEncFlushEncoderQueue(void *hEOSEvent) {
    NV_ENC_PIC_PARAMS encPicParams;
    INIT_CONFIG(encPicParams, NV_ENC_PIC_PARAMS);
    encPicParams.encodePicFlags = NV_ENC_PIC_FLAG_EOS;
    encPicParams.completionEvent = hEOSEvent;

    NVENCSTATUS nvStatus = m_pEncodeAPI->nvEncEncodePicture(m_hEncoder, &encPicParams);
    if (nvStatus != NV_ENC_SUCCESS) {
        NVPrintFuncError(_T("nvEncEncodePicture"), nvStatus);
        return nvStatus;
    }
    return nvStatus;
}

NVENCSTATUS NVEncoder::NvEncOpenEncodeSessionEx(void *device, NV_ENC_DEVICE_TYPE deviceType, const int sessionRetry) {
    NV_ENC_OPEN_ENCODE_SESSION_EX_PARAMS openSessionExParams;
    INIT_CONFIG(openSessionExParams, NV_ENC_OPEN_ENCODE_SESSION_EX_PARAMS);

    openSessionExParams.device = device;
    openSessionExParams.deviceType = deviceType;
    openSessionExParams.reserved = NULL;
    openSessionExParams.apiVersion = NVENCAPI_VERSION;

    static const int retry_millisec = 500;
    static const int retry_max = sessionRetry * 1000 / retry_millisec;
    NVENCSTATUS nvStatus = NV_ENC_SUCCESS;
    for (int retry = 0; NV_ENC_SUCCESS != (nvStatus = m_pEncodeAPI->nvEncOpenEncodeSessionEx(&openSessionExParams, &m_hEncoder)); retry++) {
        if (nvStatus != NV_ENC_ERR_OUT_OF_MEMORY) {
            NVPrintFuncError(_T("nvEncOpenEncodeSessionEx"), nvStatus);
            break;
        }
        if (retry >= retry_max) {
            PrintMes(RGY_LOG_ERROR,
                FOR_AUO ? _T("このエラーはメモリが不足しているか、同時にNVEncで3ストリーム以上エンコードしようとすると発生することがあります。\n")
                          _T("Geforceでは、NVIDIAのドライバの制限により3ストリーム以上の同時エンコードが行えません。\n")
                        : _T("This error might occur when shortage of memory, or when trying to encode more than 2 streams by NVEnc.\n")
                          _T("In Geforce, simultaneous encoding is limited up to 2, due to the NVIDIA's driver limitation.\n"));
            break;
        }
        if ((retry % (10 * 1000 / retry_millisec)) == 0) {
            PrintMes(RGY_LOG_INFO, _T("Waiting for other encode to finish...\n"));
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(retry_millisec));
    }

    return nvStatus;
}

NVENCSTATUS NVEncoder::InitSession() {
    if (m_hinstLib) {
        return NV_ENC_SUCCESS;
    }
    if (NULL == m_hinstLib) {
        if (NULL == (m_hinstLib = LoadLibrary(NVENCODE_API_DLL))) {
            PrintMes(RGY_LOG_ERROR, _T("%s does not exists in your system.\n"), NVENCODE_API_DLL);
            PrintMes(RGY_LOG_ERROR, _T("Please check if the GPU driver is propery installed."));
            return NV_ENC_ERR_OUT_OF_MEMORY;
        }
    }
    PrintMes(RGY_LOG_DEBUG, _T("Loaded %s.\n"), NVENCODE_API_DLL);

    MYPROC nvEncodeAPICreateInstance; // function pointer to create instance in nvEncodeAPI
    if (NULL == (nvEncodeAPICreateInstance = (MYPROC)GetProcAddress(m_hinstLib, "NvEncodeAPICreateInstance"))) {
        PrintMes(RGY_LOG_ERROR, _T("Failed to load address of NvEncodeAPICreateInstance from %s.\n"), NVENCODE_API_DLL);
        return NV_ENC_ERR_OUT_OF_MEMORY;
    }

    m_pEncodeAPI = std::make_unique<NV_ENCODE_API_FUNCTION_LIST>();
    if (!m_pEncodeAPI) {
        PrintMes(RGY_LOG_ERROR, FOR_AUO ? _T("NV_ENCODE_API_FUNCTION_LIST用のメモリ確保に失敗しました。\n") : _T("Failed to allocate memory for NV_ENCODE_API_FUNCTION_LIST.\n"));
        return NV_ENC_ERR_OUT_OF_MEMORY;
    }

    memset(m_pEncodeAPI.get(), 0, sizeof(NV_ENCODE_API_FUNCTION_LIST));
    m_pEncodeAPI->version = NV_ENCODE_API_FUNCTION_LIST_VER;

    auto nvStatus = nvEncodeAPICreateInstance(m_pEncodeAPI.get());
    if (nvStatus != NV_ENC_SUCCESS) {
        if (nvStatus == NV_ENC_ERR_INVALID_VERSION) {
            PrintMes(RGY_LOG_ERROR, _T("Failed to create instance of nvEncodeAPI(ver=0x%x), please consider updating your GPU driver.\n"), NV_ENCODE_API_FUNCTION_LIST_VER);
        } else {
            NVPrintFuncError(_T("nvEncodeAPICreateInstance"), nvStatus);
        }
        return nvStatus;
    }
    PrintMes(RGY_LOG_DEBUG, _T("nvEncodeAPICreateInstance(APIVer=0x%x): Success.\n"), NV_ENCODE_API_FUNCTION_LIST_VER);

    if (NV_ENC_SUCCESS != (nvStatus = NvEncOpenEncodeSessionEx(m_device, NV_ENC_DEVICE_TYPE_CUDA))) {
        if (nvStatus == NV_ENC_ERR_INVALID_VERSION) {
            PrintMes(RGY_LOG_ERROR, _T("Failed to create instance of NvEncOpenEncodeSessionEx(device_type=NV_ENC_DEVICE_TYPE_CUDA), please consider updating your GPU driver.\n"), NV_ENCODE_API_FUNCTION_LIST_VER);
        } else {
            NVPrintFuncError(_T("NvEncOpenEncodeSessionEx(device_type=NV_ENC_DEVICE_TYPE_CUDA)"), nvStatus);
        }
        return nvStatus;
    }
    PrintMes(RGY_LOG_DEBUG, _T("NvEncOpenEncodeSessionEx(device_type=NV_ENC_DEVICE_TYPE_CUDA): Success.\n"));
    return NV_ENC_SUCCESS;
}

NVENCSTATUS NVEncoder::CreateEncoder(NV_ENC_INITIALIZE_PARAMS *initParams) {
    NVENCSTATUS nvStatus;
    if (NV_ENC_SUCCESS != (nvStatus = m_pEncodeAPI->nvEncInitializeEncoder(m_hEncoder, initParams))) {
        PrintMes(RGY_LOG_ERROR,
            _T("%s: %d (%s)\n"), FOR_AUO ? _T("エンコーダの初期化に失敗しました。\n") : _T("Failed to Initialize the encoder\n."),
            nvStatus, char_to_tstring(_nvencGetErrorEnum(nvStatus)).c_str());
        return nvStatus;
    }
    PrintMes(RGY_LOG_DEBUG, _T("m_pEncodeAPI->nvEncInitializeEncoder: Success.\n"));

    return nvStatus;
}

NVENCSTATUS NVEncoder::SetEncodeCodecList() {
    NVENCSTATUS nvStatus = NV_ENC_SUCCESS;
    uint32_t dwEncodeGUIDCount = 0;
    if (NV_ENC_SUCCESS != (nvStatus = m_pEncodeAPI->nvEncGetEncodeGUIDCount(m_hEncoder, &dwEncodeGUIDCount))) {
        NVPrintFuncError(_T("nvEncGetEncodeGUIDCount"), nvStatus);
        return nvStatus;
    }
    uint32_t uArraysize = 0;
    GUID guid_init = { 0 };
    std::vector<GUID> list_codecs;
    list_codecs.resize(dwEncodeGUIDCount, guid_init);
    if (NV_ENC_SUCCESS != (nvStatus = m_pEncodeAPI->nvEncGetEncodeGUIDs(m_hEncoder, &list_codecs[0], dwEncodeGUIDCount, &uArraysize))) {
        NVPrintFuncError(_T("nvEncGetEncodeGUIDs"), nvStatus);
        return nvStatus;
    }
    for (auto codec : list_codecs) {
        m_EncodeFeatures.push_back(NVEncCodecFeature(codec));
    }
    return nvStatus;
}

NVENCSTATUS NVEncoder::setCodecProfileList(NVEncCodecFeature& codecFeature) {
    NVENCSTATUS nvStatus = NV_ENC_SUCCESS;
    uint32_t dwCodecProfileGUIDCount = 0;
    if (NV_ENC_SUCCESS != (nvStatus = m_pEncodeAPI->nvEncGetEncodeProfileGUIDCount(m_hEncoder, codecFeature.codec, &dwCodecProfileGUIDCount))) {
        NVPrintFuncError(_T("nvEncGetEncodeProfileGUIDCount"), nvStatus);
        return nvStatus;
    }
    uint32_t uArraysize = 0;
    GUID guid_init = { 0 };
    codecFeature.profiles.resize(dwCodecProfileGUIDCount, guid_init);
    if (NV_ENC_SUCCESS != (nvStatus = m_pEncodeAPI->nvEncGetEncodeProfileGUIDs(m_hEncoder, codecFeature.codec, &codecFeature.profiles[0], dwCodecProfileGUIDCount, &uArraysize))) {
        NVPrintFuncError(_T("nvEncGetEncodeProfileGUIDs"), nvStatus);
        return nvStatus;
    }
    return nvStatus;
}

NVENCSTATUS NVEncoder::setCodecPresetList(NVEncCodecFeature& codecFeature, bool getPresetConfig) {
    NVENCSTATUS nvStatus = NV_ENC_SUCCESS;
    uint32_t dwCodecProfileGUIDCount = 0;
    if (NV_ENC_SUCCESS != (nvStatus = m_pEncodeAPI->nvEncGetEncodePresetCount(m_hEncoder, codecFeature.codec, &dwCodecProfileGUIDCount))) {
        NVPrintFuncError(_T("nvEncGetEncodePresetCount"), nvStatus);
        return nvStatus;
    }
    uint32_t uArraysize = 0;
    GUID guid_init = { 0 };
    NV_ENC_PRESET_CONFIG config_init = { 0 };
    codecFeature.presets.resize(dwCodecProfileGUIDCount, guid_init);
    codecFeature.presetConfigs.resize(dwCodecProfileGUIDCount, config_init);
    if (NV_ENC_SUCCESS != (nvStatus = m_pEncodeAPI->nvEncGetEncodePresetGUIDs(m_hEncoder, codecFeature.codec, &codecFeature.presets[0], dwCodecProfileGUIDCount, &uArraysize))) {
        NVPrintFuncError(_T("nvEncGetEncodePresetGUIDs"), nvStatus);
        return nvStatus;
    }
    if (getPresetConfig) {
        for (uint32_t i = 0; i < codecFeature.presets.size(); i++) {
            INIT_CONFIG(codecFeature.presetConfigs[i], NV_ENC_PRESET_CONFIG);
            SET_VER(codecFeature.presetConfigs[i].presetCfg, NV_ENC_CONFIG);
            if (NV_ENC_SUCCESS != (nvStatus = m_pEncodeAPI->nvEncGetEncodePresetConfig(m_hEncoder, codecFeature.codec, codecFeature.presets[i], &codecFeature.presetConfigs[i]))) {
                NVPrintFuncError(_T("nvEncGetEncodePresetConfig"), nvStatus);
                return nvStatus;
            }
        }
    }
    return nvStatus;
}

NVENCSTATUS NVEncoder::setInputFormatList(NVEncCodecFeature& codecFeature) {
    NVENCSTATUS nvStatus = NV_ENC_SUCCESS;
    uint32_t dwInputFmtCount = 0;
    if (NV_ENC_SUCCESS != (nvStatus = m_pEncodeAPI->nvEncGetInputFormatCount(m_hEncoder, codecFeature.codec, &dwInputFmtCount))) {
        NVPrintFuncError(_T("nvEncGetInputFormatCount"), nvStatus);
        return nvStatus;
    }
    uint32_t uArraysize = 0;
    codecFeature.surfaceFmt.resize(dwInputFmtCount);
    if (NV_ENC_SUCCESS != (nvStatus = m_pEncodeAPI->nvEncGetInputFormats(m_hEncoder, codecFeature.codec, &codecFeature.surfaceFmt[0], dwInputFmtCount, &uArraysize))) {
        NVPrintFuncError(_T("nvEncGetInputFormats"), nvStatus);
        return nvStatus;
    }

    return nvStatus;
}

NVENCSTATUS NVEncoder::GetCurrentDeviceNVEncCapability(NVEncCodecFeature& codecFeature) {
    NVENCSTATUS nvStatus = NV_ENC_SUCCESS;
    bool check_h264 = get_value_from_guid(codecFeature.codec, list_nvenc_codecs) == NV_ENC_H264;
    auto add_cap_info = [&](NV_ENC_CAPS cap_id, bool for_h264_only, bool is_boolean, const TCHAR *cap_name) {
        if (!(!check_h264 && for_h264_only)) {
            NV_ENC_CAPS_PARAM param;
            INIT_CONFIG(param, NV_ENC_CAPS_PARAM);
            param.capsToQuery = cap_id;
            int value = 0;
            NVENCSTATUS result = m_pEncodeAPI->nvEncGetEncodeCaps(m_hEncoder, codecFeature.codec, &param, &value);
            if (NV_ENC_SUCCESS == result) {
                NVEncCap cap = { 0 };
                cap.id = cap_id;
                cap.isBool = is_boolean;
                cap.name = cap_name;
                cap.value = value;
                codecFeature.caps.push_back(cap);
            } else {
                nvStatus = result;
            }
        }
    };

    add_cap_info(NV_ENC_CAPS_NUM_MAX_BFRAMES,              false, false, _T("Max Bframes"));
    add_cap_info(NV_ENC_CAPS_SUPPORT_BFRAME_REF_MODE,      false, true,  _T("B Ref Mode"));
    add_cap_info(NV_ENC_CAPS_SUPPORTED_RATECONTROL_MODES,  false, false, _T("RC Modes"));
    add_cap_info(NV_ENC_CAPS_SUPPORT_FIELD_ENCODING,       false, true,  _T("Field Encoding"));
    add_cap_info(NV_ENC_CAPS_SUPPORT_MONOCHROME,           false, true,  _T("MonoChrome"));
    add_cap_info(NV_ENC_CAPS_SUPPORT_FMO,                  true,  true,  _T("FMO"));
    add_cap_info(NV_ENC_CAPS_SUPPORT_QPELMV,               false, true,  _T("Quater-Pel MV"));
    add_cap_info(NV_ENC_CAPS_SUPPORT_BDIRECT_MODE,         false, true,  _T("B Direct Mode"));
    add_cap_info(NV_ENC_CAPS_SUPPORT_CABAC,                true,  true,  _T("CABAC"));
    add_cap_info(NV_ENC_CAPS_SUPPORT_ADAPTIVE_TRANSFORM,   true,  true,  _T("Adaptive Transform"));
    add_cap_info(NV_ENC_CAPS_NUM_MAX_TEMPORAL_LAYERS,      false, false, _T("Max Temporal Layers"));
    add_cap_info(NV_ENC_CAPS_SUPPORT_HIERARCHICAL_PFRAMES, false, true,  _T("Hierarchial P Frames"));
    add_cap_info(NV_ENC_CAPS_SUPPORT_HIERARCHICAL_BFRAMES, false, true,  _T("Hierarchial B Frames"));
    add_cap_info(NV_ENC_CAPS_LEVEL_MAX,                    false, false, _T("Max Level"));
    add_cap_info(NV_ENC_CAPS_LEVEL_MIN,                    false, false, _T("Min Level"));
    add_cap_info(NV_ENC_CAPS_SUPPORT_YUV444_ENCODE,        false, true,  _T("4:4:4"));
    add_cap_info(NV_ENC_CAPS_WIDTH_MIN,                    false, false, _T("Min Width"));
    add_cap_info(NV_ENC_CAPS_WIDTH_MAX,                    false, false, _T("Max Width"));
    add_cap_info(NV_ENC_CAPS_HEIGHT_MIN,                   false, false, _T("Min Height"));
    add_cap_info(NV_ENC_CAPS_HEIGHT_MAX,                   false, false, _T("Max Height"));
    add_cap_info(NV_ENC_CAPS_SUPPORT_MULTIPLE_REF_FRAMES,  false, true,  _T("Multiple Refs"));
    add_cap_info(NV_ENC_CAPS_NUM_MAX_LTR_FRAMES,           false, false, _T("Max LTR Frames"));
    add_cap_info(NV_ENC_CAPS_SUPPORT_DYN_RES_CHANGE,       false, true,  _T("Dynamic Resolution Change"));
    add_cap_info(NV_ENC_CAPS_SUPPORT_DYN_BITRATE_CHANGE,   false, true,  _T("Dynamic Bitrate Change"));
    add_cap_info(NV_ENC_CAPS_SUPPORT_DYN_FORCE_CONSTQP,    false, true,  _T("Forced constant QP"));
    add_cap_info(NV_ENC_CAPS_SUPPORT_DYN_RCMODE_CHANGE,    false, true,  _T("Dynamic RC Mode Change"));
    add_cap_info(NV_ENC_CAPS_SUPPORT_SUBFRAME_READBACK,    false, true,  _T("Subframe Readback"));
    add_cap_info(NV_ENC_CAPS_SUPPORT_CONSTRAINED_ENCODING, false, true,  _T("Constrained Encoding"));
    add_cap_info(NV_ENC_CAPS_SUPPORT_INTRA_REFRESH,        false, true,  _T("Intra Refresh"));
    add_cap_info(NV_ENC_CAPS_SUPPORT_CUSTOM_VBV_BUF_SIZE,  false, true,  _T("Custom VBV Bufsize"));
    add_cap_info(NV_ENC_CAPS_SUPPORT_DYNAMIC_SLICE_MODE,   false, true,  _T("Dynamic Slice Mode"));
    add_cap_info(NV_ENC_CAPS_SUPPORT_REF_PIC_INVALIDATION, false, true,  _T("Ref Pic Invalidiation"));
    add_cap_info(NV_ENC_CAPS_PREPROC_SUPPORT,              false, true,  _T("PreProcess"));
    add_cap_info(NV_ENC_CAPS_ASYNC_ENCODE_SUPPORT,         false, true,  _T("Async Encoding"));
    add_cap_info(NV_ENC_CAPS_MB_NUM_MAX,                   false, false, _T("Max MBs"));
    //add_cap_info(NV_ENC_CAPS_MB_PER_SEC_MAX,               false, false, _T("MAX MB per sec"));
    add_cap_info(NV_ENC_CAPS_SUPPORT_LOSSLESS_ENCODE,      false, true,  _T("Lossless"));
    add_cap_info(NV_ENC_CAPS_SUPPORT_SAO,                  false, true,  _T("SAO"));
    add_cap_info(NV_ENC_CAPS_SUPPORT_MEONLY_MODE,          false, true,  _T("Me Only Mode"));
    add_cap_info(NV_ENC_CAPS_SUPPORT_LOOKAHEAD,            false, true,  _T("Lookahead"));
    add_cap_info(NV_ENC_CAPS_SUPPORT_TEMPORAL_AQ,          false, true,  _T("AQ (temporal)"));
    add_cap_info(NV_ENC_CAPS_SUPPORT_WEIGHTED_PREDICTION,  false, true,  _T("Weighted Prediction"));
    add_cap_info(NV_ENC_CAPS_SUPPORT_10BIT_ENCODE,         false, true,  _T("10bit depth"));
    return nvStatus;
}

NVENCSTATUS NVEncoder::createDeviceCodecList() {
    return SetEncodeCodecList();
}

NVENCSTATUS NVEncoder::createDeviceFeatureList(bool getPresetConfig) {
    NVENCSTATUS nvStatus = NV_ENC_SUCCESS;
    //m_EncodeFeaturesが作成されていなければ、自動的に作成
    if (m_EncodeFeatures.size() == 0) {
        SetEncodeCodecList();
    }

    if (NV_ENC_SUCCESS == nvStatus) {
        for (uint32_t i = 0; i < m_EncodeFeatures.size(); i++) {
            setCodecProfileList(m_EncodeFeatures[i]);
            setCodecPresetList(m_EncodeFeatures[i], getPresetConfig);
            setInputFormatList(m_EncodeFeatures[i]);
            GetCurrentDeviceNVEncCapability(m_EncodeFeatures[i]);
        }
    }
    return nvStatus;
}

const std::vector<NVEncCodecFeature> &NVEncoder::GetNVEncCapability() {
    if (m_EncodeFeatures.size() == 0) {
        createDeviceFeatureList();
    }
    return m_EncodeFeatures;
}

const NVEncCodecFeature *NVEncoder::getCodecFeature(const GUID &codec) {
    for (uint32_t i = 0; i < m_EncodeFeatures.size(); i++) {
        if (0 == memcmp(&m_EncodeFeatures[i].codec, &codec, sizeof(codec))) {
            return &m_EncodeFeatures[i];
        }
    }
    return nullptr;
}

RGY_ERR NVGPUInfo::initDevice(int deviceID, CUctx_flags ctxFlags, bool error_if_fail) {
#define GETATTRIB_CHECK(val, attrib, dev) { \
        cudaError_t cuErr = cudaDeviceGetAttribute(&(val), (attrib), (dev)); \
        if (cuErr == cudaErrorInvalidDevice || cuErr == cudaErrorInvalidValue) { \
            writeLog(error_level, _T("  Error: cudaDeviceGetAttribute(): %s\n"), char_to_tstring(cudaGetErrorString(cuErr)).c_str()); \
            return RGY_ERR_CUDA; \
        } \
        if (cuErr != cudaSuccess) { \
            writeLog(error_level, _T("  Warn: cudaDeviceGetAttribute(): %s\n"), char_to_tstring(cudaGetErrorString(cuErr)).c_str()); \
            val = 0; \
        } \
    }
    char pci_bus_name[64] = { 0 };
    char dev_name[256] = { 0 };
    CUdevice cuDevice = 0;
    const int error_level = (error_if_fail) ? RGY_LOG_ERROR : RGY_LOG_DEBUG;
    writeLog(RGY_LOG_DEBUG, _T("checking for device #%d.\n"), deviceID);
    auto cuResult = cuDeviceGet(&cuDevice, deviceID);
    if (cuResult != CUDA_SUCCESS) {
        writeLog(error_level, _T("  Error: cuDeviceGet(%d): %s\n"), deviceID, char_to_tstring(_cudaGetErrorEnum(cuResult)).c_str());
        return RGY_ERR_DEVICE_NOT_FOUND;
    }
    writeLog(RGY_LOG_DEBUG, _T("  cuDeviceGet(%d): success\n"), deviceID);

    if ((cuResult = cuDeviceGetName(dev_name, _countof(dev_name), cuDevice)) != CUDA_SUCCESS) {
        writeLog(error_level, _T("  Error: cuDeviceGetName(%d): %s\n"), deviceID, char_to_tstring(_cudaGetErrorEnum(cuResult)).c_str());
        return RGY_ERR_DEVICE_NOT_AVAILABLE;
    }
    writeLog(RGY_LOG_DEBUG, _T("  cuDeviceGetName(%d): %s\n"), deviceID, char_to_tstring(dev_name).c_str());
    int cudaDevMajor = 0, cudaDevMinor = 0;
    GETATTRIB_CHECK(cudaDevMajor, cudaDevAttrComputeCapabilityMajor, deviceID);
    GETATTRIB_CHECK(cudaDevMinor, cudaDevAttrComputeCapabilityMinor, deviceID);

    if (((cudaDevMajor << 4) + cudaDevMinor) < 0x30) {
        writeLog(error_level, _T("  Error: device does not satisfy required CUDA version (>=3.0): %d.%d\n"), cudaDevMajor, cudaDevMinor);
        return RGY_ERR_UNSUPPORTED;
    }
    writeLog(RGY_LOG_DEBUG, _T("  cudaDeviceGetAttribute: CUDA %d.%d\n"), cudaDevMajor, cudaDevMinor);

    {
        auto cuErr = cudaDeviceGetPCIBusId(pci_bus_name, sizeof(pci_bus_name), deviceID);
        if (cuErr == cudaErrorInvalidDevice || cuErr == cudaErrorInvalidValue) {
            writeLog((error_if_fail) ? RGY_LOG_WARN : RGY_LOG_DEBUG, _T("  Warn: cudaDeviceGetPCIBusId(): %s\n"), char_to_tstring(cudaGetErrorString(cuErr)).c_str());
        } else {
            writeLog(RGY_LOG_DEBUG, _T("  PCIBusId: %s\n"), char_to_tstring(pci_bus_name).c_str());
        }
    }

    int clockRate = 0, multiProcessorCount = 0;
    GETATTRIB_CHECK(clockRate, cudaDevAttrClockRate, deviceID);
    GETATTRIB_CHECK(multiProcessorCount, cudaDevAttrMultiProcessorCount, deviceID);

    m_id = deviceID;
    m_cudevice = cuDevice;
    m_pciBusId = pci_bus_name;
    m_name = char_to_tstring(dev_name);
    m_compute_capability.first = cudaDevMajor;
    m_compute_capability.second = cudaDevMinor;
    m_clock_rate = clockRate;
    m_cuda_cores = _ConvertSMVer2Cores(cudaDevMajor, cudaDevMinor) * multiProcessorCount;
    m_nv_driver_version = std::numeric_limits<int>::max();
    m_pcie_gen = 0;
    m_pcie_link = 0;
#if ENABLE_NVML
    if (m_pciBusId.length() > 0) {
        int version = 0, pcie_gen = 0, pcie_link = 0;
        NVMLMonitor nvml_monitor;
        if (NVML_SUCCESS == nvml_monitor.Init(m_pciBusId)
            && NVML_SUCCESS == nvml_monitor.getDriverVersionx1000(version)
            && NVML_SUCCESS == nvml_monitor.getMaxPCIeLink(pcie_gen, pcie_link)) {
            m_nv_driver_version = version;
            m_pcie_gen = pcie_gen;
            m_pcie_link = pcie_link;
            writeLog(RGY_LOG_DEBUG, _T("  Got GPU Info from NVML.\n"));
        }
    }
#endif //#if ENABLE_NVML
    if (m_nv_driver_version != std::numeric_limits<int>::max()) {
        if (0 < m_nv_driver_version && m_nv_driver_version < NV_DRIVER_VER_MIN) {
            writeLog(RGY_LOG_ERROR, _T("Insufficient NVIDIA driver version, Required %d.%d, Installed %d.%d\n"),
                NV_DRIVER_VER_MIN / 1000, (NV_DRIVER_VER_MIN % 1000) / 10,
                m_nv_driver_version / 1000, (m_nv_driver_version % 1000) / 10);
            return RGY_ERR_UNSUPPORTED;
        }
    }
    writeLog(RGY_LOG_DEBUG, _T("  NV Driver version: %d.\n"), m_nv_driver_version);

    m_cuda_driver_version = 0;
    if (CUDA_SUCCESS != (cuResult = cuDriverGetVersion(&m_cuda_driver_version))) {
        m_cuda_driver_version = -1;
    }
    writeLog(RGY_LOG_DEBUG, _T("  CUDA Driver version: %d.\n"), m_cuda_driver_version);

    writeLog(RGY_LOG_DEBUG, _T("using cuda schedule mode: %s.\n"), get_chr_from_value(list_cuda_schedule, ctxFlags));
    CUcontext cuCtxCreated;
    if (CUDA_SUCCESS != (cuResult = cuCtxCreate(&cuCtxCreated, ctxFlags, cuDevice))) {
        if (ctxFlags != 0) {
            writeLog(RGY_LOG_WARN, _T("cuCtxCreate error:0x%x (%s)\n"), cuResult, char_to_tstring(_cudaGetErrorEnum(cuResult)).c_str());
            writeLog(RGY_LOG_WARN, _T("retry cuCtxCreate with auto scheduling mode.\n"));
            if (CUDA_SUCCESS != (cuResult = cuCtxCreate(&cuCtxCreated, 0, cuDevice))) {
                writeLog(RGY_LOG_ERROR, _T("cuCtxCreate error:0x%x (%s)\n"), cuResult, char_to_tstring(_cudaGetErrorEnum(cuResult)).c_str());
                return RGY_ERR_DEVICE_NOT_FOUND;
            }
        } else {
            writeLog(RGY_LOG_ERROR, _T("cuCtxCreate error:0x%x (%s)\n"), cuResult, char_to_tstring(_cudaGetErrorEnum(cuResult)).c_str());
            return RGY_ERR_CUDA;
        }
    }
    writeLog(RGY_LOG_DEBUG, _T("cuCtxCreate: Success.\n"));

    CUcontext cuCtxTemp;
    if (CUDA_SUCCESS != (cuResult = cuCtxPopCurrent(&cuCtxTemp))) {
        writeLog(RGY_LOG_ERROR, _T("cuCtxPopCurrent error:0x%x (%s)\n"), cuResult, char_to_tstring(_cudaGetErrorEnum(cuResult)).c_str());
        return RGY_ERR_CUDA;
    }
    writeLog(RGY_LOG_DEBUG, _T("cuCtxPopCurrent: Success.\n"));
    m_cuCtx = std::unique_ptr<std::remove_pointer<CUcontext>::type, decltype(&cuCtxDestroy)>(cuCtxTemp, cuCtxDestroy);

    CUvideoctxlock vidCtxLockTmp;
    if (CUDA_SUCCESS != (cuResult = cuvidCtxLockCreate(&vidCtxLockTmp, m_cuCtx.get()))) {
        writeLog(RGY_LOG_ERROR, _T("Failed cuvidCtxLockCreate: 0x%x (%s)\n"), cuResult, char_to_tstring(_cudaGetErrorEnum(cuResult)).c_str());
        return RGY_ERR_CUDA;
    }
    writeLog(RGY_LOG_DEBUG, _T("cuvidCtxLockCreate: Success.\n"));
    m_vidCtxLock = std::unique_ptr<std::remove_pointer<CUvideoctxlock>::type, decltype(cuvidCtxLockDestroy)>(vidCtxLockTmp, cuvidCtxLockDestroy);
    {
        NVEncCtxAutoLock(ctxlock(m_vidCtxLock.get()));
        m_cuvid_csp = getHWDecCodecCsp();
    }
    m_encoder = std::make_unique<NVEncoder>(cuCtxCreated, m_log);
    auto nvsts = m_encoder->InitSession();
    if (nvsts != NV_ENC_SUCCESS) {
        writeLog(RGY_LOG_ERROR, _T("Failed to init encoder session.\n"));
        return RGY_ERR_UNSUPPORTED;
    }
    nvsts = m_encoder->createDeviceFeatureList();
    if (nvsts != NV_ENC_SUCCESS) {
        writeLog(RGY_LOG_ERROR, _T("Failed to create device codec list.\n"));
        return RGY_ERR_UNSUPPORTED;
    }
    writeLog(RGY_LOG_DEBUG, _T("  createDeviceFeatureList\n"));
    m_nvenc_codec_features = m_encoder->GetNVEncCapability();
    return RGY_ERR_NONE;
}

void NVGPUInfo::close_device() {
    writeLog(RGY_LOG_DEBUG, _T("Closing device #%d: %s...\n"), m_id, m_name.c_str());
    if (m_encoder) {
        writeLog(RGY_LOG_DEBUG, _T("Closing Encoder...\n"));
        m_encoder.reset();
        writeLog(RGY_LOG_DEBUG, _T("Closed Encoder.\n"));
    }
    if (m_vidCtxLock) {
        writeLog(RGY_LOG_DEBUG, _T("Closed cuvid Ctx Lock...\n"));
        m_vidCtxLock.reset();
        writeLog(RGY_LOG_DEBUG, _T("Closed cuvid Ctx Lock.\n"));
    }
    if (m_cuCtx) {
        writeLog(RGY_LOG_DEBUG, _T("Closed CUDA Context...\n"));
        m_cuCtx.reset();
        writeLog(RGY_LOG_DEBUG, _T("Closed CUDA Context.\n"));
    }
    writeLog(RGY_LOG_DEBUG, _T("Closed device #%d: %s.\n"), m_id, m_name.c_str());
    m_log.reset();
    m_pciBusId.clear();
    m_name.clear();
    m_nvenc_codec_features.clear();
    m_id = -1;
}

tstring NVGPUInfo::infostr() const {
    auto gpu_info = strsprintf(_T("#%d: %s"), m_id, m_name.c_str());
    if (m_cuda_cores > 0) {
        gpu_info += strsprintf(_T(" (%d cores"), m_cuda_cores);
        if (m_clock_rate > 0) {
            gpu_info += strsprintf(_T(", %d MHz"), m_clock_rate / 1000);
        }
        gpu_info += strsprintf(_T(")"));
    }
    if (m_pcie_gen > 0 && m_pcie_link > 0) {
        gpu_info += strsprintf(_T("[PCIe%dx%d]"), m_pcie_gen, m_pcie_link);
    }
    if (m_nv_driver_version) {
        gpu_info += strsprintf(_T("[%d.%02d]"), m_nv_driver_version / 1000, (m_nv_driver_version % 1000) / 10);
    }
    return gpu_info;
}
