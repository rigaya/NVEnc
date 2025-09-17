﻿// -----------------------------------------------------------------------------------------
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
#include "rgy_env.h"
#include "NVEncDevice.h"
#include "NVEncUtil.h"
#include "rgy_perf_monitor.h"

#define INIT_CONFIG_EX

bool check_if_nvcuda_dll_available() {
#if defined(_WIN32) || defined(_WIN64)
    //check for nvcuda.dll
    HMODULE hModule = LoadLibrary(_T("nvcuda.dll"));
    if (hModule == NULL)
        return false;
    FreeLibrary(hModule);
#endif //#if defined(_WIN32) || defined(_WIN64)
    return true;
}

// NVENC_STRUCT_VER1 と NVENC_STRUCT_VER2 は 1<<31 の有無の違い
// 構造体により異なる
#define NVENC_STRUCT_VER1(structver, apiver) ((uint32_t)(apiver) | (uint32_t)((structver) << 16) | (0x7 << 28) | (1<<31))
#define NVENC_STRUCT_VER2(structver, apiver) ((uint32_t)(apiver) | (uint32_t)((structver) << 16) | (0x7 << 28))

#define INIT_STRUCT(configStruct) { \
    memset(&(configStruct), 0, sizeof(configStruct)); \
    setStructVer(configStruct); \
}
#define INIT_STRUCT_BY_APIVER(configStruct, apiver) { \
    memset(&(configStruct), 0, sizeof(configStruct)); \
    setStructVer(configStruct, (apiver)); \
}

//前提とするAPIバージョンのチェック
static_assert(NVENCAPI_MAJOR_VERSION == 13);
static_assert(NVENCAPI_MINOR_VERSION == 0);
//対応するAPIバージョンの管理
static constexpr auto API_VER_LIST = make_array<uint32_t>(
    nvenc_api_ver(NVENCAPI_MAJOR_VERSION, NVENCAPI_MINOR_VERSION),
    nvenc_api_ver(12, 2),
    nvenc_api_ver(12, 1),
    nvenc_api_ver(12, 0),
    nvenc_api_ver(11, 1),
    nvenc_api_ver(11, 0),
    nvenc_api_ver(10, 0),
    nvenc_api_ver(9, 1),
    nvenc_api_ver(9, 0)
    );

//対応するAPIバージョンによる構造体のバージョン管理
//APIバージョンが異なると、構造体のバージョンも異なる場合があるので、これを一括で管理する
//API v9 - v11.1までは変更がなかった
//現在のAPIバージョンによって、必要に応じて異なるバージョンで初期化し、互換性を維持する
//static_assertでは現在のAPIバージョンが一致するかを確認し、将来のアップデートに備える
void NVEncoder::setStructVer(NV_ENC_OPEN_ENCODE_SESSION_EX_PARAMS& obj, const uint32_t apiver) const {
    if (nvenc_api_ver_check(apiver, nvenc_api_ver(12, 0))) {
        static const int latest_ver = 1;
        static_assert(NV_ENC_OPEN_ENCODE_SESSION_EX_PARAMS_VER == NVENC_STRUCT_VER2(latest_ver, NVENCAPI_VERSION));
        obj.version = NVENC_STRUCT_VER2(latest_ver, apiver);
    } else {
        obj.version = NVENC_STRUCT_VER2(1, apiver); // いまは同じ
    }
}
void NVEncoder::setStructVer(NV_ENCODE_API_FUNCTION_LIST& obj, const uint32_t apiver) const {
    if (nvenc_api_ver_check(apiver, nvenc_api_ver(12, 0))) {
        static const int latest_ver = 2;
        static_assert(NV_ENCODE_API_FUNCTION_LIST_VER == NVENC_STRUCT_VER2(latest_ver, NVENCAPI_VERSION));
        obj.version = NVENC_STRUCT_VER2(latest_ver, apiver);
    } else {
        obj.version = NVENC_STRUCT_VER2(2, apiver); // いまは同じ
    }
}
void NVEncoder::setStructVer(NV_ENC_INITIALIZE_PARAMS& obj) const {
    if (nvenc_api_ver_check(m_apiVer, nvenc_api_ver(12, 2))) {
        static const int latest_ver = 7;
        static_assert(NV_ENC_INITIALIZE_PARAMS_VER == NVENC_STRUCT_VER1(latest_ver, NVENCAPI_VERSION));
        obj.version = NVENC_STRUCT_VER1(latest_ver, m_apiVer);
    } else if (nvenc_api_ver_check(m_apiVer, nvenc_api_ver(12, 1))) {
        //API 12.1までは6
        obj.version = NVENC_STRUCT_VER1(6, m_apiVer);
    } else {
        //API 12.0までは7
        obj.version = NVENC_STRUCT_VER1(5, m_apiVer);
    }
}

void NVEncoder::setStructVer(NV_ENC_CONFIG& obj) const {
    if (nvenc_api_ver_check(m_apiVer, nvenc_api_ver(12, 2))) {
        static const int latest_ver = 9;
        static_assert(NV_ENC_CONFIG_VER == NVENC_STRUCT_VER1(latest_ver, NVENCAPI_VERSION));
        obj.version = NVENC_STRUCT_VER1(latest_ver, m_apiVer);
    } else if (nvenc_api_ver_check(m_apiVer, nvenc_api_ver(12, 0))) {
        //API 12.1までは8
        obj.version = NVENC_STRUCT_VER1(8, m_apiVer);
    } else {
        //API 11.1までは7
        obj.version = NVENC_STRUCT_VER1(7, m_apiVer);
    }
}

void NVEncoder::setStructVer(NV_ENC_RC_PARAMS& obj) const {
    static const int latest_ver = 1;
    static_assert(NV_ENC_RC_PARAMS_VER == NVENC_STRUCT_VER2(latest_ver, NVENCAPI_VERSION));
    obj.version = NVENC_STRUCT_VER2(latest_ver, m_apiVer);
}

void NVEncoder::setStructVer(NV_ENC_PIC_PARAMS& obj) const {
    if (nvenc_api_ver_check(m_apiVer, nvenc_api_ver(12, 2))) {
        static const int latest_ver = 7;
        static_assert(NV_ENC_PIC_PARAMS_VER == NVENC_STRUCT_VER1(latest_ver, NVENCAPI_VERSION));
        obj.version = NVENC_STRUCT_VER1(latest_ver, m_apiVer);
    } else if (nvenc_api_ver_check(m_apiVer, nvenc_api_ver(12, 0))) {
        //API 12.1までは6
        obj.version = NVENC_STRUCT_VER1(6, m_apiVer);
    } else {
        //API 11.1までは7
        obj.version = NVENC_STRUCT_VER1(4, m_apiVer);
    }
}

void NVEncoder::setStructVer(NV_ENC_LOCK_BITSTREAM& obj) const {
    if (nvenc_api_ver_check(m_apiVer, nvenc_api_ver(12, 2))) {
        static const int latest_ver = 2;
        static_assert(NV_ENC_LOCK_BITSTREAM_VER == NVENC_STRUCT_VER1(latest_ver, NVENCAPI_VERSION));
        obj.version = NVENC_STRUCT_VER2(latest_ver, m_apiVer);
    } else if (nvenc_api_ver_check(m_apiVer, nvenc_api_ver(12, 1))) {
        //API 12.1は1
        obj.version = NVENC_STRUCT_VER2(1, m_apiVer);
    } else if (nvenc_api_ver_check(m_apiVer, nvenc_api_ver(12, 0))) {
        //API 12.0は2
        obj.version = NVENC_STRUCT_VER2(2, m_apiVer);
    } else {
        //API 11.1までは1
        obj.version = NVENC_STRUCT_VER2(1, m_apiVer);
    }
}

void NVEncoder::setStructVer(NV_ENC_REGISTER_RESOURCE& obj) const {
    if (nvenc_api_ver_check(m_apiVer, nvenc_api_ver(12, 2))) {
        static const int latest_ver = 5;
        static_assert(NV_ENC_REGISTER_RESOURCE_VER == NVENC_STRUCT_VER2(latest_ver, NVENCAPI_VERSION));
        obj.version = NVENC_STRUCT_VER2(latest_ver, m_apiVer);
    } else if (nvenc_api_ver_check(m_apiVer, nvenc_api_ver(12, 0))) {
        //API 12.1までは4
        obj.version = NVENC_STRUCT_VER2(4, m_apiVer);
    } else {
        //API 11.1までは3
        obj.version = NVENC_STRUCT_VER2(3, m_apiVer);
    }
}

void NVEncoder::setStructVer(NV_ENC_RECONFIGURE_PARAMS& obj) const {
    if (nvenc_api_ver_check(m_apiVer, nvenc_api_ver(12, 2))) {
        static const int latest_ver = 2;
        static_assert(NV_ENC_RECONFIGURE_PARAMS_VER == NVENC_STRUCT_VER1(latest_ver, NVENCAPI_VERSION));
        obj.version = NVENC_STRUCT_VER1(latest_ver, m_apiVer);
    } else {
        //API 12.1までは1
        obj.version = NVENC_STRUCT_VER1(1, m_apiVer);
    }
}
void NVEncoder::setStructVer(NV_ENC_CREATE_INPUT_BUFFER& obj) const {
    if (nvenc_api_ver_check(m_apiVer, nvenc_api_ver(12, 2))) {
        static const int latest_ver = 2;
        static_assert(NV_ENC_CREATE_INPUT_BUFFER_VER == NVENC_STRUCT_VER2(latest_ver, NVENCAPI_VERSION));
        obj.version = NVENC_STRUCT_VER2(latest_ver, m_apiVer);
    } else {
        //API 12.1までは1
        obj.version = NVENC_STRUCT_VER2(1, m_apiVer);
    }
}
void NVEncoder::setStructVer(NV_ENC_CREATE_BITSTREAM_BUFFER& obj) const {
    static const int latest_ver = 1;
    static_assert(NV_ENC_CREATE_BITSTREAM_BUFFER_VER == NVENC_STRUCT_VER2(latest_ver, NVENCAPI_VERSION));
    obj.version = NVENC_STRUCT_VER2(latest_ver, m_apiVer);
}
void NVEncoder::setStructVer(NV_ENC_LOCK_INPUT_BUFFER& obj) const {
    static const int latest_ver = 1;
    static_assert(NV_ENC_LOCK_INPUT_BUFFER_VER == NVENC_STRUCT_VER2(latest_ver, NVENCAPI_VERSION));
    obj.version = NVENC_STRUCT_VER2(latest_ver, m_apiVer);
}
void NVEncoder::setStructVer(NV_ENC_EVENT_PARAMS& obj) const {
    if (nvenc_api_ver_check(m_apiVer, nvenc_api_ver(12, 2))) {
        static const int latest_ver = 2;
        static_assert(NV_ENC_EVENT_PARAMS_VER == NVENC_STRUCT_VER2(latest_ver, NVENCAPI_VERSION));
        obj.version = NVENC_STRUCT_VER2(latest_ver, m_apiVer);
    } else {
        //API 12.1までは1
        obj.version = NVENC_STRUCT_VER2(1, m_apiVer);
    }
}
void NVEncoder::setStructVer(NV_ENC_MAP_INPUT_RESOURCE& obj) const {
    static const int latest_ver = 4;
    static_assert(NV_ENC_MAP_INPUT_RESOURCE_VER == NVENC_STRUCT_VER2(latest_ver, NVENCAPI_VERSION));
    obj.version = NVENC_STRUCT_VER2(latest_ver, m_apiVer);
}
void NVEncoder::setStructVer(NV_ENC_PRESET_CONFIG& obj) const {
    if (nvenc_api_ver_check(m_apiVer, nvenc_api_ver(12, 2))) {
        static const int latest_ver = 5;
        static_assert(NV_ENC_PRESET_CONFIG_VER == NVENC_STRUCT_VER1(latest_ver, NVENCAPI_VERSION));
        obj.version = NVENC_STRUCT_VER1(latest_ver, m_apiVer);
    } else {
        //API 12.1までは4
        obj.version = NVENC_STRUCT_VER1(4, m_apiVer);
    }
}
void NVEncoder::setStructVer(NV_ENC_CAPS_PARAM& obj) const {
    static const int latest_ver = 1;
    static_assert(NV_ENC_CAPS_PARAM_VER == NVENC_STRUCT_VER2(latest_ver, NVENCAPI_VERSION));
    obj.version = NVENC_STRUCT_VER2(latest_ver, m_apiVer);
}
#undef NVENC_STRUCT_VER1
#undef NVENC_STRUCT_VER2

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
    m_apiVer(0),
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
        RGY_FREE_LIBRARY(m_hinstLib);
        m_hinstLib = nullptr;
    }
    m_apiVer = 0;

    return sts;
}

void NVEncoder::NVPrintFuncError(const TCHAR *funcName, NVENCSTATUS nvStatus) {
    tstring err_mes;
    if (m_hEncoder && m_pEncodeAPI->nvEncGetLastErrorString) {
        auto ptr = m_pEncodeAPI->nvEncGetLastErrorString(m_hEncoder);
        if (ptr && strlen(ptr) > 0) {
            err_mes = char_to_tstring(ptr);
        }
    }
    if (err_mes.empty()) {
        err_mes = char_to_tstring(_nvencGetErrorEnum(nvStatus));
    }
    PrintMes(RGY_LOG_ERROR, _T("Error on %s: %d (%s)\n"), funcName, nvStatus, err_mes.c_str());
}

void NVEncoder::NVPrintFuncError(const TCHAR *funcName, CUresult code) {
    PrintMes(RGY_LOG_ERROR, _T("Error on %s: %d (%s)\n"), funcName, (int)code, char_to_tstring(_cudaGetErrorEnum(code)).c_str());
}

void NVEncoder::PrintMes(RGYLogLevel log_level, const tstring &str) {
    if (m_log == nullptr || log_level < m_log->getLogLevel(RGY_LOGT_DEV)) {
        return;
    }
    auto lines = split(str, _T("\n"));
    for (const auto &line : lines) {
        if (line[0] != _T('\0')) {
            m_log->write(log_level, RGY_LOGT_DEV, (_T("nvenc : ") + line + _T("\n")).c_str());
        }
    }
}
void NVEncoder::PrintMes(RGYLogLevel log_level, const TCHAR *format, ...) {
    if (m_log == nullptr || log_level < m_log->getLogLevel(RGY_LOGT_DEV)) {
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
    INIT_STRUCT(createInputBufferParams);

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
    INIT_STRUCT(createBitstreamBufferParams);

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
    INIT_STRUCT(lockInputBufferParams);

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
#if ENABLE_ASYNC
    NV_ENC_EVENT_PARAMS eventParams;
    INIT_STRUCT(eventParams);

    eventParams.completionEvent = CreateEvent(NULL, FALSE, FALSE, NULL);
    NVENCSTATUS nvStatus = m_pEncodeAPI->nvEncRegisterAsyncEvent(m_hEncoder, &eventParams);
    if (nvStatus != NV_ENC_SUCCESS) {
        NVPrintFuncError(_T("nvEncRegisterAsyncEvent"), nvStatus);
        return nvStatus;
    }

    *completionEvent = eventParams.completionEvent;

    return nvStatus;
#else //#if ENABLE_ASYNC
    *completionEvent = nullptr;
    return NV_ENC_SUCCESS;
#endif //#if ENABLE_ASYNC
}

NVENCSTATUS NVEncoder::NvEncUnregisterAsyncEvent(void *completionEvent) {
#if ENABLE_ASYNC
    if (completionEvent) {
        NV_ENC_EVENT_PARAMS eventParams;
        INIT_STRUCT(eventParams);

        eventParams.completionEvent = completionEvent;

        NVENCSTATUS nvStatus = m_pEncodeAPI->nvEncUnregisterAsyncEvent(m_hEncoder, &eventParams);
        if (nvStatus != NV_ENC_SUCCESS) {
            NVPrintFuncError(_T("nvEncUnregisterAsyncEvent"), nvStatus);
            return nvStatus;
        }
    }
#endif //#if ENABLE_ASYNC
    return NV_ENC_SUCCESS;
}

NVENCSTATUS NVEncoder::NvEncRegisterResource(NV_ENC_INPUT_RESOURCE_TYPE resourceType, void* resourceToRegister, uint32_t width, uint32_t height, uint32_t pitch, NV_ENC_BUFFER_FORMAT inputFormat, void** registeredResource) {
    NVENCSTATUS nvStatus = NV_ENC_SUCCESS;
    NV_ENC_REGISTER_RESOURCE registerResParams;
    INIT_STRUCT(registerResParams);

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
    INIT_STRUCT(mapInputResParams);

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

NVENCSTATUS NVEncoder::NvEncSetIOCudaStreams(cudaStream_t& streamIn, cudaStream_t& streamOut) {
    NVENCSTATUS nvStatus = m_pEncodeAPI->nvEncSetIOCudaStreams(m_hEncoder, &streamIn, &streamOut);
    if (nvStatus != NV_ENC_SUCCESS) {
        NVPrintFuncError(_T("nvEncSetIOCudaStreams"), nvStatus);
        return nvStatus;
    }
    return NV_ENC_SUCCESS;
}

NVENCSTATUS NVEncoder::NvEncFlushEncoderQueue(void *hEOSEvent) {
    NV_ENC_PIC_PARAMS encPicParams;
    INIT_STRUCT(encPicParams);
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

    MYPROC nvEncodeAPICreateInstance; // function pointer to create instance in nvEncodeAPI
    if (NULL == (nvEncodeAPICreateInstance = (MYPROC)RGY_GET_PROC_ADDRESS(m_hinstLib, "NvEncodeAPICreateInstance"))) {
        PrintMes(RGY_LOG_ERROR, _T("Failed to load address of NvEncodeAPICreateInstance from %s.\n"), NVENCODE_API_DLL);
        return NV_ENC_ERR_OUT_OF_MEMORY;
    }

    m_pEncodeAPI = std::make_unique<NV_ENCODE_API_FUNCTION_LIST>();
    if (!m_pEncodeAPI) {
        PrintMes(RGY_LOG_ERROR, _T("Failed to allocate memory for NV_ENCODE_API_FUNCTION_LIST.\n"));
        return NV_ENC_ERR_OUT_OF_MEMORY;
    }

    NV_ENC_OPEN_ENCODE_SESSION_EX_PARAMS openSessionExParams;
    INIT_STRUCT_BY_APIVER(openSessionExParams, API_VER_LIST.front());

    openSessionExParams.device = device;
    openSessionExParams.deviceType = deviceType;
    openSessionExParams.reserved = NULL;

    NVENCSTATUS nvStatus = NV_ENC_SUCCESS;
    for (const auto apiver : API_VER_LIST) {
        INIT_STRUCT_BY_APIVER((*m_pEncodeAPI), apiver);

        nvStatus = nvEncodeAPICreateInstance(m_pEncodeAPI.get());
        if (nvStatus != NV_ENC_SUCCESS) {
            if (nvStatus == NV_ENC_ERR_INVALID_VERSION) {
                PrintMes(RGY_LOG_ERROR, _T("Failed to create instance of nvEncodeAPI(ver=0x%x: %d.%d), please consider updating your GPU driver.\n"), apiver, nvenc_api_ver_major(apiver), nvenc_api_ver_minor(apiver));
            } else {
                NVPrintFuncError(_T("nvEncodeAPICreateInstance"), nvStatus);
            }
            return nvStatus;
        }
        PrintMes(RGY_LOG_DEBUG, _T("nvEncodeAPICreateInstance(APIVer=0x%x: %d.%d): Success.\n"), apiver, nvenc_api_ver_major(apiver), nvenc_api_ver_minor(apiver));

        openSessionExParams.apiVersion = apiver;
        setStructVer(openSessionExParams, apiver);
        nvStatus = m_pEncodeAPI->nvEncOpenEncodeSessionEx(&openSessionExParams, &m_hEncoder);
        if (nvStatus != NV_ENC_ERR_INVALID_VERSION) {
            break;
        }

        if (m_pEncodeAPI) {
            m_pEncodeAPI->nvEncDestroyEncoder(m_hEncoder);
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
        PrintMes(RGY_LOG_DEBUG, _T("Failed to open Encode Session as API ver %d.%d\n"), nvenc_api_ver_major(apiver), nvenc_api_ver_minor(apiver));
    }

    if (nvStatus == NV_ENC_ERR_OUT_OF_MEMORY || nvStatus == NV_ENC_ERR_INCOMPATIBLE_CLIENT_KEY) {
        static const int retry_millisec = 500;
        static const int retry_max = sessionRetry * 1000 / retry_millisec;
        std::this_thread::sleep_for(std::chrono::milliseconds(retry_millisec));
        nvStatus = m_pEncodeAPI->nvEncOpenEncodeSessionEx(&openSessionExParams, &m_hEncoder);
        for (int retry = 0; nvStatus == NV_ENC_ERR_OUT_OF_MEMORY || nvStatus == NV_ENC_ERR_INCOMPATIBLE_CLIENT_KEY; retry++) {
            if (retry >= retry_max) {
                NVPrintFuncError(_T("nvEncOpenEncodeSessionEx"), nvStatus);
                PrintMes(RGY_LOG_ERROR,
                    FOR_AUO ? _T("このエラーはメモリが不足しているか、同時にNVEncで9ストリーム以上エンコードしようとすると発生することがあります。\n")
                    _T("Geforceでは、NVIDIAのドライバの制限により8ストリーム以上の同時エンコードが行えません。\n")
                    : _T("This error might occur when shortage of memory, or when trying to encode more than 9 streams by NVEnc.\n")
                    _T("In Geforce, simultaneous encoding is limited up to 8, due to the NVIDIA's driver limitation.\n"));
                break;
            }
            if ((retry % (10 * 1000 / retry_millisec)) == 0) {
                PrintMes(RGY_LOG_INFO, _T("Waiting for other encode to finish...\n"));
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(retry_millisec));
            nvStatus = m_pEncodeAPI->nvEncOpenEncodeSessionEx(&openSessionExParams, &m_hEncoder);
        }
    }
    if (nvStatus == NV_ENC_SUCCESS) {
        m_apiVer = openSessionExParams.apiVersion;
        PrintMes(RGY_LOG_DEBUG, _T("Opened Encode Session (API ver %d.%d)\n"), nvenc_api_ver_major(m_apiVer), nvenc_api_ver_minor(m_apiVer));
    } else {
        NVPrintFuncError(_T("nvEncOpenEncodeSessionEx"), nvStatus);
    }
    return nvStatus;
}

NVENCSTATUS NVEncoder::loadNVEncAPIDLL() {
    if (m_hinstLib == nullptr) {
        m_hinstLib = RGY_LOAD_LIBRARY(NVENCODE_API_DLL);
        if (m_hinstLib == nullptr && NVENCODE_API_DLL2 != nullptr) {
            m_hinstLib = RGY_LOAD_LIBRARY(NVENCODE_API_DLL2);
        }
        if (m_hinstLib == nullptr) {
            PrintMes(RGY_LOG_ERROR, _T("%s does not exists in your system.\n"), NVENCODE_API_DLL);
            PrintMes(RGY_LOG_ERROR, _T("Please check if the GPU driver is propery installed."));
            return NV_ENC_ERR_OUT_OF_MEMORY;
        }
    }
    PrintMes(RGY_LOG_DEBUG, _T("Loaded %s.\n"), NVENCODE_API_DLL);

    return NV_ENC_SUCCESS;
}

NVENCSTATUS NVEncoder::InitSession() {
    if (m_hinstLib) {
        return NV_ENC_SUCCESS;
    }

    auto nvStatus = loadNVEncAPIDLL();
    if (nvStatus != NV_ENC_SUCCESS) {
        return nvStatus;
    }

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
        NVPrintFuncError(_T("nvEncInitializeEncoder"), nvStatus);
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
    PrintMes(RGY_LOG_DEBUG, _T("m_pEncodeAPI->nvEncGetEncodeGUIDCount: %d.\n"), dwEncodeGUIDCount);

    uint32_t uArraysize = 0;
    GUID guid_init = { 0 };
    std::vector<GUID> list_codecs;
    list_codecs.resize(dwEncodeGUIDCount, guid_init);
    if (NV_ENC_SUCCESS != (nvStatus = m_pEncodeAPI->nvEncGetEncodeGUIDs(m_hEncoder, &list_codecs[0], dwEncodeGUIDCount, &uArraysize))) {
        NVPrintFuncError(_T("nvEncGetEncodeGUIDs"), nvStatus);
        return nvStatus;
    }
    for (const auto& codec : list_codecs) {
        PrintMes(RGY_LOG_DEBUG, _T("Found codec %s.\n"), get_name_from_guid(codec, list_nvenc_codecs));
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
    PrintMes(RGY_LOG_DEBUG, _T("m_pEncodeAPI->nvEncGetEncodeProfileGUIDCount: %d.\n"), dwCodecProfileGUIDCount);

    uint32_t uArraysize = 0;
    GUID guid_init = { 0 };
    codecFeature.profiles.resize(dwCodecProfileGUIDCount, guid_init);
    if (NV_ENC_SUCCESS != (nvStatus = m_pEncodeAPI->nvEncGetEncodeProfileGUIDs(m_hEncoder, codecFeature.codec, &codecFeature.profiles[0], dwCodecProfileGUIDCount, &uArraysize))) {
        NVPrintFuncError(_T("nvEncGetEncodeProfileGUIDs"), nvStatus);
        return nvStatus;
    }

    const auto codec_profile = get_codec_profile_list(codec_guid_enc_to_rgy(codecFeature.codec));
    for (const auto& profile : codecFeature.profiles) {
        PrintMes(RGY_LOG_DEBUG, _T("Found %s %s profile.\n"),
            get_name_from_guid(codecFeature.codec, list_nvenc_codecs), get_name_from_guid(profile, codec_profile));
    }
    return nvStatus;
}

NVENCSTATUS NVEncoder::setCodecPresetList(NVEncCodecFeature& codecFeature, bool getPresetConfig) {
    NVENCSTATUS nvStatus = NV_ENC_SUCCESS;
    uint32_t dwCodecPresetGUIDCount = 0;
    if (NV_ENC_SUCCESS != (nvStatus = m_pEncodeAPI->nvEncGetEncodePresetCount(m_hEncoder, codecFeature.codec, &dwCodecPresetGUIDCount))) {
        NVPrintFuncError(_T("nvEncGetEncodePresetCount"), nvStatus);
        return nvStatus;
    }
    PrintMes(RGY_LOG_DEBUG, _T("m_pEncodeAPI->nvEncGetEncodePresetCount: %d.\n"), dwCodecPresetGUIDCount);

    uint32_t uArraysize = 0;
    GUID guid_init = { 0 };
    NV_ENC_PRESET_CONFIG config_init = { 0 };
    codecFeature.presets.resize(dwCodecPresetGUIDCount, guid_init);
    codecFeature.presetConfigs.resize(dwCodecPresetGUIDCount, config_init);
    if (NV_ENC_SUCCESS != (nvStatus = m_pEncodeAPI->nvEncGetEncodePresetGUIDs(m_hEncoder, codecFeature.codec, &codecFeature.presets[0], dwCodecPresetGUIDCount, &uArraysize))) {
        NVPrintFuncError(_T("nvEncGetEncodePresetGUIDs"), nvStatus);
        return nvStatus;
    }
    PrintMes(RGY_LOG_DEBUG, _T("m_pEncodeAPI->nvEncGetEncodePresetGUIDs: Success.\n"));

    if (getPresetConfig) {
        for (uint32_t i = 0; i < codecFeature.presets.size(); i++) {
            INIT_STRUCT(codecFeature.presetConfigs[i]);
            setStructVer(codecFeature.presetConfigs[i].presetCfg);
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
    PrintMes(RGY_LOG_DEBUG, _T("m_pEncodeAPI->nvEncGetInputFormatCount: %d.\n"), dwInputFmtCount);

    uint32_t uArraysize = 0;
    codecFeature.surfaceFmt.resize(dwInputFmtCount);
    if (NV_ENC_SUCCESS != (nvStatus = m_pEncodeAPI->nvEncGetInputFormats(m_hEncoder, codecFeature.codec, &codecFeature.surfaceFmt[0], dwInputFmtCount, &uArraysize))) {
        NVPrintFuncError(_T("nvEncGetInputFormats"), nvStatus);
        return nvStatus;
    }
    PrintMes(RGY_LOG_DEBUG, _T("Found input fmt: "));
    for (const auto& fmt : codecFeature.surfaceFmt) {
        PrintMes(RGY_LOG_DEBUG, _T("%s "), RGY_CSP_NAMES[csp_enc_to_rgy(fmt)]);
    }
    PrintMes(RGY_LOG_DEBUG, _T("\n"));
    return nvStatus;
}

NVENCSTATUS NVEncoder::GetCurrentDeviceNVEncCapability(NVEncCodecFeature& codecFeature) {
    NVENCSTATUS nvStatus = NV_ENC_SUCCESS;
    const auto codec = codec_guid_enc_to_rgy(codecFeature.codec);
    bool check_h264 = codec == RGY_CODEC_H264;
    auto add_cap_info = [&](NV_ENC_CAPS cap_id, bool for_h264_only, bool is_boolean, const TCHAR *cap_name, const CX_DESC *desc = nullptr, const CX_DESC* desc_bitflag = nullptr) {
        if (!(!check_h264 && for_h264_only)) {
            NV_ENC_CAPS_PARAM param;
            INIT_STRUCT(param);
            param.capsToQuery = cap_id;
            int value = 0;
            NVENCSTATUS result = m_pEncodeAPI->nvEncGetEncodeCaps(m_hEncoder, codecFeature.codec, &param, &value);
            if (NV_ENC_SUCCESS == result) {
                NVEncCap cap = { 0 };
                cap.id = cap_id;
                cap.isBool = is_boolean;
                cap.name = cap_name;
                cap.value = value;
                cap.desc = desc;
                cap.desc_bit_flag = desc_bitflag;
                codecFeature.caps.push_back(cap);
            } else {
                nvStatus = result;
            }
        }
    };

    if (nvenc_api_ver_check(m_apiVer, nvenc_api_ver(10,0))) {
        add_cap_info(NV_ENC_CAPS_NUM_ENCODER_ENGINES, false, false, _T("Encoder Engines"));
    }
    add_cap_info(NV_ENC_CAPS_NUM_MAX_BFRAMES,              false, false, _T("Max Bframes"));
    add_cap_info(NV_ENC_CAPS_SUPPORT_BFRAME_REF_MODE,      false, false, _T("B Ref Mode"), list_nvenc_caps_bref_mode);
    add_cap_info(NV_ENC_CAPS_SUPPORTED_RATECONTROL_MODES,  false, false, _T("RC Modes"), nullptr, list_nvenc_rc_method_en);
    add_cap_info(NV_ENC_CAPS_SUPPORT_FIELD_ENCODING,       false, false, _T("Field Encoding"), list_nvenc_caps_field_encoding);
    add_cap_info(NV_ENC_CAPS_SUPPORT_MONOCHROME,           false, true,  _T("MonoChrome"));
    add_cap_info(NV_ENC_CAPS_SUPPORT_ALPHA_LAYER_ENCODING, false, true,  _T("Alpha Channel"));
    add_cap_info(NV_ENC_CAPS_SUPPORT_FMO,                  true,  true,  _T("FMO"));
    add_cap_info(NV_ENC_CAPS_SUPPORT_QPELMV,               false, true,  _T("Quater-Pel MV"));
    add_cap_info(NV_ENC_CAPS_SUPPORT_BDIRECT_MODE,         false, true,  _T("B Direct Mode"));
    add_cap_info(NV_ENC_CAPS_SUPPORT_CABAC,                true,  true,  _T("CABAC"));
    add_cap_info(NV_ENC_CAPS_SUPPORT_ADAPTIVE_TRANSFORM,   true,  true,  _T("Adaptive Transform"));
    add_cap_info(NV_ENC_CAPS_NUM_MAX_TEMPORAL_LAYERS,      false, false, _T("Max Temporal Layers"));
    add_cap_info(NV_ENC_CAPS_SUPPORT_HIERARCHICAL_PFRAMES, false, true,  _T("Hierarchial P Frames"));
    add_cap_info(NV_ENC_CAPS_SUPPORT_HIERARCHICAL_BFRAMES, false, true,  _T("Hierarchial B Frames"));
    add_cap_info(NV_ENC_CAPS_LEVEL_MAX,                    false, false, _T("Max Level"), get_level_list(codec));
    add_cap_info(NV_ENC_CAPS_LEVEL_MIN,                    false, false, _T("Min Level"), get_level_list(codec));
    add_cap_info(NV_ENC_CAPS_SUPPORT_YUV444_ENCODE,        false, true,  _T("4:4:4"));
    if (nvenc_api_ver_check(m_apiVer, nvenc_api_ver(13, 0))) {
        add_cap_info(NV_ENC_CAPS_SUPPORT_YUV422_ENCODE,    false, true,  _T("4:2:2"));
    }
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
    add_cap_info(NV_ENC_CAPS_SUPPORT_MEONLY_MODE,          false, false, _T("Me Only Mode"), list_nvenc_caps_me_only);
    add_cap_info(NV_ENC_CAPS_SUPPORT_LOOKAHEAD,            false, true,  _T("Lookahead"));
    add_cap_info(NV_ENC_CAPS_SUPPORT_TEMPORAL_AQ,          false, true,  _T("AQ (temporal)"));
    add_cap_info(NV_ENC_CAPS_SUPPORT_WEIGHTED_PREDICTION,  false, true,  _T("Weighted Prediction"));
    if (nvenc_api_ver_check(m_apiVer, nvenc_api_ver(12, 2))) {
        add_cap_info(NV_ENC_CAPS_SUPPORT_TEMPORAL_FILTER,  false, true, _T("Temporal Filter"));
        add_cap_info(NV_ENC_CAPS_SUPPORT_LOOKAHEAD_LEVEL,  false, true, _T("Lookahead Level"));
        add_cap_info(NV_ENC_CAPS_SUPPORT_UNIDIRECTIONAL_B, false, true, _T("Undirectional B"));
    }
    if (nvenc_api_ver_check(m_apiVer, nvenc_api_ver(11, 0))) {
        add_cap_info(NV_ENC_CAPS_SUPPORT_TEMPORAL_SVC,   false, true, _T("Temporal SVC"));
    }
    if (nvenc_api_ver_check(m_apiVer, nvenc_api_ver(13, 0))) {
        add_cap_info(NV_ENC_CAPS_SUPPORT_MVHEVC_ENCODE,  false, true, _T("MV-HEVC"));
    }
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

bool NVEncoder::checkAPIver(uint32_t major, uint8_t minor) const {
    return nvenc_api_ver_check(m_apiVer, nvenc_api_ver(major, minor));
}

const NVEncCodecFeature *NVEncoder::getCodecFeature(const GUID &codec) {
    for (uint32_t i = 0; i < m_EncodeFeatures.size(); i++) {
        if (0 == memcmp(&m_EncodeFeatures[i].codec, &codec, sizeof(codec))) {
            return &m_EncodeFeatures[i];
        }
    }
    return nullptr;
}

RGY_ERR NVGPUInfo::initDevice(int deviceID, CUctx_flags ctxFlags, bool error_if_fail, [[maybe_unused]] bool initDX11, [[maybe_unused]] RGYParamInitVulkan initVulkan, bool skipHWDecodeCheck, bool disableNVML) {
#define GETATTRIB_CHECK(val, attrib, dev) { \
        auto cuGetAttResult = cuDeviceGetAttribute(&(val), (attrib), (dev)); \
        if (cuGetAttResult == CUDA_ERROR_INVALID_DEVICE || cuGetAttResult == CUDA_ERROR_INVALID_VALUE) { \
            writeLog(error_level, _T("  Error: cuDeviceGetAttribute(%s): %s\n"), char_to_tstring(#attrib).c_str(), char_to_tstring(_cudaGetErrorEnum(cuGetAttResult)).c_str()); \
            return err_to_rgy(cuGetAttResult); \
        } \
        if (cuGetAttResult != CUDA_SUCCESS) { \
            writeLog(error_level, _T("  Warn: cuDeviceGetAttribute(%s): %s\n"), char_to_tstring(#attrib).c_str(), char_to_tstring(_cudaGetErrorEnum(cuGetAttResult)).c_str()); \
            val = 0; \
        } \
    }
    char pci_bus_name[64] = { 0 };
    char dev_name[256] = { 0 };
    CUdevice cuDevice = 0;
    CUresult cuResult = CUDA_SUCCESS;
    const auto error_level = (error_if_fail) ? RGY_LOG_ERROR : RGY_LOG_DEBUG;
#if ENABLE_D3D11
    if (initDX11) {
        writeLog(RGY_LOG_DEBUG, _T("checking for DX11 device #%d.\n"), deviceID);
        m_dx11 = std::make_unique<DeviceDX11>();
        auto err = m_dx11->Init(deviceID, m_log);
        if (err != RGY_ERR_NONE) {
            writeLog(RGY_LOG_DEBUG, _T("Failed to init DX11 device #%d: %s\n"), deviceID, get_err_mes(err));
            return err;
        }
        writeLog(RGY_LOG_DEBUG, _T("Init DX11 device %d.\n"), wstring_to_tstring(m_dx11->GetDisplayDeviceName()).c_str());
        //DX11デバイスの初期化に成功したら、そのデバイスのCUDAをcudaD3D11GetDeviceを使って初期化
        cuResult = cuD3D11GetDevice(&cuDevice, m_dx11->GetAdaptor());
        if (cuResult != CUDA_SUCCESS) {
            writeLog(RGY_LOG_WARN, _T("Failed to init CUDA device #%d from DX11 device.\n"), deviceID);
            m_dx11.reset();
        } else {
            writeLog(RGY_LOG_DEBUG, _T("  cuDeviceGet:DX11(%d): success: %d\n"), deviceID, cuDevice);
        }
    }
    if (!m_dx11)
#endif // #if ENABLE_D3D11
#if ENABLE_VULKAN
    if (initVulkan != RGYParamInitVulkan::Disable) {
        m_vulkan = std::make_unique<DeviceVulkan>();
        std::vector<const char *> extInstance;
        extInstance.push_back(VK_KHR_EXTERNAL_MEMORY_CAPABILITIES_EXTENSION_NAME);
        extInstance.push_back(VK_KHR_EXTERNAL_SEMAPHORE_CAPABILITIES_EXTENSION_NAME);
        
        std::vector<const char *> extDevice;
        extDevice.push_back(VK_KHR_EXTERNAL_MEMORY_EXTENSION_NAME);
        extDevice.push_back(VK_KHR_EXTERNAL_SEMAPHORE_EXTENSION_NAME);
        extDevice.push_back(VK_KHR_TIMELINE_SEMAPHORE_EXTENSION_NAME);
#if defined(_WIN32) || defined(_WIN64)
        extDevice.push_back(VK_KHR_EXTERNAL_MEMORY_WIN32_EXTENSION_NAME);
        extDevice.push_back(VK_KHR_EXTERNAL_SEMAPHORE_WIN32_EXTENSION_NAME);
#else
        extDevice.push_back(VK_KHR_EXTERNAL_MEMORY_FD_EXTENSION_NAME);
        extDevice.push_back(VK_KHR_EXTERNAL_SEMAPHORE_FD_EXTENSION_NAME);
#endif //defined(_WIN32) || defined(_WIN64)
        auto hexString = [](const void *uuid) {
            tstring str;
            const uint8_t *buf = (const uint8_t *)uuid;
            for (size_t i = 0; i < VK_UUID_SIZE; ++i) {
                str += strsprintf(_T("%02x"), buf[i]);
            }
            return str;
        };

        auto err = m_vulkan->Init(deviceID, extInstance, extDevice, m_log, !error_if_fail);
        if (err != RGY_ERR_NONE) {
            writeLog(RGY_LOG_WARN, _T("Failed to init Vulkan device #%d: %s\n"), deviceID, get_err_mes(err));
            m_vulkan.reset();
        } else {
            writeLog(RGY_LOG_DEBUG, _T("Init Vulkan device %d, name %s, uuid %s.\n"), deviceID, char_to_tstring(m_vulkan->GetDisplayDeviceName()).c_str(), hexString(m_vulkan->GetUUID()).c_str());

            int deviceCount = 0;
            cuResult = cuDeviceGetCount(&deviceCount);
            if (cuResult != CUDA_SUCCESS) {
                writeLog(error_level, _T("  Error: cuDeviceGetCount(): %s\n"), char_to_tstring(_cudaGetErrorEnum(cuResult)).c_str());
                m_vulkan.reset();
            } else {
                bool deviceFound = false;
                for (int idev = 0; idev < deviceCount; idev++) {
                    cuResult = cuDeviceGet(&cuDevice, idev);
                    if (cuResult != CUDA_SUCCESS) {
                        writeLog(error_level, _T("  Error: cuDeviceGet(%d): %s\n"), idev, char_to_tstring(_cudaGetErrorEnum(cuResult)).c_str());
                        break;
                    }
                    CUuuid cuuuid;
                    cuResult = cuDeviceGetUuid(&cuuuid, cuDevice);
                    if (cuResult != CUDA_SUCCESS) {
                        writeLog(error_level, _T("  Error: cuDeviceGetUuid(%d): %s\n"), idev, char_to_tstring(_cudaGetErrorEnum(cuResult)).c_str());
                        break;
                    }
                    writeLog(RGY_LOG_DEBUG, _T("  cuDeviceGetUuid(%d): success: %s\n"), idev, hexString(cuuuid.bytes).c_str());
                    if (memcmp(m_vulkan->GetUUID(), cuuuid.bytes, VK_UUID_SIZE) == 0) {
                        deviceFound = true;
                        break;
                    }
                    cuDevice = 0;
                }
                if (!deviceFound) {
                    writeLog(error_level, _T("Failed to init CUDA device #%d from Vulkan device.\n"), deviceID);
                    m_vulkan.reset();
                }
            }
        }
    }
    if (!m_vulkan)
#endif // #if ENABLE_VULKAN
    {
        writeLog(RGY_LOG_DEBUG, _T("checking for CUDA device #%d.\n"), deviceID);
        cuResult = cuDeviceGet(&cuDevice, deviceID);
        if (cuResult != CUDA_SUCCESS) {
            writeLog(error_level, _T("  Error: cuDeviceGet(%d): %s\n"), deviceID, char_to_tstring(_cudaGetErrorEnum(cuResult)).c_str());
            return RGY_ERR_DEVICE_NOT_FOUND;
        }
        writeLog(RGY_LOG_DEBUG, _T("  cuDeviceGet(%d): success\n"), deviceID);
    }

    if ((cuResult = cuDeviceGetName(dev_name, _countof(dev_name), cuDevice)) != CUDA_SUCCESS) {
        writeLog(error_level, _T("  Error: cuDeviceGetName(%d): %s\n"), deviceID, char_to_tstring(_cudaGetErrorEnum(cuResult)).c_str());
        return RGY_ERR_DEVICE_NOT_AVAILABLE;
    }
    writeLog(RGY_LOG_DEBUG, _T("  cuDeviceGetName(%d): %s\n"), deviceID, char_to_tstring(dev_name).c_str());
    int cudaDevMajor = 0, cudaDevMinor = 0;
    GETATTRIB_CHECK(cudaDevMajor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, cuDevice);
    GETATTRIB_CHECK(cudaDevMinor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, cuDevice);

    if (((cudaDevMajor << 4) + cudaDevMinor) < 0x30) {
        writeLog(error_level, _T("  Error: device does not satisfy required CUDA version (>=3.0): %d.%d\n"), cudaDevMajor, cudaDevMinor);
        return RGY_ERR_UNSUPPORTED;
    }
    writeLog(RGY_LOG_DEBUG, _T("  cudaDeviceGetAttribute: CUDA %d.%d\n"), cudaDevMajor, cudaDevMinor);

    if (!disableNVML) {
        cuResult = cuDeviceGetPCIBusId(pci_bus_name, sizeof(pci_bus_name), cuDevice);
        if (cuResult == CUDA_ERROR_INVALID_DEVICE || cuResult == CUDA_ERROR_INVALID_VALUE) {
            writeLog((error_if_fail) ? RGY_LOG_WARN : RGY_LOG_DEBUG, _T("  Warn: cuDeviceGetPCIBusId(): %s\n"), char_to_tstring(_cudaGetErrorEnum(cuResult)).c_str());
        } else {
            writeLog(RGY_LOG_DEBUG, _T("  PCIBusId: %s\n"), char_to_tstring(pci_bus_name).c_str());
        }
    }

    int clockRate = 0, multiProcessorCount = 0;
    GETATTRIB_CHECK(clockRate, CU_DEVICE_ATTRIBUTE_CLOCK_RATE, cuDevice);
    GETATTRIB_CHECK(multiProcessorCount, CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT, cuDevice);

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
            writeLog(RGY_LOG_ERROR, _T("Insufficient NVIDIA driver version, Required %d.%02d, Installed %d.%02d\n"),
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
        m_cuvid_csp = getHWDecCodecCsp(skipHWDecodeCheck);
    }

    // DeviceFeature取得のため、一時的なencoder sessionを作成する
    // session数の上限に達するのを防ぐため、featureを取得したらすぐに破棄する
    auto encoder = std::make_unique<NVEncoder>(cuCtxCreated, m_log);
    auto nvsts = encoder->InitSession();
    if (nvsts != NV_ENC_SUCCESS) {
        writeLog(RGY_LOG_ERROR, _T("Failed to init encoder session for getting features.\n"));
        return RGY_ERR_UNSUPPORTED;
    }
    nvsts = encoder->createDeviceFeatureList();
    if (nvsts != NV_ENC_SUCCESS) {
        writeLog(RGY_LOG_ERROR, _T("Failed to create device codec list.\n"));
        return RGY_ERR_UNSUPPORTED;
    }
    writeLog(RGY_LOG_DEBUG, _T("  createDeviceFeatureList\n"));
    m_nvenc_codec_features = encoder->GetNVEncCapability();
    return RGY_ERR_NONE;
}

RGY_ERR NVGPUInfo::initEncoder() {
    m_encoder = std::make_unique<NVEncoder>(m_cuCtx.get(), m_log);
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
    return RGY_ERR_NONE;
}

int NVGPUInfo::encoder_count(const GUID& codecGUID) const {
    const auto codecFeatures = nvenc_codec_features();
    for (const auto& codecFeature : codecFeatures) {
        if (codecFeature.codec == codecGUID) {
            return std::max(codecFeature.getCapLimit(NV_ENC_CAPS_NUM_ENCODER_ENGINES), 1);
        }
    }
    return 0;
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

NVEncCtrl::NVEncCtrl() :
    m_pLog(0),
    m_nDeviceId(-1) {
};

NVEncCtrl::~NVEncCtrl() {};

#pragma warning(push)
#pragma warning(disable:4100)
void NVEncCtrl::PrintMes(RGYLogLevel logLevel, const TCHAR *format, ...) {
    if (m_pLog.get() == nullptr) {
        if (logLevel <= RGY_LOG_INFO) {
            return;
        }
    } else if (logLevel < m_pLog->getLogLevel(RGY_LOGT_APP)) {
        return;
    }

    va_list args;
    va_start(args, format);

    int len = _vsctprintf(format, args) + 1; // _vscprintf doesn't count terminating '\0'
    vector<TCHAR> buffer(len, 0);
    _vstprintf_s(buffer.data(), len, format, args);
    va_end(args);

    if (m_pLog.get() != nullptr) {
        m_pLog->write(logLevel, RGY_LOGT_APP, buffer.data());
    } else {
        _ftprintf(stderr, _T("%s"), buffer.data());
    }
}

RGY_ERR NVEncCtrl::Initialize(const int deviceID, const RGYParamInitVulkan enableVulkan, RGYLogLevel logLevel) {
    RGY_ERR sts = RGY_ERR_NONE;

    initLogLevel(logLevel);

#if ENABLE_VULKAN
    if (enableVulkan == RGYParamInitVulkan::TargetVendor) {
        setenv("VK_LOADER_DRIVERS_SELECT", "*nvidia*", 1);
    }
#endif

    //m_pDeviceを初期化
    if (!check_if_nvcuda_dll_available()) {
        PrintMes(RGY_LOG_ERROR,
            FOR_AUO ? _T("CUDAが使用できないため、NVEncによるエンコードが行えません。(check_if_nvcuda_dll_available)\n") : _T("CUDA not available.\n"));
        return RGY_ERR_UNSUPPORTED;
    }
    m_nDeviceId = deviceID;

    if (RGY_ERR_NONE != (sts = InitCuda())) {
        PrintMes(RGY_LOG_ERROR, FOR_AUO ? _T("Cudaの初期化に失敗しました。\n") : _T("Failed to initialize CUDA.\n"));
        return sts;
    }
    PrintMes(RGY_LOG_DEBUG, _T("InitCuda: Success.\n"));
    return RGY_ERR_NONE;
}

RGY_ERR NVEncCtrl::initLogLevel(RGYLogLevel loglevel) {
    m_pLog.reset(new RGYLog(nullptr, loglevel));
    return RGY_ERR_NONE;
}

RGY_ERR NVEncCtrl::initLogLevel(const RGYParamLogLevel& loglevel) {
    m_pLog.reset(new RGYLog(nullptr, loglevel));
    return RGY_ERR_NONE;
}

RGY_ERR NVEncCtrl::InitCuda() {
    //ひとまず、これまでのすべてのエラーをflush
    auto cudaerr = cudaGetLastError();
    PrintMes(RGY_LOG_DEBUG, _T("InitCuda: device #%d.\n"), m_nDeviceId);

    PrintMes(RGY_LOG_DEBUG, _T("\n"), m_nDeviceId);
    PrintMes(RGY_LOG_DEBUG, _T("Checking Environment Info...\n"));
    PrintMes(RGY_LOG_DEBUG, _T("%s\n"), get_encoder_version());
    PrintMes(RGY_LOG_DEBUG, _T("OS Version     %s\n"), getOSVersion().c_str());

    TCHAR cpu_info[1024] = { 0 };
    getCPUInfo(cpu_info, _countof(cpu_info));
    PrintMes(RGY_LOG_DEBUG, _T("CPU            %s\n"), cpu_info);

    //ひとまず、これまでのすべてのエラーをflush
    cudaerr = cudaGetLastError();

    CUresult cuResult;
    if (CUDA_SUCCESS != (cuResult = cuInit(0))) {
        PrintMes(RGY_LOG_ERROR, _T("cuInit error:0x%x (%s)\n"), cuResult, char_to_tstring(_cudaGetErrorEnum(cuResult)).c_str());
        return err_to_rgy(cuResult);
    }
    PrintMes(RGY_LOG_DEBUG, _T("cuInit: Success.\n"));

    if (CUDA_SUCCESS != (cuResult = cuvidInit(0))) {
        PrintMes(RGY_LOG_ERROR, _T("cuvidInit error:0x%x (%s)\n"), cuResult, char_to_tstring(_cudaGetErrorEnum(cuResult)).c_str());
        return err_to_rgy(cuResult);
    }
    PrintMes(RGY_LOG_DEBUG, _T("cuvidInit: Success.\n"));
    return RGY_ERR_NONE;
}

RGY_ERR NVEncCtrl::ShowDeviceList(const int cudaSchedule, const bool skipHWDecodeCheck) {
    RGY_ERR sts = RGY_ERR_NONE;
    std::vector<std::unique_ptr<NVGPUInfo>> gpuList;
    if (RGY_ERR_NONE != (sts = InitDeviceList(gpuList, cudaSchedule, true, RGYParamInitVulkan::TargetVendor, skipHWDecodeCheck, false))) {
        PrintMes(RGY_LOG_ERROR, FOR_AUO ? _T("Cudaの初期化に失敗しました。\n") : _T("Failed to initialize CUDA.\n"));
        return sts;
    }
    PrintMes(RGY_LOG_DEBUG, _T("InitDeviceList: Success.\n"));
    if (0 == gpuList.size()) {
        _ftprintf(stdout, _T("No GPU found suitable for NVEnc Encoding.\n"));
        return RGY_ERR_DEVICE_NOT_FOUND;
    }

    for (const auto &gpu : gpuList) {
        _ftprintf(stdout, _T("DeviceId #%d: %s\n"), gpu->id(), gpu->name().c_str());
    }
    return RGY_ERR_NONE;
}

RGY_ERR NVEncCtrl::ShowCodecSupport(const int cudaSchedule, const bool skipHWDecodeCheck) {
    RGY_ERR sts = RGY_ERR_NONE;
    std::vector<std::unique_ptr<NVGPUInfo>> gpuList;
    if (RGY_ERR_NONE != (sts = InitDeviceList(gpuList, cudaSchedule, true, RGYParamInitVulkan::TargetVendor, skipHWDecodeCheck, false))) {
        PrintMes(RGY_LOG_ERROR, FOR_AUO ? _T("Cudaの初期化に失敗しました。\n") : _T("Failed to initialize CUDA.\n"));
        return sts;
    }
    PrintMes(RGY_LOG_DEBUG, _T("InitDeviceList: Success.\n"));

    auto gpu = std::find_if(gpuList.begin(), gpuList.end(), [device_id = m_nDeviceId](const std::unique_ptr<NVGPUInfo> &gpuinfo) {
        return gpuinfo->id() == device_id;
        });
    if (gpu == gpuList.end()) {
        PrintMes(RGY_LOG_ERROR, _T("Selected device #%d not found\n"), m_nDeviceId);
        return RGY_ERR_INVALID_DEVICE;
    }
    _ftprintf(stdout, _T("%s\n"), (*gpu)->infostr().c_str());
    auto nvEncCaps = (*gpu)->nvenc_codec_features();
    if (nvEncCaps.size()) {
        _ftprintf(stdout, _T("Avaliable Codec(s)\n"));
        for (auto codecNVEncCaps : nvEncCaps) {
            _ftprintf(stdout, _T("%s\n"), get_name_from_guid(codecNVEncCaps.codec, list_nvenc_codecs));
        }
    } else {
        _ftprintf(stdout, _T("No NVEnc support.\n"));
        return RGY_ERR_UNSUPPORTED;
    }
    return RGY_ERR_NONE;
}

RGY_ERR NVEncCtrl::ShowNVEncFeatures(const int cudaSchedule, const bool skipHWDecodeCheck) {
    RGY_ERR sts = RGY_ERR_NONE;
    std::vector<std::unique_ptr<NVGPUInfo>> gpuList;
    if (RGY_ERR_NONE != (sts = InitDeviceList(gpuList, cudaSchedule, true, RGYParamInitVulkan::TargetVendor, skipHWDecodeCheck, false))) {
        PrintMes(RGY_LOG_ERROR, FOR_AUO ? _T("Cudaの初期化に失敗しました。\n") : _T("Failed to initialize CUDA.\n"));
        return sts;
    }
    PrintMes(RGY_LOG_DEBUG, _T("InitDeviceList: Success.\n"));

    auto gpu = std::find_if(gpuList.begin(), gpuList.end(), [device_id = m_nDeviceId](const std::unique_ptr<NVGPUInfo> &gpuinfo) {
        return gpuinfo->id() == device_id;
        });
    if (gpu == gpuList.end()) {
        PrintMes(RGY_LOG_ERROR, _T("Selected device #%d not found\n"), m_nDeviceId);
        return RGY_ERR_INVALID_DEVICE;
    }
    _ftprintf(stdout, _T("%s\n"), (*gpu)->infostr().c_str());
    auto nvEncCaps = (*gpu)->nvenc_codec_features();
    if (nvEncCaps.size() == 0) {
        _ftprintf(stdout, _T("No NVEnc support.\n"));
        sts = RGY_ERR_UNSUPPORTED;
    } else {
        _ftprintf(stdout, _T("NVEnc features\n"));
        for (auto codecNVEncCaps : nvEncCaps) {
            _ftprintf(stdout, _T("Codec: %s\n"), get_name_from_guid(codecNVEncCaps.codec, list_nvenc_codecs));
            size_t max_length = 0;
            std::for_each(codecNVEncCaps.caps.begin(), codecNVEncCaps.caps.end(), [&max_length](const NVEncCap &x) { max_length = (std::max)(max_length, _tcslen(x.name)); });
            for (const auto& cap : codecNVEncCaps.caps) {
                _ftprintf(stdout, _T("%s"), cap.name);
                for (size_t i = _tcslen(cap.name); i <= max_length; i++) {
                    _ftprintf(stdout, _T(" "));
                }
                if (cap.isBool) {
                    _ftprintf(stdout, cap.value ? _T("yes\n") : _T("no\n"));
                } else if (cap.desc) {
                    _ftprintf(stdout, _T("%d (%s)\n"), cap.value, get_cx_desc(cap.desc, cap.value));
                } else if (cap.desc_bit_flag) {
                    tstring bit_flag;
                    for (int i = 0; cap.desc_bit_flag[i].desc; i++) {
                        const uint32_t bitflag = cap.desc_bit_flag[i].value;
                        if (((uint32_t)cap.value & bitflag) == bitflag) {
                            if (bit_flag.length() > 0) bit_flag += _T(", ");
                            bit_flag += cap.desc_bit_flag[i].desc;
                        }
                    }
                    if (bit_flag.empty()) {
                        _ftprintf(stdout, _T("%d\n"), cap.value);
                    } else {
                        _ftprintf(stdout, _T("%d (%s)\n"), cap.value, bit_flag.c_str());
                    }
                } else {
                    _ftprintf(stdout, _T("%d\n"), cap.value);
                }
            }
            _ftprintf(stdout, _T("\n"));
        }
    }
    const auto cuvidCodecCsp = (*gpu)->cuvid_csp();
    if (cuvidCodecCsp.size() == 0) {
        _ftprintf(stdout, _T("No NVDec support.\n"));
    } else {
        _ftprintf(stdout, _T("\nNVDec features\n"));
        size_t max_length = 0;
        std::for_each(cuvidCodecCsp.begin(), cuvidCodecCsp.end(), [&max_length](const decltype(cuvidCodecCsp)::value_type  &codeccsp) { max_length = (std::max)(max_length, CodecToStr(codeccsp.first).length()); });
        for (auto codeccsp : cuvidCodecCsp) {
            tstring csps = CodecToStr(codeccsp.first) + _T(":");
            for (size_t i = csps.length()-1; i <= max_length; i++) {
                csps += _T(" ");
            }
            for (auto csp : codeccsp.second) {
                csps += tstring(RGY_CSP_NAMES[csp]) + _T(", ");
            }
            _ftprintf(stdout, _T("  %s\n"), csps.substr(0, csps.length()-2).c_str());
        }
    }
    return sts;
}

RGY_ERR NVEncCtrl::InitDeviceList(std::vector<std::unique_ptr<NVGPUInfo>>& gpuList, const int cudaSchedule, bool initDX11, RGYParamInitVulkan initVulkan, const bool skipHWDecodeCheck, const int disableNVML) {
    int deviceCount = 0;
#if ENABLE_D3D11
    if (initDX11) {
        deviceCount = DX11AdapterManager::getInstance(m_pLog.get())->adapterCount();
        if (deviceCount == 0) {
            PrintMes(RGY_LOG_WARN, _T("Failed to get device count from DX11 interface.\n"));
            initDX11 = false;
        }
        PrintMes(RGY_LOG_DEBUG, _T("DX11 device count: %d.\n"), deviceCount);
    }
#else
    initDX11 = false;
#endif
#if ENABLE_VULKAN
    if (initVulkan == RGYParamInitVulkan::TargetVendor) {
        setenv("VK_LOADER_DRIVERS_SELECT", "*nvidia*", 1);
    }
    if (initVulkan != RGYParamInitVulkan::Disable) {
        DeviceVulkan vulkan;
        deviceCount = vulkan.adapterCount();
        if (deviceCount == 0) {
            PrintMes(RGY_LOG_WARN, _T("Failed to get device count from Vulkan interface.\n"));
            initVulkan = RGYParamInitVulkan::Disable;
        }
        PrintMes(RGY_LOG_DEBUG, _T("vulkan.adapterCount: Success, %d.\n"), deviceCount);
    }
#else
    initVulkan = RGYParamInitVulkan::Disable;
#endif
    if (deviceCount == 0) {
        auto cuResult = cuDeviceGetCount(&deviceCount);
        if (cuResult != CUDA_SUCCESS) {
            PrintMes(RGY_LOG_ERROR, _T("cuDeviceGetCount error:0x%x (%s)\n"), cuResult, char_to_tstring(_cudaGetErrorEnum(cuResult)).c_str());
            return RGY_ERR_NO_DEVICE;
        }
        if (deviceCount == 0) {
            PrintMes(RGY_LOG_ERROR, _T("Error: no CUDA device.\n"));
            return RGY_ERR_NO_DEVICE;
        }
        PrintMes(RGY_LOG_DEBUG, _T("cuDeviceGetCount: Success, %d.\n"), deviceCount);
    }
    if (m_nDeviceId > deviceCount - 1) {
        PrintMes(RGY_LOG_ERROR, _T("Invalid Device Id = %d\n"), m_nDeviceId);
        return RGY_ERR_INVALID_DEVICE;
    }

    const bool disableNVMLCheck = (disableNVML > 1 || (disableNVML == 1 && deviceCount > 1));

    gpuList.clear();
    for (int currentDevice = 0; currentDevice < deviceCount; currentDevice++) {
        cudaGetLastError(); //これまでのエラーを初期化
        if ((m_nDeviceId < 0 || m_nDeviceId == currentDevice)) {
            auto gpu = std::make_unique<NVGPUInfo>(m_pLog);
            if (gpu->initDevice(currentDevice, (CUctx_flags)cudaSchedule, m_nDeviceId == currentDevice, initDX11, initVulkan, skipHWDecodeCheck, disableNVMLCheck) == RGY_ERR_NONE) {
                gpuList.push_back(std::move(gpu));
            }
        }
    }
    if (gpuList.size() == 0) {
        PrintMes(RGY_LOG_ERROR, _T("No GPU found suitable for NVEnc Encoding.\n"));
        return RGY_ERR_NO_DEVICE;
    }
    return RGY_ERR_NONE;
}
