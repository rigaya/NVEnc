//  -----------------------------------------------------------------------------------------
//    NVEnc by rigaya
//  -----------------------------------------------------------------------------------------
//   ソースコードについて
//   ・無保証です。
//   ・本ソースコードを使用したことによるいかなる損害・トラブルについてrigayaは責任を負いません。
//   以上に了解して頂ける場合、本ソースコードの使用、複製、改変、再頒布を行って頂いて構いません。
//  -----------------------------------------------------------------------------------------

#include <vector>
#include <string>
#include <algorithm>
#include <thread>
#include <tchar.h>
#include <cuda.h>
#include "process.h"
#pragma comment(lib, "winmm.lib")
#include "nvEncodeAPI.h"
#include "NVEncCore.h"
#include "NVEncVersion.h"
#include "NVEncStatus.h"
#include "NVEncParam.h"
#include "NVEncUtil.h"
#include "NVEncInput.h"
#include "NVEncInputRaw.h"
#include "NVEncInputAvs.h"
#include "NVEncInputVpy.h"
#include "helper_nvenc.h"
#include "shlwapi.h"
#pragma comment(lib, "shlwapi.lib")

bool check_if_nvcuda_dll_available() {
    //check for nvcuda.dll
    HMODULE hModule = LoadLibrary(_T("nvcuda.dll"));
    if (hModule == NULL)
        return false;
    FreeLibrary(hModule);
    return true;
}

NVEncoderGPUInfo::NVEncoderGPUInfo() {
    CUresult cuResult = CUDA_SUCCESS;

    if (!check_if_nvcuda_dll_available())
        return;

    if (CUDA_SUCCESS != (cuResult = cuInit(0)))
        return;

    int deviceCount = 0;
    if (CUDA_SUCCESS != (cuDeviceGetCount(&deviceCount)) || 0 == deviceCount)
        return;

    for (int currentDevice = 0; currentDevice < deviceCount; currentDevice++) {
        char gpu_name[1024] = { 0 };
        int SMminor = 0, SMmajor = 0;
        CUdevice cuDevice = 0;
        if (   CUDA_SUCCESS == cuDeviceGet(&cuDevice, currentDevice)
            && CUDA_SUCCESS == cuDeviceGetName(gpu_name, _countof(gpu_name), cuDevice)
            && CUDA_SUCCESS == cuDeviceComputeCapability(&SMmajor, &SMminor, currentDevice)
            && (((SMmajor << 4) + SMminor) >= 0x30)) {
            GPUList.push_back(std::make_pair(currentDevice, to_tchar(gpu_name)));
        }
    }
};

NVEncoderGPUInfo::~NVEncoderGPUInfo() {
};

NVEncCore::NVEncCore() {
    m_pEncodeAPI = nullptr;
    m_hinstLib = NULL;
    m_hEncoder = nullptr;
    m_fOutput = nullptr;
    m_nLogLevel = NV_LOG_INFO;
    m_pStatus = nullptr;
    m_pInput = nullptr;
    m_uEncodeBufferCount = 16;
    m_pOutputBuf = nullptr;
    m_pDevice = nullptr;

    INIT_CONFIG(m_stCreateEncodeParams, NV_ENC_INITIALIZE_PARAMS);
    INIT_CONFIG(m_stEncConfig, NV_ENC_CONFIG);

    memset(&m_stEOSOutputBfr, 0, sizeof(m_stEOSOutputBfr));
    memset(&m_stEncodeBuffer, 0, sizeof(m_stEncodeBuffer));
}

NVEncCore::~NVEncCore() {
    Deinitialize();

    if (m_pEncodeAPI) {
        delete m_pEncodeAPI;
        m_pEncodeAPI = nullptr;
    }

    if (m_hinstLib) {
        FreeLibrary(m_hinstLib);
        m_hinstLib = NULL;
    }

    if (m_pOutputBuf) {
        free(m_pOutputBuf);
        m_pOutputBuf = nullptr;
    }
}

#pragma warning(push)
#pragma warning(disable:4100)
int NVEncCore::NVPrintf(FILE *fp, int logLevel, const TCHAR *format, ...) {
    if (logLevel < m_nLogLevel)
        return 0;

    va_list args;
    va_start(args, format);

    int len = _vsctprintf(format, args);
    TCHAR *const buffer = (TCHAR*)malloc((len+1) * sizeof(buffer[0])); // _vscprintf doesn't count terminating '\0'

    _vstprintf_s(buffer, len+1, format, args);
    
#ifdef UNICODE
    DWORD mode = 0;
    bool write_with_wchar = fp == stderr && 0 != GetConsoleMode(GetStdHandle(STD_ERROR_HANDLE), &mode); //stderrの出力先がコンソールかどうか

    if (!write_with_wchar) {
        char *buffer_char = nullptr;
        //出力先がリダイレクトされるならANSIで
        if (nullptr != (buffer_char = (char *)calloc(len * 2, sizeof(buffer_char[0])))) {
            WideCharToMultiByte(CP_THREAD_ACP, WC_NO_BEST_FIT_CHARS, buffer, -1, buffer_char, len * 2, NULL, NULL);
            fprintf(fp, buffer_char);
            free(buffer_char);
        }
    }
    if (write_with_wchar) //出力先がコンソールならWCHARで
#endif
        _ftprintf(fp, buffer);

    return len;
}

void NVEncCore::NVPrintFuncError(const TCHAR *funcName, NVENCSTATUS nvStatus) {
    NVPrintf(stderr, NV_LOG_ERROR, (FOR_AUO) ? _T("%s() がエラーを返しました。: %d (%s)\n") : _T("Error on %s: %d (%s)\n") , funcName, nvStatus, to_tchar(_nvencGetErrorEnum(nvStatus)).c_str());
}

NVENCSTATUS NVEncCore::InitInput(InEncodeVideoParam *inputParam) {
#if RAW_READER
    if (inputParam->input.type == NV_ENC_INPUT_UNKNWON) {
        if (0 == _tcsicmp(_T(".y4m"), PathFindExtension(inputParam->input.filename.c_str()))) {
            inputParam->input.type = NV_ENC_INPUT_Y4M;
#if AVI_READER
        } else if (0 == _tcsicmp(_T(".avi"), PathFindExtension(inputParam->input.filename.c_str()))) {
            inputParam->input.type = NV_ENC_INPUT_AVI;
#endif
#if AVS_READER
        } else if (0 == _tcsicmp(_T(".avs"), PathFindExtension(inputParam->input.filename.c_str()))) {
            inputParam->input.type = NV_ENC_INPUT_AVS;
#endif
#if VPY_READER
        } else if (0 == _tcsicmp(_T(".vpy"), PathFindExtension(inputParam->input.filename.c_str()))) {
            inputParam->input.type = NV_ENC_INPUT_VPY_MT;
#endif
        } else {
            inputParam->input.type = NV_ENC_INPUT_RAW;
        }
    }
    
#if AVS_READER
    InputInfoAvs inputInfoAvs = { 0 };
#endif
#if VPY_READER
    InputInfoVpy inputInfoVpy = { 0 };
#endif

    switch (inputParam->input.type) {
#if AVS_READER
    case NV_ENC_INPUT_AVS:
        inputInfoAvs.interlaced = is_interlaced(inputParam->picStruct);
        inputParam->input.otherPrm = &inputInfoAvs;
        m_pInput = new NVEncInputAvs();
        break;
#endif //AVS_READER
#if VPY_READER
    case NV_ENC_INPUT_VPY:
    case NV_ENC_INPUT_VPY_MT:
        inputInfoVpy.interlaced = is_interlaced(inputParam->picStruct);
        inputInfoVpy.mt = (inputParam->input.type == NV_ENC_INPUT_VPY_MT);
        inputParam->input.otherPrm = &inputInfoVpy;
        m_pInput = new NVEncInputVpy();
        break;
#endif //VPY_READER
    case NV_ENC_INPUT_RAW:
    case NV_ENC_INPUT_Y4M:
    default:
        m_pInput = new NVEncInputRaw();
        break;
    }

    m_pStatus = new EncodeStatus();
    int ret = m_pInput->Init(&inputParam->input, m_pStatus);
    m_pStatus->m_nOutputFPSRate = inputParam->input.rate;
    m_pStatus->m_nOutputFPSScale = inputParam->input.scale;
    return (ret) ? NV_ENC_ERR_GENERIC : NV_ENC_SUCCESS;
#else
    return NV_ENC_ERR_INVALID_CALL;
#endif //RAW_READER
}
#pragma warning(pop)

NVENCSTATUS NVEncCore::InitCuda(uint32_t deviceID) {
    CUresult cuResult;
    if (CUDA_SUCCESS != (cuResult = cuInit(0))) {
        NVPrintf(stderr, NV_LOG_ERROR, _T("cuInit error:0x%x\n"), cuResult);
        return NV_ENC_ERR_NO_ENCODE_DEVICE;
    }
    
    int deviceCount = 0;
    if (CUDA_SUCCESS != (cuResult = cuDeviceGetCount(&deviceCount))) {
        NVPrintf(stderr, NV_LOG_ERROR, _T("cuDeviceGetCount error:0x%x\n"), cuResult);
        return NV_ENC_ERR_NO_ENCODE_DEVICE;
    }

    deviceID = max(0, deviceID);

    if (deviceID > (unsigned int)deviceCount - 1) {
        NVPrintf(stderr, NV_LOG_ERROR, _T("Invalid Device Id = %d\n"), deviceID);
        return NV_ENC_ERR_INVALID_ENCODERDEVICE;
    }
    
    CUdevice device;
    if (CUDA_SUCCESS != (cuResult = cuDeviceGet(&device, deviceID))) {
        NVPrintf(stderr, NV_LOG_ERROR, _T("cuDeviceGet error:0x%x\n"), cuResult);
        return NV_ENC_ERR_NO_ENCODE_DEVICE;
    }
    
    int SMminor = 0, SMmajor = 0;
    if (CUDA_SUCCESS != (cuDeviceComputeCapability(&SMmajor, &SMminor, deviceID))) {
        NVPrintf(stderr, NV_LOG_ERROR, _T("cuDeviceComputeCapability error:0x%x\n"), cuResult);
        return NV_ENC_ERR_NO_ENCODE_DEVICE;
    }

    if (((SMmajor << 4) + SMminor) < 0x30) {
        NVPrintf(stderr, NV_LOG_ERROR, _T("GPU %d does not have NVENC capabilities exiting\n"), deviceID);
        return NV_ENC_ERR_NO_ENCODE_DEVICE;
    }

    if (CUDA_SUCCESS != (cuResult = cuCtxCreate((CUcontext*)(&m_pDevice), 0, device))) {
        NVPrintf(stderr, NV_LOG_ERROR, _T("cuCtxCreate error:0x%x\n"), cuResult);
        return NV_ENC_ERR_NO_ENCODE_DEVICE;
    }
    
    CUcontext cuContextCurr;
    if (CUDA_SUCCESS != (cuResult = cuCtxPopCurrent(&cuContextCurr))) {
        NVPrintf(stderr, NV_LOG_ERROR, _T("cuCtxPopCurrent error:0x%x\n"), cuResult);
        return NV_ENC_ERR_NO_ENCODE_DEVICE;
    }
    return NV_ENC_SUCCESS;
}

NVENCSTATUS NVEncCore::NvEncCreateInputBuffer(uint32_t width, uint32_t height, void** inputBuffer, uint32_t isYuv444) {
    NV_ENC_CREATE_INPUT_BUFFER createInputBufferParams;
    INIT_CONFIG(createInputBufferParams, NV_ENC_CREATE_INPUT_BUFFER);

    createInputBufferParams.width = width;
    createInputBufferParams.height = height;
    createInputBufferParams.memoryHeap = NV_ENC_MEMORY_HEAP_SYSMEM_CACHED;
    createInputBufferParams.bufferFmt = isYuv444 ? NV_ENC_BUFFER_FORMAT_YUV444_PL : NV_ENC_BUFFER_FORMAT_NV12_PL;

    NVENCSTATUS nvStatus = m_pEncodeAPI->nvEncCreateInputBuffer(m_hEncoder, &createInputBufferParams);
    if (nvStatus != NV_ENC_SUCCESS) {
        NVPrintFuncError(_T("nvEncCreateInputBuffer"), nvStatus);
        return nvStatus;
    }

    *inputBuffer = createInputBufferParams.inputBuffer;

    return nvStatus;
}

NVENCSTATUS NVEncCore::NvEncDestroyInputBuffer(NV_ENC_INPUT_PTR inputBuffer) {
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

NVENCSTATUS NVEncCore::NvEncCreateBitstreamBuffer(uint32_t size, void** bitstreamBuffer) {
    NV_ENC_CREATE_BITSTREAM_BUFFER createBitstreamBufferParams;
    INIT_CONFIG(createBitstreamBufferParams, NV_ENC_CREATE_BITSTREAM_BUFFER);

    createBitstreamBufferParams.size = size;
    createBitstreamBufferParams.memoryHeap = NV_ENC_MEMORY_HEAP_SYSMEM_CACHED;

    NVENCSTATUS nvStatus = m_pEncodeAPI->nvEncCreateBitstreamBuffer(m_hEncoder, &createBitstreamBufferParams);
    if (nvStatus != NV_ENC_SUCCESS) {
        NVPrintFuncError(_T("nvEncCreateBitstreamBuffer"), nvStatus);
        return nvStatus;
    }

    *bitstreamBuffer = createBitstreamBufferParams.bitstreamBuffer;

    return nvStatus;
}

NVENCSTATUS NVEncCore::NvEncDestroyBitstreamBuffer(NV_ENC_OUTPUT_PTR bitstreamBuffer) {
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

NVENCSTATUS NVEncCore::NvEncLockBitstream(NV_ENC_LOCK_BITSTREAM* lockBitstreamBufferParams) {
    NVENCSTATUS nvStatus = m_pEncodeAPI->nvEncLockBitstream(m_hEncoder, lockBitstreamBufferParams);
    if (nvStatus != NV_ENC_SUCCESS) {
        NVPrintFuncError(_T("nvEncLockBitstream"), nvStatus);
        return nvStatus;
    }
    return nvStatus;
}

NVENCSTATUS NVEncCore::NvEncUnlockBitstream(NV_ENC_OUTPUT_PTR bitstreamBuffer) {
    NVENCSTATUS nvStatus = m_pEncodeAPI->nvEncUnlockBitstream(m_hEncoder, bitstreamBuffer);
    if (nvStatus != NV_ENC_SUCCESS) {
        NVPrintFuncError(_T("nvEncUnlockBitstream"), nvStatus);
        return nvStatus;
    }
    return nvStatus;
}

NVENCSTATUS NVEncCore::NvEncLockInputBuffer(void* inputBuffer, void** bufferDataPtr, uint32_t* pitch) {
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

NVENCSTATUS NVEncCore::NvEncUnlockInputBuffer(NV_ENC_INPUT_PTR inputBuffer) {
    NVENCSTATUS nvStatus = m_pEncodeAPI->nvEncUnlockInputBuffer(m_hEncoder, inputBuffer);
    if (nvStatus != NV_ENC_SUCCESS) {
        NVPrintFuncError(_T("nvEncUnlockInputBuffer"), nvStatus);
        return nvStatus;
    }
    return nvStatus;
}

NVENCSTATUS NVEncCore::NvEncGetEncodeStats(NV_ENC_STAT* encodeStats) {
    NVENCSTATUS nvStatus = m_pEncodeAPI->nvEncGetEncodeStats(m_hEncoder, encodeStats);
    if (nvStatus != NV_ENC_SUCCESS) {
        NVPrintFuncError(_T("nvEncGetEncodeStats"), nvStatus);
        return nvStatus;
    }

    return nvStatus;
}

NVENCSTATUS NVEncCore::NvEncGetSequenceParams(NV_ENC_SEQUENCE_PARAM_PAYLOAD* sequenceParamPayload) {
    NVENCSTATUS nvStatus = m_pEncodeAPI->nvEncGetSequenceParams(m_hEncoder, sequenceParamPayload);
    if (nvStatus != NV_ENC_SUCCESS) {
        NVPrintFuncError(_T("nvEncGetSequenceParams"), nvStatus);
        return nvStatus;
    }

    return nvStatus;
}

NVENCSTATUS NVEncCore::NvEncRegisterAsyncEvent(void** completionEvent) {
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

NVENCSTATUS NVEncCore::NvEncUnregisterAsyncEvent(void* completionEvent) {
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

NVENCSTATUS NVEncCore::NvEncMapInputResource(void* registeredResource, void** mappedResource) {
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

NVENCSTATUS NVEncCore::NvEncUnmapInputResource(NV_ENC_INPUT_PTR mappedInputBuffer) {
    if (mappedInputBuffer) {
        NVENCSTATUS nvStatus = m_pEncodeAPI->nvEncUnmapInputResource(m_hEncoder, mappedInputBuffer);
        if (nvStatus != NV_ENC_SUCCESS) {
            NVPrintFuncError(_T("nvEncUnmapInputResource"), nvStatus);
            return nvStatus;
        }
    }

    return NV_ENC_SUCCESS;
}

NVENCSTATUS NVEncCore::NvEncDestroyEncoder() {
    NVENCSTATUS nvStatus = NV_ENC_SUCCESS;

    if (m_hEncoder && m_pEncodeAPI) {
        nvStatus = m_pEncodeAPI->nvEncDestroyEncoder(m_hEncoder);
        m_hEncoder = NULL;
    }

    return nvStatus;
}

NVENCSTATUS NVEncCore::NvEncFlushEncoderQueue(void *hEOSEvent) {
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

NVENCSTATUS NVEncCore::ProcessOutput(const EncodeBuffer *pEncodeBuffer) {
    if (pEncodeBuffer->stOutputBfr.hBitstreamBuffer == NULL && pEncodeBuffer->stOutputBfr.bEOSFlag == FALSE) {
        return NV_ENC_ERR_INVALID_PARAM;
    }

    if (pEncodeBuffer->stOutputBfr.bWaitOnEvent == TRUE) {
        if (!pEncodeBuffer->stOutputBfr.hOutputEvent) {
            return NV_ENC_ERR_INVALID_PARAM;
        }
        WaitForSingleObject(pEncodeBuffer->stOutputBfr.hOutputEvent, INFINITE);
    }

    if (pEncodeBuffer->stOutputBfr.bEOSFlag)
        return NV_ENC_SUCCESS;

    NV_ENC_LOCK_BITSTREAM lockBitstreamData;
    INIT_CONFIG(lockBitstreamData, NV_ENC_LOCK_BITSTREAM);
    lockBitstreamData.outputBitstream = pEncodeBuffer->stOutputBfr.hBitstreamBuffer;
    lockBitstreamData.doNotWait = false;

    NVENCSTATUS nvStatus = m_pEncodeAPI->nvEncLockBitstream(m_hEncoder, &lockBitstreamData);
    if (nvStatus == NV_ENC_SUCCESS) {
        m_pStatus->AddOutputInfo(&lockBitstreamData);
        fwrite(lockBitstreamData.bitstreamBufferPtr, 1, lockBitstreamData.bitstreamSizeInBytes, m_fOutput);
        nvStatus = m_pEncodeAPI->nvEncUnlockBitstream(m_hEncoder, pEncodeBuffer->stOutputBfr.hBitstreamBuffer);
    } else {
        NVPrintFuncError(_T("nvEncLockBitstream"), nvStatus);
        return nvStatus;
    }

    return nvStatus;
}

NVENCSTATUS NVEncCore::FlushEncoder() {
    NVENCSTATUS nvStatus = NvEncFlushEncoderQueue(m_stEOSOutputBfr.hOutputEvent);
    if (nvStatus != NV_ENC_SUCCESS) {
        NVPrintFuncError(_T("NvEncFlushEncoderQueue"), nvStatus);
        return nvStatus;
    }

    EncodeBuffer *pEncodeBufer = m_EncodeBufferQueue.GetPending();
    while (pEncodeBufer) {
        ProcessOutput(pEncodeBufer);
        pEncodeBufer = m_EncodeBufferQueue.GetPending();
    }

    if (WaitForSingleObject(m_stEOSOutputBfr.hOutputEvent, 500) != WAIT_OBJECT_0) {
        NVPrintf(stderr, NV_LOG_ERROR, _T("m_stEOSOutputBfr.hOutputEvent%s"), (FOR_AUO) ? _T("が終了しません。") : _T(" does not finish within proper time."));
        nvStatus = NV_ENC_ERR_GENERIC;
    }

    return nvStatus;
}

NVENCSTATUS NVEncCore::Deinitialize() {
    NVENCSTATUS nvStatus = NV_ENC_SUCCESS;
    
    if (m_fOutput) {
        fclose(m_fOutput);
        m_fOutput = NULL;
    }
    if (m_pInput) {
        m_pInput->Close();
        delete m_pInput;
        m_pInput = nullptr;
    }

    ReleaseIOBuffers();

    nvStatus = NvEncDestroyEncoder();

    if (m_pDevice) {
        CUresult cuResult = CUDA_SUCCESS;
        cuResult = cuCtxDestroy((CUcontext)m_pDevice);
        if (cuResult != CUDA_SUCCESS)
            NVPrintf(stderr, NV_LOG_ERROR, _T("cuCtxDestroy error:0x%x\n"), cuResult);

        m_pDevice = NULL;
    }

    return nvStatus;
}

NVENCSTATUS NVEncCore::AllocateIOBuffers(uint32_t uInputWidth, uint32_t uInputHeight) {
    NVENCSTATUS nvStatus = NV_ENC_SUCCESS;

    m_EncodeBufferQueue.Initialize(m_stEncodeBuffer, m_uEncodeBufferCount);
    for (uint32_t i = 0; i < m_uEncodeBufferCount; i++) {
        nvStatus = NvEncCreateInputBuffer(uInputWidth, uInputHeight, &m_stEncodeBuffer[i].stInputBfr.hInputSurface, 0);
        if (nvStatus != NV_ENC_SUCCESS) {
            NVPrintf(stderr, NV_LOG_ERROR, _T("Failed to allocate Input Buffer, Please reduce MAX_FRAMES_TO_PRELOAD\n"));
            return nvStatus;
        }

        m_stEncodeBuffer[i].stInputBfr.bufferFmt = NV_ENC_BUFFER_FORMAT_NV12_PL;
        m_stEncodeBuffer[i].stInputBfr.dwWidth = uInputWidth;
        m_stEncodeBuffer[i].stInputBfr.dwHeight = uInputHeight;

        nvStatus = NvEncCreateBitstreamBuffer(BITSTREAM_BUFFER_SIZE, &m_stEncodeBuffer[i].stOutputBfr.hBitstreamBuffer);
        if (nvStatus != NV_ENC_SUCCESS) {
            NVPrintf(stderr, NV_LOG_ERROR, _T("Failed to allocate Output Buffer, Please reduce MAX_FRAMES_TO_PRELOAD\n"));
            return nvStatus;
        }
        m_stEncodeBuffer[i].stOutputBfr.dwBitstreamBufferSize = BITSTREAM_BUFFER_SIZE;

        nvStatus = NvEncRegisterAsyncEvent(&m_stEncodeBuffer[i].stOutputBfr.hOutputEvent);
        if (nvStatus != NV_ENC_SUCCESS)
            return nvStatus;
        m_stEncodeBuffer[i].stOutputBfr.bWaitOnEvent = true;
    }

    m_stEOSOutputBfr.bEOSFlag = TRUE;

    nvStatus = NvEncRegisterAsyncEvent(&m_stEOSOutputBfr.hOutputEvent);
    if (nvStatus != NV_ENC_SUCCESS)
        return nvStatus;

    return NV_ENC_SUCCESS;
}

NVENCSTATUS NVEncCore::ReleaseIOBuffers() {
    for (uint32_t i = 0; i < m_uEncodeBufferCount; i++) {
        if (m_stEncodeBuffer[i].stInputBfr.hInputSurface) {
            NvEncDestroyInputBuffer(m_stEncodeBuffer[i].stInputBfr.hInputSurface);
            m_stEncodeBuffer[i].stInputBfr.hInputSurface = NULL;
        }

        if (m_stEncodeBuffer[i].stOutputBfr.hBitstreamBuffer) {
            NvEncDestroyBitstreamBuffer(m_stEncodeBuffer[i].stOutputBfr.hBitstreamBuffer);
            m_stEncodeBuffer[i].stOutputBfr.hBitstreamBuffer = NULL;
        }
        if (m_stEncodeBuffer[i].stOutputBfr.hOutputEvent) {
            NvEncUnregisterAsyncEvent(m_stEncodeBuffer[i].stOutputBfr.hOutputEvent);
            nvCloseFile(m_stEncodeBuffer[i].stOutputBfr.hOutputEvent);
            m_stEncodeBuffer[i].stOutputBfr.hOutputEvent = NULL;
        }
    }

    if (m_stEOSOutputBfr.hOutputEvent) {
        NvEncUnregisterAsyncEvent(m_stEOSOutputBfr.hOutputEvent);
        nvCloseFile(m_stEOSOutputBfr.hOutputEvent);
        m_stEOSOutputBfr.hOutputEvent = NULL;
    }

    return NV_ENC_SUCCESS;
}

NVENCSTATUS NVEncCore::InitDevice(const InEncodeVideoParam *inputParam) {
    if (!check_if_nvcuda_dll_available()) {
        NVPrintf(stderr, NV_LOG_ERROR,
            FOR_AUO ? _T("CUDAが使用できないため、NVEncによるエンコードが行えません。(check_if_nvcuda_dll_available)\n") : _T("CUDA not available.\n"));
        return NV_ENC_ERR_UNSUPPORTED_DEVICE;
    }

    NVEncoderGPUInfo gpuInfo;
    if (0 == gpuInfo.getGPUList().size()) {
        NVPrintf(stderr, NV_LOG_ERROR, FOR_AUO ? _T("NVEncが使用可能なGPUが見つかりませんでした。\n") : _T("No GPU found suitable for NVEnc Encoding.\n"));
        return NV_ENC_ERR_NO_ENCODE_DEVICE;
    }

    NVENCSTATUS nvStatus;
    if (NV_ENC_SUCCESS != (nvStatus = InitCuda(inputParam->deviceID))) {
        NVPrintf(stderr, NV_LOG_ERROR, FOR_AUO ? _T("Cudaの初期化に失敗しました。\n") : _T("Failed to initialize CUDA.\n"));
        return nvStatus;
    }
    return NV_ENC_SUCCESS;
}

NVENCSTATUS NVEncCore::NvEncOpenEncodeSessionEx(void* device, NV_ENC_DEVICE_TYPE deviceType) {
    NV_ENC_OPEN_ENCODE_SESSION_EX_PARAMS openSessionExParams;
    INIT_CONFIG(openSessionExParams, NV_ENC_OPEN_ENCODE_SESSION_EX_PARAMS);

    openSessionExParams.device = device;
    openSessionExParams.deviceType = deviceType;
    openSessionExParams.reserved = NULL;
    openSessionExParams.apiVersion = NVENCAPI_VERSION;
    
    NVENCSTATUS nvStatus = NV_ENC_SUCCESS;
    if (NV_ENC_SUCCESS != (nvStatus = m_pEncodeAPI->nvEncOpenEncodeSessionEx(&openSessionExParams, &m_hEncoder))) {
        NVPrintFuncError(_T("nvEncOpenEncodeSessionEx"), nvStatus);
        if (nvStatus == NV_ENC_ERR_OUT_OF_MEMORY) {
            NVPrintf(stderr, NV_LOG_ERROR, 
                FOR_AUO ? _T("このエラーはメモリが不足しているか、同時にNVEncで3ストリーム以上エンコードしようとすると発生することがあります。")
                        : _T("This error might occur when shortage of memory, or when trying to encode more than 2 streams by NVEnc."));
        }
        return nvStatus;
    }

    return nvStatus;
}

NVENCSTATUS NVEncCore::SetEncodeCodecList(void *m_hEncoder) {
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

NVENCSTATUS NVEncCore::setCodecProfileList(void *m_hEncoder, NVEncCodecFeature& codecFeature) {
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

NVENCSTATUS NVEncCore::setCodecPresetList(void *m_hEncoder, NVEncCodecFeature& codecFeature, bool getPresetConfig) {
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

NVENCSTATUS NVEncCore::setInputFormatList(void *m_hEncoder, NVEncCodecFeature& codecFeature) {
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

NVENCSTATUS NVEncCore::GetCurrentDeviceNVEncCapability(void *m_hEncoder, NVEncCodecFeature& codecFeature) {
    NVENCSTATUS nvStatus = NV_ENC_SUCCESS;
    bool check_h264 = get_value_from_guid(codecFeature.codec, list_nvenc_codecs) == NV_ENC_H264;
    auto add_cap_info = [&](NV_ENC_CAPS cap_id, bool for_h264_only, const TCHAR *cap_name) {
        if (!(!check_h264 && for_h264_only)) {
            NV_ENC_CAPS_PARAM param;
            INIT_CONFIG(param, NV_ENC_CAPS_PARAM);
            param.capsToQuery = cap_id;
            int value = 0;
            NVENCSTATUS result = m_pEncodeAPI->nvEncGetEncodeCaps(m_hEncoder, codecFeature.codec, &param, &value);
            if (NV_ENC_SUCCESS == result) {
                NVEncCap cap ={ 0 };
                cap.id = cap_id;
                cap.name = cap_name;
                cap.value = value;
                codecFeature.caps.push_back(cap);
            } else {
                nvStatus = result;
            }
        }
    };

    add_cap_info(NV_ENC_CAPS_NUM_MAX_BFRAMES,              false, _T("Max Bframes"));
    add_cap_info(NV_ENC_CAPS_SUPPORTED_RATECONTROL_MODES,  false, _T("RC Modes"));
    add_cap_info(NV_ENC_CAPS_SUPPORT_FIELD_ENCODING,       false, _T("Field Encoding"));
    add_cap_info(NV_ENC_CAPS_SUPPORT_MONOCHROME,           false, _T("MonoChrome"));
    add_cap_info(NV_ENC_CAPS_SUPPORT_FMO,                  true,  _T("FMO"));
    add_cap_info(NV_ENC_CAPS_SUPPORT_QPELMV,               false, _T("Quater-Pel MV"));
    add_cap_info(NV_ENC_CAPS_SUPPORT_BDIRECT_MODE,         false, _T("B Direct Mode"));
    add_cap_info(NV_ENC_CAPS_SUPPORT_CABAC,                true,  _T("CABAC"));
    add_cap_info(NV_ENC_CAPS_SUPPORT_ADAPTIVE_TRANSFORM,   true,  _T("Adaptive Transform"));
    add_cap_info(NV_ENC_CAPS_NUM_MAX_TEMPORAL_LAYERS,      false, _T("Max Temporal Layers"));
    add_cap_info(NV_ENC_CAPS_SUPPORT_HIERARCHICAL_PFRAMES, false, _T("Hierarchial P Frames"));
    add_cap_info(NV_ENC_CAPS_SUPPORT_HIERARCHICAL_BFRAMES, false, _T("Hierarchial B Frames"));
    add_cap_info(NV_ENC_CAPS_LEVEL_MAX,                    false, _T("Max Level"));
    add_cap_info(NV_ENC_CAPS_LEVEL_MIN,                    false, _T("Min Level"));
    add_cap_info(NV_ENC_CAPS_SEPARATE_COLOUR_PLANE,        false, _T("4:4:4"));
    add_cap_info(NV_ENC_CAPS_WIDTH_MAX,                    false, _T("Max Width"));
    add_cap_info(NV_ENC_CAPS_HEIGHT_MAX,                   false, _T("Max Height"));
    add_cap_info(NV_ENC_CAPS_SUPPORT_DYN_RES_CHANGE,       false, _T("Dynamic Resolution Change"));
    add_cap_info(NV_ENC_CAPS_SUPPORT_DYN_BITRATE_CHANGE,   false, _T("Dynamic Bitrate Change"));
    add_cap_info(NV_ENC_CAPS_SUPPORT_DYN_FORCE_CONSTQP,    false, _T("Forced constant QP"));
    add_cap_info(NV_ENC_CAPS_SUPPORT_DYN_RCMODE_CHANGE,    false, _T("Dynamic RC Mode Change"));
    add_cap_info(NV_ENC_CAPS_SUPPORT_SUBFRAME_READBACK,    false, _T("Subframe Readback"));
    add_cap_info(NV_ENC_CAPS_SUPPORT_CONSTRAINED_ENCODING, false, _T("Constrained Encoding"));
    add_cap_info(NV_ENC_CAPS_SUPPORT_INTRA_REFRESH,        false, _T("Intra Refresh"));
    add_cap_info(NV_ENC_CAPS_SUPPORT_CUSTOM_VBV_BUF_SIZE,  false, _T("Custom VBV Bufsize"));
    add_cap_info(NV_ENC_CAPS_SUPPORT_DYNAMIC_SLICE_MODE,   false, _T("Dynamic Slice Mode"));
    add_cap_info(NV_ENC_CAPS_SUPPORT_REF_PIC_INVALIDATION, false, _T("Ref Pic Invalidiation"));
    add_cap_info(NV_ENC_CAPS_PREPROC_SUPPORT,              false, _T("PreProcess"));
    add_cap_info(NV_ENC_CAPS_ASYNC_ENCODE_SUPPORT,         false, _T("Async Encoding"));
    add_cap_info(NV_ENC_CAPS_MB_NUM_MAX,                   false, _T("Max MBs"));
    add_cap_info(NV_ENC_CAPS_MB_PER_SEC_MAX,               false, _T("MAX MB per sec"));
    add_cap_info(NV_ENC_CAPS_SUPPORT_LOSSLESS_ENCODE,      false, _T("Lossless"));
    return nvStatus;
}

NVENCSTATUS NVEncCore::createDeviceCodecList() {
    return SetEncodeCodecList(m_hEncoder);
}

NVENCSTATUS NVEncCore::createDeviceFeatureList(bool getPresetConfig) {
    NVENCSTATUS nvStatus = NV_ENC_SUCCESS;
    //m_EncodeFeaturesが作成されていなければ、自動的に作成
    if (m_EncodeFeatures.size() == 0)
        nvStatus = SetEncodeCodecList(m_hEncoder);

    if (NV_ENC_SUCCESS == nvStatus) {
        for (uint32_t i = 0; i < m_EncodeFeatures.size(); i++) {
            setCodecProfileList(m_hEncoder, m_EncodeFeatures[i]);
            setCodecPresetList(m_hEncoder, m_EncodeFeatures[i], getPresetConfig);
            setInputFormatList(m_hEncoder, m_EncodeFeatures[i]);
            GetCurrentDeviceNVEncCapability(m_hEncoder, m_EncodeFeatures[i]);
        }
    }
    return nvStatus;
}

const std::vector<NVEncCodecFeature>& NVEncCore::GetNVEncCapability() {
    if (m_EncodeFeatures.size() == 0) {
        createDeviceFeatureList();
    }
    return m_EncodeFeatures;
}

const NVEncCodecFeature *NVEncCore::getCodecFeature(const GUID& codec) {
    for (uint32_t i = 0; i < m_EncodeFeatures.size(); i++) {
        if (0 == memcmp(&m_EncodeFeatures[i].codec, &codec, sizeof(m_stCodecGUID))) {
            return &m_EncodeFeatures[i];
        }
    }
    return nullptr;
}

int NVEncCore::getCapLimit(NV_ENC_CAPS flag, const NVEncCodecFeature *codecFeature) {
    if (nullptr == codecFeature) {
        if (nullptr == (codecFeature = getCodecFeature(m_stCodecGUID))) {
            return 0;
        }
    }
    return get_value(flag, codecFeature->caps);
}

bool NVEncCore::checkProfileSupported(GUID profile, const NVEncCodecFeature *codecFeature) {
    if (nullptr == codecFeature) {
        if (nullptr == (codecFeature = getCodecFeature(m_stCodecGUID))) {
            return false;
        }
    }
    for (auto codecProf : codecFeature->profiles) {
        if (0 == memcmp(&profile, &codecProf, sizeof(codecProf))) {
            return true;
        }
    }
    return false;
}

bool NVEncCore::checkPresetSupported(GUID preset, const NVEncCodecFeature *codecFeature) {
    if (nullptr == codecFeature) {
        if (nullptr == (codecFeature = getCodecFeature(m_stCodecGUID))) {
            return false;
        }
    }
    for (auto codecPreset : codecFeature->presets) {
        if (0 == memcmp(&preset, &codecPreset, sizeof(codecPreset))) {
            return true;
        }
    }
    return false;
}

bool NVEncCore::checkSurfaceFmtSupported(NV_ENC_BUFFER_FORMAT surfaceFormat, const NVEncCodecFeature *codecFeature) {
    if (nullptr == codecFeature) {
        if (nullptr == (codecFeature = getCodecFeature(m_stCodecGUID))) {
            return false;
        }
    }
    for (auto codecFmt : codecFeature->surfaceFmt) {
        if (0 == memcmp(&surfaceFormat, &codecFmt, sizeof(surfaceFormat))) {
            return true;
        }
    }
    return false;
}

NVENCSTATUS NVEncCore::SetInputParam(const InEncodeVideoParam *inputParam) {
    memcpy(&m_stEncConfig, &inputParam->encConfig, sizeof(m_stEncConfig));
    memcpy(&m_stPicStruct, &inputParam->picStruct, sizeof(m_stPicStruct));
    
    //コーデックの決定とチェック
    m_stCodecGUID = inputParam->codec == NV_ENC_H264 ? NV_ENC_CODEC_H264_GUID : NV_ENC_CODEC_HEVC_GUID;
    if (nullptr == getCodecFeature(m_stCodecGUID)) {
        NVPrintf(stderr, NV_LOG_ERROR, FOR_AUO ? _T("指定されたコーデックはサポートされていません。\n") : _T("Selected codec is not supported.\n"));
        return NV_ENC_ERR_UNSUPPORTED_PARAM;
    }

    //プロファイルのチェック
    if (inputParam->codec == NV_ENC_HEVC) {
        //m_stEncConfig.profileGUIDはデフォルトではH.264のプロファイル情報
        //HEVCのプロファイル情報は、m_stEncConfig.encodeCodecConfig.hevcConfig.tierに保存されている
        m_stEncConfig.profileGUID = get_guid_from_value(m_stEncConfig.encodeCodecConfig.hevcConfig.tier, h265_profile_names);
    }
    if (!checkProfileSupported(m_stEncConfig.profileGUID)) {
        NVPrintf(stderr, NV_LOG_ERROR, FOR_AUO ? _T("指定されたプロファイルはサポートされていません。\n") : _T("Selected profile is not supported.\n"));
        return NV_ENC_ERR_UNSUPPORTED_PARAM;
    }

    //プリセットのチェック
    if (!checkPresetSupported(get_guid_from_value(inputParam->preset, preset_names))) {
        NVPrintf(stderr, NV_LOG_ERROR, FOR_AUO ? _T("指定されたプリセットはサポートされていません。\n") : _T("Selected preset is not supported.\n"));
        return NV_ENC_ERR_UNSUPPORTED_PARAM;
    }

    //入力フォーマットはここでは気にしない
    //NV_ENC_BUFFER_FORMAT_NV12_TILED64x16
    //if (!checkSurfaceFmtSupported(NV_ENC_BUFFER_FORMAT_NV12_TILED64x16)) {
    //    NVPrintf(stderr, NV_LOG_ERROR, FOR_AUO ? _T("入力フォーマットが決定できません。\n") : _T("Input format is not supported.\n"));
    //    return NV_ENC_ERR_UNSUPPORTED_PARAM;
    //}

    //バッファサイズ (固定で16として与える)
    m_uEncodeBufferCount = 16; // inputParam->inputBuffer;
    if (m_uEncodeBufferCount > MAX_ENCODE_QUEUE) {
#if FOR_AUO
        NVPrintf(stderr, NV_LOG_ERROR, _T("入力バッファは多すぎます。: %d フレーム\n"), m_uEncodeBufferCount);
        NVPrintf(stderr, NV_LOG_ERROR, _T("%d フレームまでに設定して下さい。\n"), MAX_ENCODE_QUEUE);
#else
        NVPrintf(stderr, NV_LOG_ERROR, _T("Input frame of %d exceeds the maximum size allowed (%d).\n"), m_uEncodeBufferCount, MAX_ENCODE_QUEUE);
#endif
        return NV_ENC_ERR_UNSUPPORTED_PARAM;
    }

    //解像度の決定
    m_uEncWidth   = inputParam->input.width  - inputParam->input.crop[0] - inputParam->input.crop[2];
    m_uEncHeight  = inputParam->input.height - inputParam->input.crop[1] - inputParam->input.crop[3];

    //picStructの設定
    m_stPicStruct = (inputParam->picStruct == 0) ? NV_ENC_PIC_STRUCT_FRAME : inputParam->picStruct;
    
    //制限事項チェック
    if (inputParam->input.width < 0 && inputParam->input.height < 0) {
        NVPrintf(stderr, NV_LOG_ERROR, _T("%s: %dx%d\n"), FOR_AUO ? _T("解像度が無効です。") : _T("Invalid resolution."), inputParam->input.width, inputParam->input.height);
        return NV_ENC_ERR_UNSUPPORTED_PARAM;
    }
    if (   inputParam->input.width  <= inputParam->input.crop[0] + inputParam->input.crop[2]
        && inputParam->input.height <= inputParam->input.crop[1] + inputParam->input.crop[3]) {
        NVPrintf(stderr, NV_LOG_ERROR, _T("%s: %dx%d, Crop [%d,%d,%d,%d]\n"),
             FOR_AUO ? _T("Crop値が無効です。") : _T("Invalid crop value."), 
            inputParam->input.width, inputParam->input.height,
            inputParam->input.crop[0], inputParam->input.crop[1], inputParam->input.crop[2], inputParam->input.crop[3]);
        return NV_ENC_ERR_UNSUPPORTED_PARAM;
    }

    const int height_check_mask = 1 + 2 * !!is_interlaced(inputParam->picStruct);
    if ((m_uEncWidth & 1) || (m_uEncHeight & height_check_mask)) {
        NVPrintf(stderr, NV_LOG_ERROR, _T("%s: %dx%d\n"), FOR_AUO ? _T("解像度が無効です。") : _T("Invalid resolution."), m_uEncWidth, m_uEncHeight);
        NVPrintf(stderr, NV_LOG_ERROR, FOR_AUO ? _T("縦横の解像度は2の倍数である必要があります。\n") : _T("Relosution of mod2 required.\n"));
        if (is_interlaced(inputParam->picStruct)) {
            NVPrintf(stderr, NV_LOG_ERROR, FOR_AUO ? _T("さらに、インタレ保持エンコードでは縦解像度は4の倍数である必要があります。\n") : _T("For interlaced encoding, mod4 is required for height.\n"));
        }
        return NV_ENC_ERR_UNSUPPORTED_PARAM;
    }
    if ((inputParam->input.crop[0] & 1) || (inputParam->input.crop[2] & 1)
        || (inputParam->input.crop[1] & height_check_mask) || (inputParam->input.crop[3] & height_check_mask)) {
        NVPrintf(stderr, NV_LOG_ERROR, _T("%s: %dx%d, Crop [%d,%d,%d,%d]\n"),
             FOR_AUO ? _T("Crop値が無効です。") : _T("Invalid crop value."), 
            inputParam->input.width, inputParam->input.height,
            inputParam->input.crop[0], inputParam->input.crop[1], inputParam->input.crop[2], inputParam->input.crop[3]);
        NVPrintf(stderr, NV_LOG_ERROR, FOR_AUO ? _T("Crop値は2の倍数である必要があります。\n") : _T("Crop value of mod2 required.\n"));
        if (is_interlaced(inputParam->picStruct)) {
            NVPrintf(stderr, NV_LOG_ERROR, FOR_AUO ? _T("さらに、インタレ保持エンコードでは縦Crop値は4の倍数である必要があります。\n") : _T("For interlaced encoding, mod4 is required for height.\n"));
        }
        return NV_ENC_ERR_UNSUPPORTED_PARAM;
    }
    //環境による制限
    auto error_resolution_over_limit = [&](const TCHAR *feature, uint32_t featureValue, NV_ENC_CAPS featureID) {
        const TCHAR *error_mes = FOR_AUO ? _T("解像度が上限を超えています。") : _T("Resolution is over limit.");
        if (nullptr == feature)
            NVPrintf(stderr, NV_LOG_ERROR, _T("%s: %dx%d [上限: %dx%d]\n"), error_mes, m_uEncWidth, m_uEncHeight, getCapLimit(NV_ENC_CAPS_WIDTH_MAX), getCapLimit(NV_ENC_CAPS_HEIGHT_MAX));
        else
            NVPrintf(stderr, NV_LOG_ERROR, _T("%s: %dx%d %s: %d [上限: %d]\n"), m_uEncWidth, m_uEncHeight, feature, featureValue, getCapLimit(featureID));
    };

    if (m_uEncWidth > (uint32_t)getCapLimit(NV_ENC_CAPS_WIDTH_MAX) || m_uEncHeight > (uint32_t)getCapLimit(NV_ENC_CAPS_HEIGHT_MAX)) {
        error_resolution_over_limit(nullptr, 0, (NV_ENC_CAPS)0);
        return NV_ENC_ERR_UNSUPPORTED_PARAM;
    }
    uint32_t heightMod = 16 * (1 + !!is_interlaced(inputParam->picStruct));
    uint32_t targetMB = ((m_uEncWidth + 15) / 16) * ((m_uEncHeight + (heightMod - 1)) / heightMod);
    if (targetMB > (uint32_t)getCapLimit(NV_ENC_CAPS_MB_NUM_MAX)) {
        error_resolution_over_limit(_T("MB"), targetMB, NV_ENC_CAPS_MB_NUM_MAX);
        return NV_ENC_ERR_UNSUPPORTED_PARAM;
    }
    uint32_t targetMBperSec = (targetMB * inputParam->input.rate + inputParam->input.scale - 1) / inputParam->input.scale;
    if (targetMBperSec > (uint32_t)getCapLimit(NV_ENC_CAPS_MB_PER_SEC_MAX)) {
        error_resolution_over_limit(_T("MB/s"), targetMBperSec, NV_ENC_CAPS_MB_PER_SEC_MAX);
        return NV_ENC_ERR_UNSUPPORTED_PARAM;
    }

    auto error_feature_unsupported = [&](int log_level, const TCHAR *feature_name) {
        NVPrintf(stderr, log_level, FOR_AUO ? _T("%sはサポートされていません。\n") : _T("%s unsupported.\n"), feature_name);
    };

    if (is_interlaced(inputParam->picStruct) && !getCapLimit(NV_ENC_CAPS_SUPPORT_FIELD_ENCODING)) {
        return NV_ENC_ERR_UNSUPPORTED_PARAM;
    }
    if (m_stEncConfig.rcParams.rateControlMode != (m_stEncConfig.rcParams.rateControlMode & getCapLimit(NV_ENC_CAPS_SUPPORTED_RATECONTROL_MODES))) {
        error_feature_unsupported(NV_LOG_ERROR, FOR_AUO ? _T("選択されたレート制御モード") : _T("Selected encode mode"));
        return NV_ENC_ERR_UNSUPPORTED_PARAM;
    }
    if (m_stEncConfig.frameIntervalP < 0) {
        NVPrintf(stderr, NV_LOG_ERROR, _T("%s: %d\n"),
            FOR_AUO ? _T("Bフレーム設定が無効です。正の値を使用してください。") : _T("B frame settings are invalid. Please use a number > 0."),
            m_stEncConfig.frameIntervalP - 1);
        return NV_ENC_ERR_UNSUPPORTED_PARAM;
    }
    if (inputParam->bluray) {
        if (inputParam->codec == NV_ENC_HEVC) {
            NVPrintf(stderr, NV_LOG_ERROR, FOR_AUO ? _T("HEVCではBluray用出力はサポートされていません。") : _T("Bluray output is not supported for HEVC codec."));
            return NV_ENC_ERR_UNSUPPORTED_PARAM;
        }
        if (inputParam->encConfig.rcParams.rateControlMode != NV_ENC_PARAMS_RC_VBR) {
            NVPrintf(stderr, NV_LOG_ERROR, FOR_AUO ? _T("Bluray用出力では、VBRモードを使用してください。") :  _T("Please use VBR mode for bluray output."));
            return NV_ENC_ERR_UNSUPPORTED_PARAM;
        }
        if (!getCapLimit(NV_ENC_CAPS_SUPPORT_CUSTOM_VBV_BUF_SIZE)) {
            error_feature_unsupported(NV_LOG_ERROR, FOR_AUO ? _T("VBVバッファサイズの指定") : _T("Custom VBV Bufsize"));
            NVPrintf(stderr, NV_LOG_ERROR, FOR_AUO ? _T("Bluray用出力を行えません。") :  _T("Therfore you cannot output for bluray."));
            return NV_ENC_ERR_UNSUPPORTED_PARAM;
        }
    }
    if (m_stEncConfig.frameIntervalP - 1 > getCapLimit(NV_ENC_CAPS_NUM_MAX_BFRAMES)) {
        m_stEncConfig.frameIntervalP = getCapLimit(NV_ENC_CAPS_NUM_MAX_BFRAMES) + 1;
        NVPrintf(stderr, NV_LOG_WARN, FOR_AUO ? _T("Bフレームの最大数は%dです。\n") : _T("Max B frames are %d frames.\n"), getCapLimit(NV_ENC_CAPS_NUM_MAX_BFRAMES));
    }
    if (inputParam->codec == NV_ENC_H264) {
        if (NV_ENC_H264_ENTROPY_CODING_MODE_CABAC == m_stEncConfig.encodeCodecConfig.h264Config.entropyCodingMode && !getCapLimit(NV_ENC_CAPS_SUPPORT_CABAC)) {
            m_stEncConfig.encodeCodecConfig.h264Config.entropyCodingMode = NV_ENC_H264_ENTROPY_CODING_MODE_CAVLC;
            error_feature_unsupported(NV_LOG_WARN, _T("CABAC"));
        }
        if (NV_ENC_H264_FMO_ENABLE == m_stEncConfig.encodeCodecConfig.h264Config.fmoMode && !getCapLimit(NV_ENC_CAPS_SUPPORT_FMO)) {
            m_stEncConfig.encodeCodecConfig.h264Config.fmoMode = NV_ENC_H264_FMO_DISABLE;
            error_feature_unsupported(NV_LOG_WARN, _T("FMO"));
        }
        if (NV_ENC_H264_BDIRECT_MODE_TEMPORAL & m_stEncConfig.encodeCodecConfig.h264Config.bdirectMode && !getCapLimit(NV_ENC_CAPS_SUPPORT_BDIRECT_MODE)) {
            m_stEncConfig.encodeCodecConfig.h264Config.bdirectMode = NV_ENC_H264_BDIRECT_MODE_DISABLE;
            error_feature_unsupported(NV_LOG_WARN, _T("B Direct mode"));
        }
        if (NV_ENC_H264_ADAPTIVE_TRANSFORM_ENABLE != m_stEncConfig.encodeCodecConfig.h264Config.adaptiveTransformMode && !getCapLimit(NV_ENC_CAPS_SUPPORT_ADAPTIVE_TRANSFORM)) {
            m_stEncConfig.encodeCodecConfig.h264Config.adaptiveTransformMode = NV_ENC_H264_ADAPTIVE_TRANSFORM_DISABLE;
            error_feature_unsupported(NV_LOG_WARN, _T("Adaptive Tranform"));
        }
    }
    if ((NV_ENC_MV_PRECISION_QUARTER_PEL == m_stEncConfig.mvPrecision) && !getCapLimit(NV_ENC_CAPS_SUPPORT_QPELMV)) {
        m_stEncConfig.mvPrecision = NV_ENC_MV_PRECISION_HALF_PEL;
        error_feature_unsupported(NV_LOG_WARN, FOR_AUO ? _T("1/4画素精度MV探索") : _T("Q-Pel MV"));
    }
    if (0 != m_stEncConfig.rcParams.vbvBufferSize && !getCapLimit(NV_ENC_CAPS_SUPPORT_CUSTOM_VBV_BUF_SIZE)) {
        m_stEncConfig.rcParams.vbvBufferSize = 0;
        error_feature_unsupported(NV_LOG_WARN, FOR_AUO ? _T("VBVバッファサイズの指定") : _T("Custom VBV Bufsize"));
    }
    //自動決定パラメータ
    if (0 == m_stEncConfig.gopLength) {
        m_stEncConfig.gopLength = (int)(inputParam->input.rate / (double)inputParam->input.scale + 0.5) * 10;
    }
    //SAR自動設定
    std::pair<int, int> par = std::make_pair(inputParam->par[0], inputParam->par[1]);
    adjust_sar(&par.first, &par.second, m_uEncWidth, m_uEncHeight);

    //色空間設定自動
    int frame_height = m_uEncHeight;
    auto apply_auto_colormatrix = [frame_height](uint32_t& value, const CX_DESC *list) {
        if (COLOR_VALUE_AUTO == value)
            value = (frame_height >= HD_HEIGHT_THRESHOLD) ? list[HD_INDEX].value : list[SD_INDEX].value;
    };

    apply_auto_colormatrix(m_stEncConfig.encodeCodecConfig.h264Config.h264VUIParameters.colourPrimaries,         list_colorprim);
    apply_auto_colormatrix(m_stEncConfig.encodeCodecConfig.h264Config.h264VUIParameters.transferCharacteristics, list_transfer);
    apply_auto_colormatrix(m_stEncConfig.encodeCodecConfig.h264Config.h264VUIParameters.colourMatrix,            list_colormatrix);

    INIT_CONFIG(m_stCreateEncodeParams, NV_ENC_INITIALIZE_PARAMS);
    m_stCreateEncodeParams.encodeConfig        = &m_stEncConfig;
    m_stCreateEncodeParams.encodeHeight        = m_uEncHeight;
    m_stCreateEncodeParams.encodeWidth         = m_uEncWidth;
    m_stCreateEncodeParams.darHeight           = m_uEncHeight;
    m_stCreateEncodeParams.darWidth            = m_uEncWidth;
    get_dar_pixels(&m_stCreateEncodeParams.darWidth, &m_stCreateEncodeParams.darHeight, par.first, par.second);

    m_stCreateEncodeParams.maxEncodeHeight     = m_uEncHeight;
    m_stCreateEncodeParams.maxEncodeWidth      = m_uEncWidth;

    m_stCreateEncodeParams.frameRateNum        = inputParam->input.rate;
    m_stCreateEncodeParams.frameRateDen        = inputParam->input.scale;
    //Fix me add theading model
    m_stCreateEncodeParams.enableEncodeAsync   = true;
    m_stCreateEncodeParams.enablePTD           = true;
    m_stCreateEncodeParams.encodeGUID          = m_stCodecGUID;
    m_stCreateEncodeParams.presetGUID          = preset_names[inputParam->preset].id;

    if (inputParam->codec == NV_ENC_HEVC) {
        //整合性チェック (一般, H.265/HEVC)
        m_stCreateEncodeParams.encodeConfig->encodeCodecConfig.hevcConfig.idrPeriod = m_stCreateEncodeParams.encodeConfig->gopLength;
    } else if (inputParam->codec == NV_ENC_H264) {
        //Bluray 互換出力
        if (inputParam->bluray) {
            m_stCreateEncodeParams.encodeConfig->encodeCodecConfig.h264Config.outputPictureTimingSEI = 1;
            m_stCreateEncodeParams.encodeConfig->encodeCodecConfig.h264Config.outputRecoveryPointSEI = 1;
            m_stCreateEncodeParams.encodeConfig->encodeCodecConfig.h264Config.outputAUD = 1;
            m_stCreateEncodeParams.encodeConfig->encodeCodecConfig.h264Config.sliceMode = 3;
            m_stCreateEncodeParams.encodeConfig->encodeCodecConfig.h264Config.sliceModeData = 4;
            m_stCreateEncodeParams.encodeConfig->encodeCodecConfig.h264Config.level = min(m_stCreateEncodeParams.encodeConfig->encodeCodecConfig.h264Config.level, NV_ENC_LEVEL_H264_41);
            m_stCreateEncodeParams.encodeConfig->rcParams.maxBitRate = min(m_stCreateEncodeParams.encodeConfig->rcParams.maxBitRate, 40000 * 1000);
            if (m_stCreateEncodeParams.encodeConfig->rcParams.vbvBufferSize == 0) {
                m_stCreateEncodeParams.encodeConfig->rcParams.vbvBufferSize = m_stCreateEncodeParams.encodeConfig->rcParams.maxBitRate;
            }
            m_stCreateEncodeParams.encodeConfig->rcParams.vbvInitialDelay = m_stCreateEncodeParams.encodeConfig->rcParams.vbvBufferSize / 2;
            m_stCreateEncodeParams.encodeConfig->rcParams.averageBitRate = min(m_stCreateEncodeParams.encodeConfig->rcParams.averageBitRate, m_stCreateEncodeParams.encodeConfig->rcParams.maxBitRate);
            m_stCreateEncodeParams.encodeConfig->frameIntervalP = min(m_stCreateEncodeParams.encodeConfig->frameIntervalP, 3+1);
            m_stCreateEncodeParams.encodeConfig->gopLength = (min(m_stCreateEncodeParams.encodeConfig->gopLength, 30) / m_stCreateEncodeParams.encodeConfig->frameIntervalP) * m_stCreateEncodeParams.encodeConfig->frameIntervalP;
        }

        //整合性チェック (一般, H.264/AVC)
        m_stCreateEncodeParams.encodeConfig->frameFieldMode = (inputParam->picStruct == NV_ENC_PIC_STRUCT_FRAME) ? NV_ENC_PARAMS_FRAME_FIELD_MODE_FRAME : NV_ENC_PARAMS_FRAME_FIELD_MODE_FIELD;
        //m_stCreateEncodeParams.encodeConfig->encodeCodecConfig.h264Config.entropyCodingMode = (m_stEncoderInput[0].profile > 66) ? NV_ENC_H264_ENTROPY_CODING_MODE_CABAC : NV_ENC_H264_ENTROPY_CODING_MODE_CAVLC;
        m_stCreateEncodeParams.encodeConfig->encodeCodecConfig.h264Config.idrPeriod = m_stCreateEncodeParams.encodeConfig->gopLength;
        if (m_stCreateEncodeParams.encodeConfig->frameIntervalP - 1 <= 0) {
            m_stCreateEncodeParams.encodeConfig->encodeCodecConfig.h264Config.bdirectMode = NV_ENC_H264_BDIRECT_MODE_DISABLE;
        }

        //整合性チェック (H.264 VUI)
        m_stCreateEncodeParams.encodeConfig->encodeCodecConfig.h264Config.h264VUIParameters.overscanInfoPresentFlag =
            (m_stCreateEncodeParams.encodeConfig->encodeCodecConfig.h264Config.h264VUIParameters.overscanInfo) ? 1 : 0;

        m_stCreateEncodeParams.encodeConfig->encodeCodecConfig.h264Config.h264VUIParameters.videoSignalTypePresentFlag =
            (get_cx_value(list_videoformat, _T("undef")) != (int)m_stCreateEncodeParams.encodeConfig->encodeCodecConfig.h264Config.h264VUIParameters.videoFormat
            || m_stCreateEncodeParams.encodeConfig->encodeCodecConfig.h264Config.h264VUIParameters.videoFullRangeFlag) ? 1 : 0;

        m_stCreateEncodeParams.encodeConfig->encodeCodecConfig.h264Config.h264VUIParameters.colourDescriptionPresentFlag =
            (  get_cx_value(list_colorprim,   _T("undef")) != (int)m_stCreateEncodeParams.encodeConfig->encodeCodecConfig.h264Config.h264VUIParameters.colourPrimaries
            || get_cx_value(list_transfer,    _T("undef")) != (int)m_stCreateEncodeParams.encodeConfig->encodeCodecConfig.h264Config.h264VUIParameters.transferCharacteristics
            || get_cx_value(list_colormatrix, _T("undef")) != (int)m_stCreateEncodeParams.encodeConfig->encodeCodecConfig.h264Config.h264VUIParameters.colourMatrix) ? 1 : 0;
    }

    return NV_ENC_SUCCESS;
}

NVENCSTATUS NVEncCore::CreateEncoder(const InEncodeVideoParam *inputParam) {
    NVENCSTATUS nvStatus;

    if (NV_ENC_SUCCESS != (nvStatus = SetInputParam(inputParam)))
        return nvStatus;

    if (NV_ENC_SUCCESS != (nvStatus = m_pEncodeAPI->nvEncInitializeEncoder(m_hEncoder, &m_stCreateEncodeParams))) {
        NVPrintf(stderr, NV_LOG_ERROR,
            _T("%s: %d (%s)\n"), FOR_AUO ? _T("エンコーダの初期化に失敗しました。\n") : _T("Failed to Initialize the encoder\n."),
            nvStatus, to_tchar(_nvencGetErrorEnum(nvStatus)).c_str());
        return nvStatus;
    }

    return nvStatus;
}

NVENCSTATUS NVEncCore::InitEncode(InEncodeVideoParam *inputParam) {
    NVENCSTATUS nvStatus = NV_ENC_SUCCESS;

    //入力ファイルを開き、入力情報も取得
    if (NV_ENC_SUCCESS != (nvStatus = InitInput(inputParam))) {
        NVPrintf(stderr, NV_LOG_ERROR, FOR_AUO ? _T("入力ファイルを開けませんでした。\n") : _T("Failed to open input file.\n"));
        NVPrintf(stderr, NV_LOG_ERROR, m_pInput->getInputMes());
        return nvStatus;
    }

    //出力ファイルを開く
    if (_tfopen_s(&m_fOutput, inputParam->outputFilename.c_str(), _T("wb")) || NULL == m_fOutput) {
        NVPrintf(stderr, NV_LOG_ERROR, FOR_AUO ? _T("出力ファイルのオープンに失敗しました。\n") : _T("Failed to open output file.\n"));
        return NV_ENC_ERR_GENERIC;
    }

    //出力バッファ
    if (nullptr != (m_pOutputBuf = (char *)malloc(OUTPUT_BUF_SIZE))) {
        setvbuf(m_fOutput, m_pOutputBuf, _IOFBF, OUTPUT_BUF_SIZE);
    }
    
    //作成したデバイスの情報をfeature取得
    if (NV_ENC_SUCCESS != (nvStatus = createDeviceFeatureList(false))) {
        return nvStatus;
    }

    //エンコーダにパラメータを渡し、初期化
    if (NV_ENC_SUCCESS != (nvStatus = CreateEncoder(inputParam))) {
        return nvStatus;
    }
    
    //入出力用メモリ確保
    if (NV_ENC_SUCCESS != (nvStatus = AllocateIOBuffers(m_uEncWidth, m_uEncHeight))) {
        return nvStatus;
    }
    return nvStatus;
}

NVENCSTATUS NVEncCore::Initialize(InEncodeVideoParam *inputParam) {
    NVENCSTATUS nvStatus = NV_ENC_SUCCESS;

    if (NULL == m_hinstLib) {
        if (NULL == (m_hinstLib = LoadLibrary(NVENCODE_API_DLL))) {
#if FOR_AUO
            NVPrintf(stderr, NV_LOG_ERROR, _T("%sがシステムに存在しません。\n"), NVENCODE_API_DLL);
            NVPrintf(stderr, NV_LOG_ERROR, _T("NVIDIAのドライバが動作条件を満たしているか確認して下さい。"));
#else
            NVPrintf(stderr, NV_LOG_ERROR, _T("%s does not exists in your system.\n"), NVENCODE_API_DLL);
            NVPrintf(stderr, NV_LOG_ERROR, _T("Please check if the GPU driver is propery installed."));
#endif
            return NV_ENC_ERR_OUT_OF_MEMORY;
        }
    }
    
    MYPROC nvEncodeAPICreateInstance; // function pointer to create instance in nvEncodeAPI
    if (NULL == (nvEncodeAPICreateInstance = (MYPROC)GetProcAddress(m_hinstLib, "NvEncodeAPICreateInstance"))) {
        NVPrintf(stderr, NV_LOG_ERROR, FOR_AUO ? _T("NvEncodeAPICreateInstanceのアドレス取得に失敗しました。\n") : _T("Failed to get address of NvEncodeAPICreateInstance.\n"));
        return NV_ENC_ERR_OUT_OF_MEMORY;
    }

    if (NULL == (m_pEncodeAPI = new NV_ENCODE_API_FUNCTION_LIST)) {
        NVPrintf(stderr, NV_LOG_ERROR, FOR_AUO ? _T("NV_ENCODE_API_FUNCTION_LIST用のメモリ確保に失敗しました。\n") : _T("Failed to allocate memory for NV_ENCODE_API_FUNCTION_LIST.\n"));
        return NV_ENC_ERR_OUT_OF_MEMORY;
    }

    memset(m_pEncodeAPI, 0, sizeof(NV_ENCODE_API_FUNCTION_LIST));
    m_pEncodeAPI->version = NV_ENCODE_API_FUNCTION_LIST_VER;

    if (NV_ENC_SUCCESS != (nvStatus = nvEncodeAPICreateInstance(m_pEncodeAPI))) {
        if (nvStatus == NV_ENC_ERR_INVALID_VERSION) {
#if FOR_AUO
            NVPrintf(stderr, NV_LOG_ERROR, _T("nvEncodeAPIのインスタンス作成に失敗しました。ドライバのバージョンが古い可能性があります。\n"));
            NVPrintf(stderr, NV_LOG_ERROR, _T("最新のドライバに更新して試してみてください。\n"));
#else
            NVPrintf(stderr, NV_LOG_ERROR, _T("Failed to create instance of nvEncodeAPI, please consider updating your GPU driver.\n"));
#endif
        } else {
            NVPrintFuncError(_T("nvEncodeAPICreateInstance"), nvStatus);
        }
        return nvStatus;
    }

    //m_pDeviceを初期化
    if (NV_ENC_SUCCESS != (nvStatus = InitDevice(inputParam))) {
        return nvStatus;
    }

    if (NV_ENC_SUCCESS != (nvStatus = NvEncOpenEncodeSessionEx(m_pDevice, NV_ENC_DEVICE_TYPE_CUDA))) {
        return nvStatus;
    }
    return nvStatus;
}

NVENCSTATUS NVEncCore::EncodeFrame(int encode_idx) {
    EncodeBuffer *pEncodeBuffer = m_EncodeBufferQueue.GetAvailable();
    if (!pEncodeBuffer) {
        ProcessOutput(m_EncodeBufferQueue.GetPending());
        pEncodeBuffer = m_EncodeBufferQueue.GetAvailable();
    }
    
    NV_ENC_PIC_PARAMS encPicParams;
    INIT_CONFIG(encPicParams, NV_ENC_PIC_PARAMS);

    encPicParams.inputBuffer = pEncodeBuffer->stInputBfr.hInputSurface;
    encPicParams.bufferFmt = pEncodeBuffer->stInputBfr.bufferFmt;
    encPicParams.inputWidth = m_uEncWidth;
    encPicParams.inputHeight = m_uEncHeight;
    encPicParams.outputBitstream = pEncodeBuffer->stOutputBfr.hBitstreamBuffer;
    encPicParams.completionEvent = pEncodeBuffer->stOutputBfr.hOutputEvent;
    encPicParams.inputTimeStamp = encode_idx;
    encPicParams.pictureStruct = m_stPicStruct;
    //encPicParams.qpDeltaMap = qpDeltaMapArray;
    //encPicParams.qpDeltaMapSize = qpDeltaMapArraySize;


    //if (encPicCommand)
    //{
    //    if (encPicCommand->bForceIDR)
    //    {
    //        encPicParams.encodePicFlags |= NV_ENC_PIC_FLAG_FORCEIDR;
    //    }

    //    if (encPicCommand->bForceIntraRefresh)
    //    {
    //        if (codecGUID == NV_ENC_CODEC_HEVC_GUID)
    //        {
    //            encPicParams.codecPicParams.hevcPicParams.forceIntraRefreshWithFrameCnt = encPicCommand->intraRefreshDuration;
    //        }
    //        else
    //        {
    //            encPicParams.codecPicParams.h264PicParams.forceIntraRefreshWithFrameCnt = encPicCommand->intraRefreshDuration;
    //        }
    //    }
    //}

    NVENCSTATUS nvStatus = m_pEncodeAPI->nvEncEncodePicture(m_hEncoder, &encPicParams);
    if (nvStatus != NV_ENC_SUCCESS && nvStatus != NV_ENC_ERR_NEED_MORE_INPUT) {
        NVPrintf(stderr, NV_LOG_ERROR, FOR_AUO ? _T("フレームの投入に失敗しました。\n") : _T("Failed to add frame into the encoder.\n"));
        return nvStatus;
    }

    return NV_ENC_SUCCESS;
}

NVENCSTATUS NVEncCore::Encode() {
    NVENCSTATUS nvStatus = NV_ENC_SUCCESS;

    m_pStatus->SetStart();

    int ret = 0;
    const int bufferCount = m_uEncodeBufferCount;
    for (int iFrame = 0; nvStatus == NV_ENC_SUCCESS; iFrame++) {
        uint32_t lockedPitch = 0;
        unsigned char *pInputSurface = nullptr;
        const int index = iFrame % bufferCount;
        NvEncLockInputBuffer(m_stEncodeBuffer[index].stInputBfr.hInputSurface, (void**)&pInputSurface, &lockedPitch);
        ret = m_pInput->LoadNextFrame(pInputSurface, lockedPitch);
        NvEncUnlockInputBuffer(m_stEncodeBuffer[index].stInputBfr.hInputSurface);
        if (ret)
            break;
        nvStatus = EncodeFrame(m_pStatus->m_sData.frameIn);
    }
    if (nvStatus == NV_ENC_SUCCESS) {
        FlushEncoder();
    }
    if (m_fOutput) {
        fclose(m_fOutput);
        m_fOutput = NULL;
    }
    if (m_pInput) {
        m_pInput->Close();
        delete m_pInput;
        m_pInput = nullptr;
    }
    m_pStatus->writeResult();
    return nvStatus;
}

NV_ENC_CODEC_CONFIG NVEncCore::DefaultParamH264() {
    NV_ENC_CODEC_CONFIG config = { 0 };

    config.h264Config.level     = NV_ENC_LEVEL_AUTOSELECT;
    config.h264Config.idrPeriod = DEFAULT_GOP_LENGTH;

    config.h264Config.chromaFormatIDC            = 1;
    config.h264Config.disableDeblockingFilterIDC = 0;
    config.h264Config.disableSPSPPS              = 0;
    config.h264Config.sliceMode                  = 3;
    config.h264Config.sliceModeData              = DEFAULT_NUM_SLICES;
    config.h264Config.maxNumRefFrames            = DEFAULT_REF_FRAMES;
    config.h264Config.bdirectMode                = (DEFAULT_B_FRAMES > 0) ? NV_ENC_H264_BDIRECT_MODE_SPATIAL : NV_ENC_H264_BDIRECT_MODE_DISABLE;
    config.h264Config.adaptiveTransformMode      = NV_ENC_H264_ADAPTIVE_TRANSFORM_ENABLE;
    config.h264Config.entropyCodingMode          = NV_ENC_H264_ENTROPY_CODING_MODE_CABAC;

    config.h264Config.h264VUIParameters.overscanInfo = 0;
    config.h264Config.h264VUIParameters.colourMatrix            = get_cx_value(list_colormatrix, _T("undef"));
    config.h264Config.h264VUIParameters.colourPrimaries         = get_cx_value(list_colorprim,   _T("undef"));
    config.h264Config.h264VUIParameters.transferCharacteristics = get_cx_value(list_transfer,    _T("undef"));
    config.h264Config.h264VUIParameters.videoFormat             = get_cx_value(list_videoformat, _T("undef"));

    return config;
}

NV_ENC_CODEC_CONFIG NVEncCore::DefaultParamHEVC() {
    NV_ENC_CODEC_CONFIG config = { 0 };

    config.hevcConfig.level = NV_ENC_LEVEL_AUTOSELECT;
    config.hevcConfig.tier  = NV_ENC_TIER_HEVC_MAIN;
    config.hevcConfig.minCUSize = NV_ENC_HEVC_CUSIZE_8x8;
    config.hevcConfig.maxCUSize = NV_ENC_HEVC_CUSIZE_32x32;
    config.hevcConfig.sliceMode = 0;
    config.hevcConfig.sliceModeData = 0;
    config.hevcConfig.maxNumRefFramesInDPB = DEFAULT_REF_FRAMES;

    return config;
}

NV_ENC_CONFIG NVEncCore::DefaultParam() {

    NV_ENC_CONFIG config = { 0 };
    SET_VER(config, NV_ENC_CONFIG);
    config.frameFieldMode                 = NV_ENC_PARAMS_FRAME_FIELD_MODE_FRAME;
    config.profileGUID                    = NV_ENC_H264_PROFILE_HIGH_GUID;
    config.gopLength                      = DEFAULT_GOP_LENGTH;
    config.rcParams.rateControlMode       = NV_ENC_PARAMS_RC_CONSTQP;
    //config.encodeCodecConfig.h264Config.level;
    config.frameIntervalP                 = DEFAULT_B_FRAMES + 1;
    config.mvPrecision                    = NV_ENC_MV_PRECISION_QUARTER_PEL;
    config.monoChromeEncoding             = 0;
    config.rcParams.version               = NV_ENC_RC_PARAMS_VER;
    config.rcParams.averageBitRate        = DEFAULT_AVG_BITRATE;
    config.rcParams.maxBitRate            = DEFAULT_MAX_BITRATE;
    config.rcParams.enableInitialRCQP     = 1;
    config.rcParams.initialRCQP.qpInterB  = DEFAULT_QP_B;
    config.rcParams.initialRCQP.qpInterP  = DEFAULT_QP_P;
    config.rcParams.initialRCQP.qpIntra   = DEFAUTL_QP_I;
    config.rcParams.constQP.qpInterB      = DEFAULT_QP_B;
    config.rcParams.constQP.qpInterP      = DEFAULT_QP_P;
    config.rcParams.constQP.qpIntra       = DEFAUTL_QP_I;

    config.rcParams.vbvBufferSize         = 0;
    config.rcParams.vbvInitialDelay       = 0;
    config.encodeCodecConfig              = DefaultParamH264();

    return config;
}

tstring NVEncCore::GetEncodingParamsInfo(int output_level) {
    tstring str;
    auto add_str =[output_level, &str](int info_level, const TCHAR *fmt, ...) {
        if (info_level >= output_level) {
            va_list args;
            va_start(args, fmt);
            const size_t append_len = _vsctprintf(fmt, args) + 1;
            size_t current_len = _tcslen(str.c_str());
            str.resize(current_len + append_len, 0);
            _vstprintf_s(&str[current_len], append_len, fmt, args);
        }
    };

    auto value_or_auto =[](int value, int value_auto, const TCHAR *unit) {
        tstring str;
        if (value == value_auto) {
            str = _T("auto");
        } else {
            TCHAR buf[256];
            _stprintf_s(buf, _countof(buf), _T("%d %s"), value, unit);
            str = buf;
        }
        return str;
    };

    auto on_off =[](int value) {
        return (value) ? _T("on") : _T("off");
    };

    TCHAR cpu_info[1024] = { 0 };
    getCPUInfo(cpu_info, _countof(cpu_info));

    TCHAR gpu_info[1024] = { 0 };
    getGPUInfo("NVIDIA", gpu_info, _countof(gpu_info));

    int codec = get_value_from_guid(m_stCodecGUID, list_nvenc_codecs);
    auto sar = get_sar(m_uEncWidth, m_uEncHeight, m_stCreateEncodeParams.darWidth, m_stCreateEncodeParams.darHeight);
#if FOR_AUO
    add_str(NV_LOG_ERROR, _T("NVEnc %s (%s), using NVENC API v%d.%d\n"), VER_STR_FILEVERSION_TCHAR, BUILD_ARCH_STR, NVENCAPI_MAJOR_VERSION, NVENCAPI_MINOR_VERSION);
    add_str(NV_LOG_INFO,  _T("OS バージョン           %s (%s)\n"), getOSVersion(), nv_is_64bit_os() ? _T("x64") : _T("x86"));
    add_str(NV_LOG_INFO,  _T("CPU                     %s\n"), cpu_info);
    add_str(NV_LOG_INFO,  _T("GPU                     %s\n"), gpu_info);
    add_str(NV_LOG_ERROR, _T("入力フレームバッファ    %s, %d frames\n"), _T("CUDA"), m_uEncodeBufferCount);
    add_str(NV_LOG_ERROR, _T("入力フレーム情報        %s\n"), m_pInput->getInputMes());
    add_str(NV_LOG_ERROR, _T("出力動画情報            %s %s\n"), get_name_from_guid(m_stCodecGUID, list_nvenc_codecs),
        (codec == NV_ENC_H264) ? get_name_from_guid(m_stEncConfig.profileGUID, h264_profile_names) : get_name_from_guid(m_stEncConfig.profileGUID, h265_profile_names));
    add_str(NV_LOG_ERROR, _T("                        %dx%d%s %d:%d %.3ffps (%d/%dfps)\n"), m_uEncWidth, m_uEncHeight, (m_stEncConfig.frameFieldMode != NV_ENC_PARAMS_FRAME_FIELD_MODE_FRAME) ? _T("i") : _T("p"), sar.first, sar.second, m_stCreateEncodeParams.frameRateNum / (double)m_stCreateEncodeParams.frameRateDen, m_stCreateEncodeParams.frameRateNum, m_stCreateEncodeParams.frameRateDen);
    add_str(NV_LOG_DEBUG, _T("Encoder Preset          %s\n"), get_name_from_guid(m_stCreateEncodeParams.presetGUID, preset_names));
    add_str(NV_LOG_ERROR, _T("レート制御モード        %s\n"), get_desc(list_nvenc_rc_method, m_stEncConfig.rcParams.rateControlMode));
    if (NV_ENC_PARAMS_RC_CONSTQP == m_stEncConfig.rcParams.rateControlMode) {
        add_str(NV_LOG_ERROR, _T("CQP値                   I:%d  P:%d  B:%d\n"), m_stEncConfig.rcParams.constQP.qpIntra, m_stEncConfig.rcParams.constQP.qpInterP, m_stEncConfig.rcParams.constQP.qpInterB);
    } else {
        add_str(NV_LOG_ERROR, _T("ビットレート            %d kbps (Max: %d kbps)\n"), m_stEncConfig.rcParams.averageBitRate / 1000, m_stEncConfig.rcParams.maxBitRate / 1000);
        if (m_stEncConfig.rcParams.enableInitialRCQP)
            add_str(NV_LOG_INFO,  _T("初期QP値                I:%d  P:%d  B:%d\n"), m_stEncConfig.rcParams.constQP.qpIntra, m_stEncConfig.rcParams.constQP.qpInterP, m_stEncConfig.rcParams.constQP.qpInterB);
        if (m_stEncConfig.rcParams.enableMaxQP || m_stEncConfig.rcParams.enableMinQP) {
            int minQPI = (m_stEncConfig.rcParams.enableMinQP) ? m_stEncConfig.rcParams.minQP.qpIntra  :  0;
            int maxQPI = (m_stEncConfig.rcParams.enableMaxQP) ? m_stEncConfig.rcParams.maxQP.qpIntra  : 51;
            int minQPP = (m_stEncConfig.rcParams.enableMinQP) ? m_stEncConfig.rcParams.minQP.qpInterP :  0;
            int maxQPP = (m_stEncConfig.rcParams.enableMaxQP) ? m_stEncConfig.rcParams.maxQP.qpInterP : 51;
            int minQPB = (m_stEncConfig.rcParams.enableMinQP) ? m_stEncConfig.rcParams.minQP.qpInterB :  0;
            int maxQPB = (m_stEncConfig.rcParams.enableMaxQP) ? m_stEncConfig.rcParams.maxQP.qpInterB : 51;
            add_str(NV_LOG_INFO,  _T("QP制御範囲              I:%d-%d  P:%d-%d  B:%d-%d\n"), minQPI, maxQPI, minQPP, maxQPP, minQPB, maxQPB);
        }
        add_str(NV_LOG_INFO,  _T("VBVバッファサイズ       %s\n"), value_or_auto(m_stEncConfig.rcParams.vbvBufferSize / 1000,   0, _T("kbit")).c_str());
        add_str(NV_LOG_DEBUG, _T("VBV initial delay       %s\n"), value_or_auto(m_stEncConfig.rcParams.vbvInitialDelay / 1000, 0, _T("kbit")).c_str());
    }
    add_str(NV_LOG_INFO,  _T("GOP長                   %d frames\n"), m_stEncConfig.gopLength);
    add_str(NV_LOG_INFO,  _T("連続Bフレーム数         %d frames\n"), m_stEncConfig.frameIntervalP - 1);
    if (codec == NV_ENC_H264) {
        add_str(NV_LOG_DEBUG, _T("出力                    "));
        TCHAR bitstream_info[256] ={ 0 };
        if (m_stEncConfig.encodeCodecConfig.h264Config.outputBufferingPeriodSEI) _tcscat_s(bitstream_info, _countof(bitstream_info), _T("BufferingPeriodSEI,"));
        if (m_stEncConfig.encodeCodecConfig.h264Config.outputPictureTimingSEI)   _tcscat_s(bitstream_info, _countof(bitstream_info), _T("PicTimingSEI,"));
        if (m_stEncConfig.encodeCodecConfig.h264Config.outputAUD)                _tcscat_s(bitstream_info, _countof(bitstream_info), _T("AUD,"));
        if (m_stEncConfig.encodeCodecConfig.h264Config.outputFramePackingSEI)    _tcscat_s(bitstream_info, _countof(bitstream_info), _T("FramePackingSEI,"));
        if (m_stEncConfig.encodeCodecConfig.h264Config.outputRecoveryPointSEI)   _tcscat_s(bitstream_info, _countof(bitstream_info), _T("RecoveryPointSEI,"));
        if (m_stEncConfig.encodeCodecConfig.h264Config.repeatSPSPPS)             _tcscat_s(bitstream_info, _countof(bitstream_info), _T("repeatSPSPPS,"));
        if (_tcslen(bitstream_info)) {
            bitstream_info[_tcslen(bitstream_info)-1] = _T('\0');
        } else {
            _tcscpy_s(bitstream_info, _countof(bitstream_info), _T("-"));
        }
        add_str(NV_LOG_DEBUG, _T("%s\n"), bitstream_info);
    }

    add_str(NV_LOG_INFO,  _T("参照距離                %d frames\n"), (codec == NV_ENC_H264) ? m_stEncConfig.encodeCodecConfig.h264Config.maxNumRefFrames : m_stEncConfig.encodeCodecConfig.hevcConfig.maxNumRefFramesInDPB);
    add_str(NV_LOG_INFO,  _T("動きベクトル精度        %s\n"), get_desc(list_mv_presicion_ja, m_stEncConfig.mvPrecision));
    if (codec == NV_ENC_H264 && 3 == m_stEncConfig.encodeCodecConfig.h264Config.sliceMode) {
        add_str(NV_LOG_DEBUG, _T("スライス数              %d\n"), m_stEncConfig.encodeCodecConfig.h264Config.sliceModeData);
    } else {
        add_str(NV_LOG_DEBUG, _T("スライス                Mode:%d, ModeData:%d\n"), m_stEncConfig.encodeCodecConfig.h264Config.sliceMode, m_stEncConfig.encodeCodecConfig.h264Config.sliceModeData);
    }
    if (codec == NV_ENC_H264) {
        add_str(NV_LOG_INFO,  _T("CABAC/deblock           %s / %s\n"), get_desc(list_entropy_coding, m_stEncConfig.encodeCodecConfig.h264Config.entropyCodingMode), on_off(!m_stEncConfig.encodeCodecConfig.h264Config.disableDeblockingFilterIDC));
        add_str(NV_LOG_DEBUG, _T("hierarchical Frames     P:%s  B:%s\n"), on_off(m_stEncConfig.encodeCodecConfig.h264Config.hierarchicalPFrames), on_off(m_stEncConfig.encodeCodecConfig.h264Config.hierarchicalBFrames));
        add_str(NV_LOG_DEBUG, _T("可変フレームレート      %s\n"), on_off(m_stEncConfig.encodeCodecConfig.h264Config.enableVFR));
        add_str(NV_LOG_DEBUG, _T("LTR                     %s"),   on_off(m_stEncConfig.encodeCodecConfig.h264Config.enableLTR));
        if (m_stEncConfig.encodeCodecConfig.h264Config.enableLTR) {
            add_str(NV_LOG_DEBUG, _T(", Mode:%d, NumFrames:%d"), m_stEncConfig.encodeCodecConfig.h264Config.ltrTrustMode, m_stEncConfig.encodeCodecConfig.h264Config.ltrNumFrames);
        }
        add_str(NV_LOG_DEBUG, _T("\n"));
        add_str(NV_LOG_DEBUG, _T("YUV 4:4:4               %s\n"), on_off(m_stEncConfig.encodeCodecConfig.h264Config.separateColourPlaneFlag));
        add_str(NV_LOG_DEBUG, _T("Adaptive Transform      %s\n"), get_desc(list_adapt_transform, m_stEncConfig.encodeCodecConfig.h264Config.adaptiveTransformMode));
        add_str(NV_LOG_DEBUG, _T("FMO                     %s\n"), get_desc(list_fmo, m_stEncConfig.encodeCodecConfig.h264Config.fmoMode));
        add_str(NV_LOG_DEBUG, _T("動き予測                %s\n"), get_desc(list_bdirect, m_stEncConfig.encodeCodecConfig.h264Config.bdirectMode));
    }
#else
    add_str(NV_LOG_ERROR, _T("NVEnc %s (%s), using NVENC API v%d.%d\n"), VER_STR_FILEVERSION_TCHAR, BUILD_ARCH_STR, NVENCAPI_MAJOR_VERSION, NVENCAPI_MINOR_VERSION);
    add_str(NV_LOG_INFO,  _T("OS Version              %s (%s)\n"), getOSVersion(), nv_is_64bit_os() ? _T("x64") : _T("x86"));
    add_str(NV_LOG_INFO,  _T("CPU                     %s\n"), cpu_info);
    add_str(NV_LOG_INFO,  _T("GPU                     %s\n"), gpu_info);
    add_str(NV_LOG_ERROR, _T("Input Buffers           %s, %d frames\n"), _T("CUDA"), m_uEncodeBufferCount);
    add_str(NV_LOG_ERROR, _T("Input Info              %s\n"), m_pInput->getInputMes());
    add_str(NV_LOG_ERROR, _T("Output Info             %s %s\n"), get_name_from_guid(m_stCodecGUID, list_nvenc_codecs),
        (codec == NV_ENC_H264) ? get_name_from_guid(m_stEncConfig.profileGUID, h264_profile_names) : get_name_from_guid(m_stEncConfig.profileGUID, h265_profile_names));
    add_str(NV_LOG_ERROR, _T("                        %dx%d%s %d:%d %.3ffps (%d/%dfps)\n"), m_uEncWidth, m_uEncHeight, (m_stEncConfig.frameFieldMode != NV_ENC_PARAMS_FRAME_FIELD_MODE_FRAME) ? _T("i") : _T("p"), sar.first, sar.second, m_stCreateEncodeParams.frameRateNum / (double)m_stCreateEncodeParams.frameRateDen, m_stCreateEncodeParams.frameRateNum, m_stCreateEncodeParams.frameRateDen);
    add_str(NV_LOG_DEBUG, _T("Encoder Preset          %s\n"), get_name_from_guid(m_stCreateEncodeParams.presetGUID, preset_names));
    add_str(NV_LOG_ERROR, _T("Rate Control            %s\n"), get_desc(list_nvenc_rc_method_en, m_stEncConfig.rcParams.rateControlMode));
    if (NV_ENC_PARAMS_RC_CONSTQP == m_stEncConfig.rcParams.rateControlMode) {
        add_str(NV_LOG_ERROR, _T("CQP                     I:%d  P:%d  B:%d\n"), m_stEncConfig.rcParams.constQP.qpIntra, m_stEncConfig.rcParams.constQP.qpInterP, m_stEncConfig.rcParams.constQP.qpInterB);
    } else {
        add_str(NV_LOG_ERROR, _T("Bitrate                 %d kbps (Max: %d kbps)\n"), m_stEncConfig.rcParams.averageBitRate / 1000, m_stEncConfig.rcParams.maxBitRate / 1000);
        if (m_stEncConfig.rcParams.enableInitialRCQP)
            add_str(NV_LOG_INFO,  _T("Initial QP              I:%d  P:%d  B:%d\n"), m_stEncConfig.rcParams.constQP.qpIntra, m_stEncConfig.rcParams.constQP.qpInterP, m_stEncConfig.rcParams.constQP.qpInterB);
        if (m_stEncConfig.rcParams.enableMaxQP || m_stEncConfig.rcParams.enableMinQP) {
            int minQPI = (m_stEncConfig.rcParams.enableMinQP) ? m_stEncConfig.rcParams.minQP.qpIntra  :  0;
            int maxQPI = (m_stEncConfig.rcParams.enableMaxQP) ? m_stEncConfig.rcParams.maxQP.qpIntra  : 51;
            int minQPP = (m_stEncConfig.rcParams.enableMinQP) ? m_stEncConfig.rcParams.minQP.qpInterP :  0;
            int maxQPP = (m_stEncConfig.rcParams.enableMaxQP) ? m_stEncConfig.rcParams.maxQP.qpInterP : 51;
            int minQPB = (m_stEncConfig.rcParams.enableMinQP) ? m_stEncConfig.rcParams.minQP.qpInterB :  0;
            int maxQPB = (m_stEncConfig.rcParams.enableMaxQP) ? m_stEncConfig.rcParams.maxQP.qpInterB : 51;
            add_str(NV_LOG_INFO,  _T("QP range                I:%d-%d  P:%d-%d  B:%d-%d\n"), minQPI, maxQPI, minQPP, maxQPP, minQPB, maxQPB);
        }
        add_str(NV_LOG_INFO,  _T("VBV buffer size          %s\n"), value_or_auto(m_stEncConfig.rcParams.vbvBufferSize / 1000,   0, _T("kbit")).c_str());
        add_str(NV_LOG_DEBUG, _T("VBV initial delay        %s\n"), value_or_auto(m_stEncConfig.rcParams.vbvInitialDelay / 1000, 0, _T("kbit")).c_str());
    }
    add_str(NV_LOG_INFO,  _T("GOP length              %d frames\n"), m_stEncConfig.gopLength);
    add_str(NV_LOG_INFO,  _T("B frames                %d frames\n"), m_stEncConfig.frameIntervalP - 1);
    if (codec == NV_ENC_H264) {
        add_str(NV_LOG_DEBUG, _T("Output                  "));
        TCHAR bitstream_info[256] ={ 0 };
        if (m_stEncConfig.encodeCodecConfig.h264Config.outputBufferingPeriodSEI) _tcscat_s(bitstream_info, _countof(bitstream_info), _T("BufferingPeriodSEI,"));
        if (m_stEncConfig.encodeCodecConfig.h264Config.outputPictureTimingSEI)   _tcscat_s(bitstream_info, _countof(bitstream_info), _T("PicTimingSEI,"));
        if (m_stEncConfig.encodeCodecConfig.h264Config.outputAUD)                _tcscat_s(bitstream_info, _countof(bitstream_info), _T("AUD,"));
        if (m_stEncConfig.encodeCodecConfig.h264Config.outputFramePackingSEI)    _tcscat_s(bitstream_info, _countof(bitstream_info), _T("FramePackingSEI,"));
        if (m_stEncConfig.encodeCodecConfig.h264Config.outputRecoveryPointSEI)   _tcscat_s(bitstream_info, _countof(bitstream_info), _T("RecoveryPointSEI,"));
        if (m_stEncConfig.encodeCodecConfig.h264Config.repeatSPSPPS)             _tcscat_s(bitstream_info, _countof(bitstream_info), _T("repeatSPSPPS,"));
        if (_tcslen(bitstream_info)) {
            bitstream_info[_tcslen(bitstream_info)-1] = _T('\0');
        } else {
            _tcscpy_s(bitstream_info, _countof(bitstream_info), _T("-"));
        }
        add_str(NV_LOG_DEBUG, _T("%s\n"), bitstream_info);
    }

    add_str(NV_LOG_INFO,  _T("Ref frames              %d frames\n"), (codec == NV_ENC_H264) ? m_stEncConfig.encodeCodecConfig.h264Config.maxNumRefFrames : m_stEncConfig.encodeCodecConfig.hevcConfig.maxNumRefFramesInDPB);
    add_str(NV_LOG_INFO,  _T("MV Quality              %s\n"), get_desc(list_mv_presicion, m_stEncConfig.mvPrecision));
    if (codec == NV_ENC_H264 && 3 == m_stEncConfig.encodeCodecConfig.h264Config.sliceMode) {
        add_str(NV_LOG_DEBUG, _T("Slice number            %d\n"), m_stEncConfig.encodeCodecConfig.h264Config.sliceModeData);
    } else {
        add_str(NV_LOG_DEBUG, _T("Slice                   Mode:%d, ModeData:%d\n"), m_stEncConfig.encodeCodecConfig.h264Config.sliceMode, m_stEncConfig.encodeCodecConfig.h264Config.sliceModeData);
    }
    if (codec == NV_ENC_H264) {
        add_str(NV_LOG_INFO,  _T("CABAC/deblock           %s / %s\n"), get_desc(list_entropy_coding, m_stEncConfig.encodeCodecConfig.h264Config.entropyCodingMode), on_off(!m_stEncConfig.encodeCodecConfig.h264Config.disableDeblockingFilterIDC));
        add_str(NV_LOG_DEBUG, _T("hierarchical Frames     P:%s  B:%s\n"), on_off(m_stEncConfig.encodeCodecConfig.h264Config.hierarchicalPFrames), on_off(m_stEncConfig.encodeCodecConfig.h264Config.hierarchicalBFrames));
        add_str(NV_LOG_DEBUG, _T("Variable FrameRate      %s\n"), on_off(m_stEncConfig.encodeCodecConfig.h264Config.enableVFR));
        add_str(NV_LOG_DEBUG, _T("LTR                     %s"),   on_off(m_stEncConfig.encodeCodecConfig.h264Config.enableLTR));
        if (m_stEncConfig.encodeCodecConfig.h264Config.enableLTR) {
            add_str(NV_LOG_DEBUG, _T(", Mode:%d, NumFrames:%d"), m_stEncConfig.encodeCodecConfig.h264Config.ltrTrustMode, m_stEncConfig.encodeCodecConfig.h264Config.ltrNumFrames);
        }
        add_str(NV_LOG_DEBUG, _T("\n"));
        add_str(NV_LOG_DEBUG, _T("YUV 4:4:4               %s\n"), on_off(m_stEncConfig.encodeCodecConfig.h264Config.separateColourPlaneFlag));
        add_str(NV_LOG_DEBUG, _T("Adaptive Transform      %s\n"), get_desc(list_adapt_transform, m_stEncConfig.encodeCodecConfig.h264Config.adaptiveTransformMode));
        add_str(NV_LOG_DEBUG, _T("FMO                     %s\n"), get_desc(list_fmo, m_stEncConfig.encodeCodecConfig.h264Config.fmoMode));
        add_str(NV_LOG_DEBUG, _T("MV Mode                 %s\n"), get_desc(list_bdirect, m_stEncConfig.encodeCodecConfig.h264Config.bdirectMode));
    }
#endif
    return str;
}

int NVEncCore::PrintEncodingParamsInfo(int output_level) {
    return NVPrintf(stderr, NV_LOG_INFO, _T("%s"), GetEncodingParamsInfo(output_level).c_str());
}
