////////////////////////////////////////////////////////////////////////////
//
// Copyright 1993-2014 NVIDIA Corporation.  All rights reserved.
//
// Please refer to the NVIDIA end user license agreement (EULA) associated
// with this source code for terms and conditions that govern your use of
// this software. Any use, reproduction, disclosure, or distribution of
// this software and related documentation outside the terms of the EULA
// is strictly prohibited.
//
////////////////////////////////////////////////////////////////////////////

#include "../inc/NvHWEncoder.h"

NVENCSTATUS CNvHWEncoder::NvEncOpenEncodeSession(void* device, uint32_t deviceType)
{
    NVENCSTATUS nvStatus = NV_ENC_SUCCESS;

    nvStatus = m_pEncodeAPI->nvEncOpenEncodeSession(device, deviceType, &m_hEncoder);
    if (nvStatus != NV_ENC_SUCCESS)
    {
        assert(0);
    }

    return nvStatus;
}

NVENCSTATUS CNvHWEncoder::NvEncGetEncodeGUIDCount(uint32_t* encodeGUIDCount)
{
    NVENCSTATUS nvStatus = NV_ENC_SUCCESS;

    nvStatus = m_pEncodeAPI->nvEncGetEncodeGUIDCount(m_hEncoder, encodeGUIDCount);
    if (nvStatus != NV_ENC_SUCCESS)
    {
        assert(0);
    }

    return nvStatus;
}

NVENCSTATUS CNvHWEncoder::NvEncGetEncodeProfileGUIDCount(GUID encodeGUID, uint32_t* encodeProfileGUIDCount)
{
    NVENCSTATUS nvStatus = NV_ENC_SUCCESS;

    nvStatus = m_pEncodeAPI->nvEncGetEncodeProfileGUIDCount(m_hEncoder, encodeGUID, encodeProfileGUIDCount);
    if (nvStatus != NV_ENC_SUCCESS)
    {
        assert(0);
    }

    return nvStatus;
}

NVENCSTATUS CNvHWEncoder::NvEncGetEncodeProfileGUIDs(GUID encodeGUID, GUID* profileGUIDs, uint32_t guidArraySize, uint32_t* GUIDCount)
{
    NVENCSTATUS nvStatus = NV_ENC_SUCCESS;

    nvStatus = m_pEncodeAPI->nvEncGetEncodeProfileGUIDs(m_hEncoder, encodeGUID, profileGUIDs, guidArraySize, GUIDCount);
    if (nvStatus != NV_ENC_SUCCESS)
    {
        assert(0);
    }

    return nvStatus;
}

NVENCSTATUS CNvHWEncoder::NvEncGetEncodeGUIDs(GUID* GUIDs, uint32_t guidArraySize, uint32_t* GUIDCount)
{
    NVENCSTATUS nvStatus = NV_ENC_SUCCESS;

    nvStatus = m_pEncodeAPI->nvEncGetEncodeGUIDs(m_hEncoder, GUIDs, guidArraySize, GUIDCount);
    if (nvStatus != NV_ENC_SUCCESS)
    {
        assert(0);
    }

    return nvStatus;
}

NVENCSTATUS CNvHWEncoder::NvEncGetInputFormatCount(GUID encodeGUID, uint32_t* inputFmtCount)
{
    NVENCSTATUS nvStatus = NV_ENC_SUCCESS;

    nvStatus = m_pEncodeAPI->nvEncGetInputFormatCount(m_hEncoder, encodeGUID, inputFmtCount);
    if (nvStatus != NV_ENC_SUCCESS)
    {
        assert(0);
    }

    return nvStatus;
}

NVENCSTATUS CNvHWEncoder::NvEncGetInputFormats(GUID encodeGUID, NV_ENC_BUFFER_FORMAT* inputFmts, uint32_t inputFmtArraySize, uint32_t* inputFmtCount)
{
    NVENCSTATUS nvStatus = NV_ENC_SUCCESS;

    nvStatus = m_pEncodeAPI->nvEncGetInputFormats(m_hEncoder, encodeGUID, inputFmts, inputFmtArraySize, inputFmtCount);
    if (nvStatus != NV_ENC_SUCCESS)
    {
        assert(0);
    }

    return nvStatus;
}

NVENCSTATUS CNvHWEncoder::NvEncGetEncodeCaps(GUID encodeGUID, NV_ENC_CAPS_PARAM* capsParam, int* capsVal)
{
    NVENCSTATUS nvStatus = NV_ENC_SUCCESS;

    nvStatus = m_pEncodeAPI->nvEncGetEncodeCaps(m_hEncoder, encodeGUID, capsParam, capsVal);
    if (nvStatus != NV_ENC_SUCCESS)
    {
        assert(0);
    }

    return nvStatus;
}

NVENCSTATUS CNvHWEncoder::NvEncGetEncodePresetCount(GUID encodeGUID, uint32_t* encodePresetGUIDCount)
{
    NVENCSTATUS nvStatus = NV_ENC_SUCCESS;

    nvStatus = m_pEncodeAPI->nvEncGetEncodePresetCount(m_hEncoder, encodeGUID, encodePresetGUIDCount);
    if (nvStatus != NV_ENC_SUCCESS)
    {
        assert(0);
    }

    return nvStatus;
}

NVENCSTATUS CNvHWEncoder::NvEncGetEncodePresetGUIDs(GUID encodeGUID, GUID* presetGUIDs, uint32_t guidArraySize, uint32_t* encodePresetGUIDCount)
{
    NVENCSTATUS nvStatus = NV_ENC_SUCCESS;

    nvStatus = m_pEncodeAPI->nvEncGetEncodePresetGUIDs(m_hEncoder, encodeGUID, presetGUIDs, guidArraySize, encodePresetGUIDCount);
    if (nvStatus != NV_ENC_SUCCESS)
    {
        assert(0);
    }

    return nvStatus;
}

NVENCSTATUS CNvHWEncoder::NvEncGetEncodePresetConfig(GUID encodeGUID, GUID  presetGUID, NV_ENC_PRESET_CONFIG* presetConfig)
{
    NVENCSTATUS nvStatus = NV_ENC_SUCCESS;

    nvStatus = m_pEncodeAPI->nvEncGetEncodePresetConfig(m_hEncoder, encodeGUID, presetGUID, presetConfig);
    if (nvStatus != NV_ENC_SUCCESS)
    {
        assert(0);
    }

    return nvStatus;
}

NVENCSTATUS CNvHWEncoder::NvEncCreateInputBuffer(uint32_t width, uint32_t height, void** inputBuffer, uint32_t isYuv444)
{
    NVENCSTATUS nvStatus = NV_ENC_SUCCESS;
    NV_ENC_CREATE_INPUT_BUFFER createInputBufferParams;

    memset(&createInputBufferParams, 0, sizeof(createInputBufferParams));
    SET_VER(createInputBufferParams, NV_ENC_CREATE_INPUT_BUFFER);

    createInputBufferParams.width = width;
    createInputBufferParams.height = height;
    createInputBufferParams.memoryHeap = NV_ENC_MEMORY_HEAP_SYSMEM_CACHED;
    createInputBufferParams.bufferFmt = isYuv444 ? NV_ENC_BUFFER_FORMAT_YUV444_PL : NV_ENC_BUFFER_FORMAT_NV12_PL;

    nvStatus = m_pEncodeAPI->nvEncCreateInputBuffer(m_hEncoder, &createInputBufferParams);
    if (nvStatus != NV_ENC_SUCCESS)
    {
        assert(0);
    }

    *inputBuffer = createInputBufferParams.inputBuffer;

    return nvStatus;
}

NVENCSTATUS CNvHWEncoder::NvEncDestroyInputBuffer(NV_ENC_INPUT_PTR inputBuffer)
{
    NVENCSTATUS nvStatus = NV_ENC_SUCCESS;

    if (inputBuffer)
    {
        nvStatus = m_pEncodeAPI->nvEncDestroyInputBuffer(m_hEncoder, inputBuffer);
        if (nvStatus != NV_ENC_SUCCESS)
        {
            assert(0);
        }
    }

    return nvStatus;
}

NVENCSTATUS CNvHWEncoder::NvEncCreateBitstreamBuffer(uint32_t size, void** bitstreamBuffer)
{
    NVENCSTATUS nvStatus = NV_ENC_SUCCESS;
    NV_ENC_CREATE_BITSTREAM_BUFFER createBitstreamBufferParams;

    memset(&createBitstreamBufferParams, 0, sizeof(createBitstreamBufferParams));
    SET_VER(createBitstreamBufferParams, NV_ENC_CREATE_BITSTREAM_BUFFER);

    createBitstreamBufferParams.size = size;
    createBitstreamBufferParams.memoryHeap = NV_ENC_MEMORY_HEAP_SYSMEM_CACHED;

    nvStatus = m_pEncodeAPI->nvEncCreateBitstreamBuffer(m_hEncoder, &createBitstreamBufferParams);
    if (nvStatus != NV_ENC_SUCCESS)
    {
        assert(0);
    }

    *bitstreamBuffer = createBitstreamBufferParams.bitstreamBuffer;

    return nvStatus;
}

NVENCSTATUS CNvHWEncoder::NvEncDestroyBitstreamBuffer(NV_ENC_OUTPUT_PTR bitstreamBuffer)
{
    NVENCSTATUS nvStatus = NV_ENC_SUCCESS;

    if (bitstreamBuffer)
    {
        nvStatus = m_pEncodeAPI->nvEncDestroyBitstreamBuffer(m_hEncoder, bitstreamBuffer);
        if (nvStatus != NV_ENC_SUCCESS)
        {
            assert(0);
        }
    }

    return nvStatus;
}

NVENCSTATUS CNvHWEncoder::NvEncLockBitstream(NV_ENC_LOCK_BITSTREAM* lockBitstreamBufferParams)
{
    NVENCSTATUS nvStatus = NV_ENC_SUCCESS;

    nvStatus = m_pEncodeAPI->nvEncLockBitstream(m_hEncoder, lockBitstreamBufferParams);
    if (nvStatus != NV_ENC_SUCCESS)
    {
        assert(0);
    }

    return nvStatus;
}

NVENCSTATUS CNvHWEncoder::NvEncUnlockBitstream(NV_ENC_OUTPUT_PTR bitstreamBuffer)
{
    NVENCSTATUS nvStatus = NV_ENC_SUCCESS;

    nvStatus = m_pEncodeAPI->nvEncUnlockBitstream(m_hEncoder, bitstreamBuffer);
    if (nvStatus != NV_ENC_SUCCESS)
    {
        assert(0);
    }

    return nvStatus;
}

NVENCSTATUS CNvHWEncoder::NvEncLockInputBuffer(void* inputBuffer, void** bufferDataPtr, uint32_t* pitch)
{
    NVENCSTATUS nvStatus = NV_ENC_SUCCESS;
    NV_ENC_LOCK_INPUT_BUFFER lockInputBufferParams;

    memset(&lockInputBufferParams, 0, sizeof(lockInputBufferParams));
    SET_VER(lockInputBufferParams, NV_ENC_LOCK_INPUT_BUFFER);

    lockInputBufferParams.inputBuffer = inputBuffer;
    nvStatus = m_pEncodeAPI->nvEncLockInputBuffer(m_hEncoder, &lockInputBufferParams);
    if (nvStatus != NV_ENC_SUCCESS)
    {
        assert(0);
    }

    *bufferDataPtr = lockInputBufferParams.bufferDataPtr;
    *pitch = lockInputBufferParams.pitch;

    return nvStatus;
}

NVENCSTATUS CNvHWEncoder::NvEncUnlockInputBuffer(NV_ENC_INPUT_PTR inputBuffer)
{
    NVENCSTATUS nvStatus = NV_ENC_SUCCESS;

    nvStatus = m_pEncodeAPI->nvEncUnlockInputBuffer(m_hEncoder, inputBuffer);
    if (nvStatus != NV_ENC_SUCCESS)
    {
        assert(0);
    }

    return nvStatus;
}

NVENCSTATUS CNvHWEncoder::NvEncGetEncodeStats(NV_ENC_STAT* encodeStats)
{
    NVENCSTATUS nvStatus = NV_ENC_SUCCESS;

    nvStatus = m_pEncodeAPI->nvEncGetEncodeStats(m_hEncoder, encodeStats);
    if (nvStatus != NV_ENC_SUCCESS)
    {
        assert(0);
    }

    return nvStatus;
}

NVENCSTATUS CNvHWEncoder::NvEncGetSequenceParams(NV_ENC_SEQUENCE_PARAM_PAYLOAD* sequenceParamPayload)
{
    NVENCSTATUS nvStatus = NV_ENC_SUCCESS;

    nvStatus = m_pEncodeAPI->nvEncGetSequenceParams(m_hEncoder, sequenceParamPayload);
    if (nvStatus != NV_ENC_SUCCESS)
    {
        assert(0);
    }

    return nvStatus;
}

NVENCSTATUS CNvHWEncoder::NvEncRegisterAsyncEvent(void** completionEvent)
{
    NVENCSTATUS nvStatus = NV_ENC_SUCCESS;
    NV_ENC_EVENT_PARAMS eventParams;

    memset(&eventParams, 0, sizeof(eventParams));
    SET_VER(eventParams, NV_ENC_EVENT_PARAMS);

#if defined (NV_WINDOWS)
    eventParams.completionEvent = CreateEvent(NULL, FALSE, FALSE, NULL);
#else
    eventParams.completionEvent = NULL;
#endif
    nvStatus = m_pEncodeAPI->nvEncRegisterAsyncEvent(m_hEncoder, &eventParams);
    if (nvStatus != NV_ENC_SUCCESS)
    {
        assert(0);
    }

    *completionEvent = eventParams.completionEvent;

    return nvStatus;
}

NVENCSTATUS CNvHWEncoder::NvEncUnregisterAsyncEvent(void* completionEvent)
{
    NVENCSTATUS nvStatus = NV_ENC_SUCCESS;
    NV_ENC_EVENT_PARAMS eventParams;

    if (completionEvent)
    {
        memset(&eventParams, 0, sizeof(eventParams));
        SET_VER(eventParams, NV_ENC_EVENT_PARAMS);

        eventParams.completionEvent = completionEvent;

        nvStatus = m_pEncodeAPI->nvEncUnregisterAsyncEvent(m_hEncoder, &eventParams);
        if (nvStatus != NV_ENC_SUCCESS)
        {
            assert(0);
        }
    }

    return nvStatus;
}

NVENCSTATUS CNvHWEncoder::NvEncMapInputResource(void* registeredResource, void** mappedResource)
{
    NVENCSTATUS nvStatus = NV_ENC_SUCCESS;
    NV_ENC_MAP_INPUT_RESOURCE mapInputResParams;

    memset(&mapInputResParams, 0, sizeof(mapInputResParams));
    SET_VER(mapInputResParams, NV_ENC_MAP_INPUT_RESOURCE);

    mapInputResParams.registeredResource = registeredResource;

    nvStatus = m_pEncodeAPI->nvEncMapInputResource(m_hEncoder, &mapInputResParams);
    if (nvStatus != NV_ENC_SUCCESS)
    {
        assert(0);
    }

    *mappedResource = mapInputResParams.mappedResource;

    return nvStatus;
}

NVENCSTATUS CNvHWEncoder::NvEncUnmapInputResource(NV_ENC_INPUT_PTR mappedInputBuffer)
{
    NVENCSTATUS nvStatus = NV_ENC_SUCCESS;
    
    if (mappedInputBuffer)
    {
        nvStatus = m_pEncodeAPI->nvEncUnmapInputResource(m_hEncoder, mappedInputBuffer);
        if (nvStatus != NV_ENC_SUCCESS)
        {
            assert(0);
        }
    }

    return nvStatus;
}

NVENCSTATUS CNvHWEncoder::NvEncDestroyEncoder()
{
    NVENCSTATUS nvStatus = NV_ENC_SUCCESS;

    if (m_bEncoderInitialized)
    {
        nvStatus = m_pEncodeAPI->nvEncDestroyEncoder(m_hEncoder);

        m_bEncoderInitialized = false;
    }

    return nvStatus;
}

NVENCSTATUS CNvHWEncoder::NvEncInvalidateRefFrames(const NvEncPictureCommand *pEncPicCommand)
{
    NVENCSTATUS nvStatus = NV_ENC_SUCCESS;

    for (uint32_t i = 0; i < pEncPicCommand->numRefFramesToInvalidate; i++)
    {
        nvStatus = m_pEncodeAPI->nvEncInvalidateRefFrames(m_hEncoder, pEncPicCommand->refFrameNumbers[i]);
    }

    return nvStatus;
}

NVENCSTATUS CNvHWEncoder::NvEncOpenEncodeSessionEx(void* device, NV_ENC_DEVICE_TYPE deviceType)
{
    NVENCSTATUS nvStatus = NV_ENC_SUCCESS;
    NV_ENC_OPEN_ENCODE_SESSION_EX_PARAMS openSessionExParams;

    memset(&openSessionExParams, 0, sizeof(openSessionExParams));
    SET_VER(openSessionExParams, NV_ENC_OPEN_ENCODE_SESSION_EX_PARAMS);

    openSessionExParams.device = device;
    openSessionExParams.deviceType = deviceType;
    openSessionExParams.reserved = NULL;
    openSessionExParams.apiVersion = NVENCAPI_VERSION;

    nvStatus = m_pEncodeAPI->nvEncOpenEncodeSessionEx(&openSessionExParams, &m_hEncoder);
    if (nvStatus != NV_ENC_SUCCESS)
    {
        assert(0);
    }

    return nvStatus;
}

NVENCSTATUS CNvHWEncoder::NvEncRegisterResource(NV_ENC_INPUT_RESOURCE_TYPE resourceType, void* resourceToRegister, uint32_t width, uint32_t height, uint32_t pitch, void** registeredResource)
{
    NVENCSTATUS nvStatus = NV_ENC_SUCCESS;
    NV_ENC_REGISTER_RESOURCE registerResParams;

    memset(&registerResParams, 0, sizeof(registerResParams));
    SET_VER(registerResParams, NV_ENC_REGISTER_RESOURCE);

    registerResParams.resourceType = resourceType;
    registerResParams.resourceToRegister = resourceToRegister;
    registerResParams.width = width;
    registerResParams.height = height;
    registerResParams.pitch = pitch;

    nvStatus = m_pEncodeAPI->nvEncRegisterResource(m_hEncoder, &registerResParams);
    if (nvStatus != NV_ENC_SUCCESS)
    {
        assert(0);
    }

    *registeredResource = registerResParams.registeredResource;

    return nvStatus;
}

NVENCSTATUS CNvHWEncoder::NvEncUnregisterResource(NV_ENC_REGISTERED_PTR registeredRes)
{
    NVENCSTATUS nvStatus = NV_ENC_SUCCESS;

    nvStatus = m_pEncodeAPI->nvEncUnregisterResource(m_hEncoder, registeredRes);
    if (nvStatus != NV_ENC_SUCCESS)
    {
        assert(0);
    }

    return nvStatus;
}

NVENCSTATUS CNvHWEncoder::NvEncReconfigureEncoder(const NvEncPictureCommand *pEncPicCommand)
{
    NVENCSTATUS nvStatus = NV_ENC_SUCCESS;

    if (pEncPicCommand->bBitrateChangePending || pEncPicCommand->bResolutionChangePending)
    {
        if (pEncPicCommand->bResolutionChangePending)
        {
            m_uCurWidth = pEncPicCommand->newWidth;
            m_uCurHeight = pEncPicCommand->newHeight;
            if ((m_uCurWidth > m_uMaxWidth) || (m_uCurHeight > m_uMaxHeight))
            {
                return NV_ENC_ERR_INVALID_PARAM;
            }
            m_stCreateEncodeParams.encodeWidth = m_uCurWidth;
            m_stCreateEncodeParams.encodeHeight = m_uCurHeight;
            m_stCreateEncodeParams.darWidth = m_uCurWidth;
            m_stCreateEncodeParams.darHeight = m_uCurHeight;
        }

        if (pEncPicCommand->bBitrateChangePending)
        {
            m_stEncodeConfig.rcParams.averageBitRate = pEncPicCommand->newBitrate;
            m_stEncodeConfig.rcParams.maxBitRate = pEncPicCommand->newBitrate;
            m_stEncodeConfig.rcParams.vbvBufferSize = pEncPicCommand->newVBVSize != 0 ? pEncPicCommand->newVBVSize : (pEncPicCommand->newBitrate * m_stCreateEncodeParams.frameRateDen) / m_stCreateEncodeParams.frameRateNum;
            m_stEncodeConfig.rcParams.vbvInitialDelay = m_stEncodeConfig.rcParams.vbvBufferSize;
        }

        NV_ENC_RECONFIGURE_PARAMS stReconfigParams;
        memset(&stReconfigParams, 0, sizeof(stReconfigParams));
        memcpy(&stReconfigParams.reInitEncodeParams, &m_stCreateEncodeParams, sizeof(m_stCreateEncodeParams));
        stReconfigParams.version = NV_ENC_RECONFIGURE_PARAMS_VER;
        stReconfigParams.forceIDR = pEncPicCommand->bResolutionChangePending ? 1 : 0;

        nvStatus = m_pEncodeAPI->nvEncReconfigureEncoder(m_hEncoder, &stReconfigParams);
        if (nvStatus != NV_ENC_SUCCESS)
        {
            assert(0);
        }
    }

    return nvStatus;
}

CNvHWEncoder::CNvHWEncoder()
{
    m_hEncoder = NULL;
    m_bEncoderInitialized = false;
    m_pEncodeAPI = NULL;
    m_hinstLib = NULL;
    m_fOutput = NULL;
    m_EncodeIdx = 0;
    m_uCurWidth = 0;
    m_uCurHeight = 0;
    m_uMaxWidth = 0;
    m_uMaxHeight = 0;

    memset(&m_stCreateEncodeParams, 0, sizeof(m_stCreateEncodeParams));
    SET_VER(m_stCreateEncodeParams, NV_ENC_INITIALIZE_PARAMS);

    memset(&m_stEncodeConfig, 0, sizeof(m_stEncodeConfig));
    SET_VER(m_stEncodeConfig, NV_ENC_CONFIG);
}

CNvHWEncoder::~CNvHWEncoder()
{
    // clean up encode API resources here
    if (m_pEncodeAPI)
    {
        delete m_pEncodeAPI;
        m_pEncodeAPI = NULL;
    }

    if (m_hinstLib)
    {
#if defined (NV_WINDOWS)
        FreeLibrary(m_hinstLib);
#else
        dlclose(m_hinstLib);
#endif

        m_hinstLib = NULL;
    }
}

NVENCSTATUS CNvHWEncoder::ValidateEncodeGUID (GUID inputCodecGuid)
{
    unsigned int i, codecFound, encodeGUIDCount, encodeGUIDArraySize;
    NVENCSTATUS nvStatus;
    GUID *encodeGUIDArray;

    nvStatus = m_pEncodeAPI->nvEncGetEncodeGUIDCount(m_hEncoder, &encodeGUIDCount);
    if (nvStatus != NV_ENC_SUCCESS)
    {
        assert(0);
        return nvStatus;
    }

    encodeGUIDArray = new GUID[encodeGUIDCount];
    memset(encodeGUIDArray, 0, sizeof(GUID)* encodeGUIDCount);

    encodeGUIDArraySize = 0;
    nvStatus = m_pEncodeAPI->nvEncGetEncodeGUIDs(m_hEncoder, encodeGUIDArray, encodeGUIDCount, &encodeGUIDArraySize);
    if (nvStatus != NV_ENC_SUCCESS)
    {
        delete[] encodeGUIDArray;
        assert(0);
        return nvStatus;
    }

    assert(encodeGUIDArraySize <= encodeGUIDCount);

    codecFound = 0;
    for (i = 0; i < encodeGUIDArraySize; i++)
    {
        if (inputCodecGuid == encodeGUIDArray[i])
        {
            codecFound = 1;
            break;
        }
    }

    delete[] encodeGUIDArray;

    if (codecFound)
        return NV_ENC_SUCCESS;
    else
        return NV_ENC_ERR_INVALID_PARAM;
}

NVENCSTATUS CNvHWEncoder::ValidatePresetGUID(GUID inputPresetGuid, GUID inputCodecGuid)
{
    uint32_t i, presetFound, presetGUIDCount, presetGUIDArraySize;
    NVENCSTATUS nvStatus;
    GUID *presetGUIDArray;

    nvStatus = m_pEncodeAPI->nvEncGetEncodePresetCount(m_hEncoder, inputCodecGuid, &presetGUIDCount);
    if (nvStatus != NV_ENC_SUCCESS)
    {
        assert(0);
        return nvStatus;
    }

    presetGUIDArray = new GUID[presetGUIDCount];
    memset(presetGUIDArray, 0, sizeof(GUID)* presetGUIDCount);

    presetGUIDArraySize = 0;
    nvStatus = m_pEncodeAPI->nvEncGetEncodePresetGUIDs(m_hEncoder, inputCodecGuid, presetGUIDArray, presetGUIDCount, &presetGUIDArraySize);
    if (nvStatus != NV_ENC_SUCCESS)
    {
        assert(0);
        delete[] presetGUIDArray;
        return nvStatus;
    }

    assert(presetGUIDArraySize <= presetGUIDCount);

    presetFound = 0;
    for (i = 0; i < presetGUIDArraySize; i++)
    {
        if (inputPresetGuid == presetGUIDArray[i])
        {
            presetFound = 1;
            break;
        }
    }

    delete[] presetGUIDArray;

    if (presetFound)
        return NV_ENC_SUCCESS;
    else
        return NV_ENC_ERR_INVALID_PARAM;
}

NVENCSTATUS CNvHWEncoder::CreateEncoder(const EncodeConfig *pEncCfg)
{
    NVENCSTATUS nvStatus = NV_ENC_SUCCESS;

    if (pEncCfg == NULL)
    {
        return NV_ENC_ERR_INVALID_PARAM;
    }

    m_uCurWidth = pEncCfg->width;
    m_uCurHeight = pEncCfg->height;

    m_uMaxWidth = (pEncCfg->maxWidth > 0 ? pEncCfg->maxWidth : pEncCfg->width);
    m_uMaxHeight = (pEncCfg->maxHeight > 0 ? pEncCfg->maxHeight : pEncCfg->height);

    if ((m_uCurWidth > m_uMaxWidth) || (m_uCurHeight > m_uMaxHeight)) {
        return NV_ENC_ERR_INVALID_PARAM;
    }

    m_fOutput = pEncCfg->fOutput;

    if (!pEncCfg->width || !pEncCfg->height || !m_fOutput)
    {
        return NV_ENC_ERR_INVALID_PARAM;
    }

    GUID inputCodecGUID = pEncCfg->codec == NV_ENC_H264 ? NV_ENC_CODEC_H264_GUID : NV_ENC_CODEC_HEVC_GUID;
    nvStatus = ValidateEncodeGUID(inputCodecGUID);
    if (nvStatus != NV_ENC_SUCCESS)
    {
        PRINTERR("codec not supported \n");
        return nvStatus;
    }

    codecGUID = inputCodecGUID;

    m_stCreateEncodeParams.encodeGUID = inputCodecGUID;
    m_stCreateEncodeParams.presetGUID = pEncCfg->presetGUID;
    m_stCreateEncodeParams.encodeWidth = pEncCfg->width;
    m_stCreateEncodeParams.encodeHeight = pEncCfg->height;

    m_stCreateEncodeParams.darWidth = pEncCfg->width;
    m_stCreateEncodeParams.darHeight = pEncCfg->height;
    m_stCreateEncodeParams.frameRateNum = pEncCfg->fps;
    m_stCreateEncodeParams.frameRateDen = 1;
#if defined(NV_WINDOWS)
    m_stCreateEncodeParams.enableEncodeAsync = 1;
#else
    m_stCreateEncodeParams.enableEncodeAsync = 0;
#endif
    m_stCreateEncodeParams.enablePTD = 1;
    m_stCreateEncodeParams.reportSliceOffsets = 0;
    m_stCreateEncodeParams.enableSubFrameWrite = 0;
    m_stCreateEncodeParams.encodeConfig = &m_stEncodeConfig;
    m_stCreateEncodeParams.maxEncodeWidth = m_uMaxWidth;
    m_stCreateEncodeParams.maxEncodeHeight = m_uMaxHeight;

    // apply preset
    NV_ENC_PRESET_CONFIG stPresetCfg;
    memset(&stPresetCfg, 0, sizeof(NV_ENC_PRESET_CONFIG));
    SET_VER(stPresetCfg, NV_ENC_PRESET_CONFIG);
    SET_VER(stPresetCfg.presetCfg, NV_ENC_CONFIG);

    nvStatus = m_pEncodeAPI->nvEncGetEncodePresetConfig(m_hEncoder, m_stCreateEncodeParams.encodeGUID, m_stCreateEncodeParams.presetGUID, &stPresetCfg);
    if (nvStatus != NV_ENC_SUCCESS)
    {
        assert(0);
        return nvStatus;
    }
    memcpy(&m_stEncodeConfig, &stPresetCfg.presetCfg, sizeof(NV_ENC_CONFIG));

    m_stEncodeConfig.gopLength = pEncCfg->gopLength;
    m_stEncodeConfig.frameIntervalP = pEncCfg->numB + 1;
    if (pEncCfg->pictureStruct == NV_ENC_PIC_STRUCT_FRAME)
    {
        m_stEncodeConfig.frameFieldMode = NV_ENC_PARAMS_FRAME_FIELD_MODE_FRAME;
    }
    else
    {
        m_stEncodeConfig.frameFieldMode = NV_ENC_PARAMS_FRAME_FIELD_MODE_FIELD;
    }

    m_stEncodeConfig.mvPrecision = NV_ENC_MV_PRECISION_QUARTER_PEL;

    if (pEncCfg->bitrate || pEncCfg->vbvMaxBitrate)
    {
        m_stEncodeConfig.rcParams.rateControlMode = (NV_ENC_PARAMS_RC_MODE)pEncCfg->rcMode;
        m_stEncodeConfig.rcParams.averageBitRate = pEncCfg->bitrate;
        m_stEncodeConfig.rcParams.maxBitRate = pEncCfg->vbvMaxBitrate;
        m_stEncodeConfig.rcParams.vbvBufferSize = pEncCfg->vbvSize;
        m_stEncodeConfig.rcParams.vbvInitialDelay = pEncCfg->vbvSize * 9 / 10;
    }
    else
    {
        m_stEncodeConfig.rcParams.rateControlMode = NV_ENC_PARAMS_RC_CONSTQP;
    }

    if (pEncCfg->rcMode == 0)
    {
        m_stEncodeConfig.rcParams.constQP.qpInterP = pEncCfg->presetGUID == NV_ENC_PRESET_LOSSLESS_HP_GUID? 0 : pEncCfg->qp;
        m_stEncodeConfig.rcParams.constQP.qpInterB = pEncCfg->presetGUID == NV_ENC_PRESET_LOSSLESS_HP_GUID? 0 : pEncCfg->qp;
        m_stEncodeConfig.rcParams.constQP.qpIntra = pEncCfg->presetGUID == NV_ENC_PRESET_LOSSLESS_HP_GUID? 0 : pEncCfg->qp;
    }

    if (pEncCfg->isYuv444)
    {
        m_stEncodeConfig.encodeCodecConfig.h264Config.chromaFormatIDC = 3;
    }
    else
    {
        m_stEncodeConfig.encodeCodecConfig.h264Config.chromaFormatIDC = 1;
    }

    if (pEncCfg->intraRefreshEnableFlag)
    {
        if (pEncCfg->codec == NV_ENC_HEVC)
        {
            m_stEncodeConfig.encodeCodecConfig.hevcConfig.enableIntraRefresh = 1;
            m_stEncodeConfig.encodeCodecConfig.hevcConfig.intraRefreshPeriod = pEncCfg->intraRefreshPeriod;
            m_stEncodeConfig.encodeCodecConfig.hevcConfig.intraRefreshCnt = pEncCfg->intraRefreshDuration;
        }
        else
        {
            m_stEncodeConfig.encodeCodecConfig.h264Config.enableIntraRefresh = 1;
            m_stEncodeConfig.encodeCodecConfig.h264Config.intraRefreshPeriod = pEncCfg->intraRefreshPeriod;
            m_stEncodeConfig.encodeCodecConfig.h264Config.intraRefreshCnt = pEncCfg->intraRefreshDuration;
        }
    }

    if (pEncCfg->invalidateRefFramesEnableFlag)
    {
        if (pEncCfg->codec == NV_ENC_HEVC)
        {
            m_stEncodeConfig.encodeCodecConfig.hevcConfig.maxNumRefFramesInDPB = 16;
        }
        else
        {
            m_stEncodeConfig.encodeCodecConfig.h264Config.maxNumRefFrames = 16;
        }
    }

    if (pEncCfg->qpDeltaMapFile)
    {
        m_stEncodeConfig.rcParams.enableExtQPDeltaMap = 1;
    }
    if (pEncCfg->codec == NV_ENC_H264)
    {
        m_stEncodeConfig.encodeCodecConfig.h264Config.idrPeriod = pEncCfg->gopLength;
    }
    else if (pEncCfg->codec == NV_ENC_HEVC)
    {
        m_stEncodeConfig.encodeCodecConfig.hevcConfig.idrPeriod = pEncCfg->gopLength;
    }

    nvStatus = m_pEncodeAPI->nvEncInitializeEncoder(m_hEncoder, &m_stCreateEncodeParams);
    if (nvStatus != NV_ENC_SUCCESS)
        return nvStatus;

    m_bEncoderInitialized = true;

    return nvStatus;
}

GUID CNvHWEncoder::GetPresetGUID(char* encoderPreset, int codec)
{
    NVENCSTATUS nvStatus = NV_ENC_SUCCESS;
    GUID presetGUID = NV_ENC_PRESET_DEFAULT_GUID;

    if (encoderPreset && (stricmp(encoderPreset, "hq") == 0))
    {
        presetGUID = NV_ENC_PRESET_HQ_GUID;
    }
    else if (encoderPreset && (stricmp(encoderPreset, "lowLatencyHP") == 0))
    {
        presetGUID = NV_ENC_PRESET_LOW_LATENCY_HP_GUID;
    }
    else if (encoderPreset && (stricmp(encoderPreset, "hp") == 0))
    {
        presetGUID = NV_ENC_PRESET_HP_GUID;
    }
    else if (encoderPreset && (stricmp(encoderPreset, "lowLatencyHQ") == 0))
    {
        presetGUID = NV_ENC_PRESET_LOW_LATENCY_HQ_GUID;
    }
    else if (encoderPreset && (stricmp(encoderPreset, "lossless") == 0))
    {
        presetGUID = NV_ENC_PRESET_LOSSLESS_HP_GUID;
    }
    else
    {
        presetGUID = NV_ENC_PRESET_DEFAULT_GUID;
    }

    GUID inputCodecGUID = codec == NV_ENC_H264 ? NV_ENC_CODEC_H264_GUID : NV_ENC_CODEC_HEVC_GUID;
    nvStatus = ValidatePresetGUID(presetGUID, inputCodecGUID);
    if (nvStatus != NV_ENC_SUCCESS)
    {
        presetGUID = NV_ENC_PRESET_DEFAULT_GUID;
        PRINTERR("Unsupported preset guid %s\n", encoderPreset);
    }

    return presetGUID;
}

NVENCSTATUS CNvHWEncoder::ProcessOutput(const EncodeBuffer *pEncodeBuffer)
{
    NVENCSTATUS nvStatus = NV_ENC_SUCCESS;

    if (pEncodeBuffer->stOutputBfr.hBitstreamBuffer == NULL && pEncodeBuffer->stOutputBfr.bEOSFlag == FALSE)
    {
        return NV_ENC_ERR_INVALID_PARAM;
    }

    if (pEncodeBuffer->stOutputBfr.bWaitOnEvent == TRUE)
    {
        if (!pEncodeBuffer->stOutputBfr.hOutputEvent)
        {
            return NV_ENC_ERR_INVALID_PARAM;
        }
#if defined(NV_WINDOWS)
        WaitForSingleObject(pEncodeBuffer->stOutputBfr.hOutputEvent, INFINITE);
#endif
    }

    if (pEncodeBuffer->stOutputBfr.bEOSFlag)
        return NV_ENC_SUCCESS;

    nvStatus = NV_ENC_SUCCESS;
    NV_ENC_LOCK_BITSTREAM lockBitstreamData;
    memset(&lockBitstreamData, 0, sizeof(lockBitstreamData));
    SET_VER(lockBitstreamData, NV_ENC_LOCK_BITSTREAM);
    lockBitstreamData.outputBitstream = pEncodeBuffer->stOutputBfr.hBitstreamBuffer;
    lockBitstreamData.doNotWait = false;

    nvStatus = m_pEncodeAPI->nvEncLockBitstream(m_hEncoder, &lockBitstreamData);
    if (nvStatus == NV_ENC_SUCCESS)
    {
        fwrite(lockBitstreamData.bitstreamBufferPtr, 1, lockBitstreamData.bitstreamSizeInBytes, m_fOutput);
        nvStatus = m_pEncodeAPI->nvEncUnlockBitstream(m_hEncoder, pEncodeBuffer->stOutputBfr.hBitstreamBuffer);
    }
    else
    {
        PRINTERR("lock bitstream function failed \n");
    }

    return nvStatus;
}

NVENCSTATUS CNvHWEncoder::Initialize(void* device, NV_ENC_DEVICE_TYPE deviceType)
{
    NVENCSTATUS nvStatus = NV_ENC_SUCCESS;
    MYPROC nvEncodeAPICreateInstance; // function pointer to create instance in nvEncodeAPI

#if defined(NV_WINDOWS)
#if defined (_WIN64)
    m_hinstLib = LoadLibrary(TEXT("nvEncodeAPI64.dll"));
#else
    m_hinstLib = LoadLibrary(TEXT("nvEncodeAPI.dll"));
#endif
#else
    m_hinstLib = dlopen("libnvidia-encode.so.1", RTLD_LAZY);
#endif
    if (m_hinstLib == NULL)
        return NV_ENC_ERR_OUT_OF_MEMORY;

#if defined(NV_WINDOWS)
    nvEncodeAPICreateInstance = (MYPROC)GetProcAddress(m_hinstLib, "NvEncodeAPICreateInstance");
#else
    nvEncodeAPICreateInstance = (MYPROC)dlsym(m_hinstLib, "NvEncodeAPICreateInstance");
#endif

    if (nvEncodeAPICreateInstance == NULL)
        return NV_ENC_ERR_OUT_OF_MEMORY;

    m_pEncodeAPI = new NV_ENCODE_API_FUNCTION_LIST;
    if (m_pEncodeAPI == NULL)
        return NV_ENC_ERR_OUT_OF_MEMORY;

    memset(m_pEncodeAPI, 0, sizeof(NV_ENCODE_API_FUNCTION_LIST));
    m_pEncodeAPI->version = NV_ENCODE_API_FUNCTION_LIST_VER;
    nvStatus = nvEncodeAPICreateInstance(m_pEncodeAPI);
    if (nvStatus != NV_ENC_SUCCESS)
        return nvStatus;

    nvStatus = NvEncOpenEncodeSessionEx(device, deviceType);
    if (nvStatus != NV_ENC_SUCCESS)
        return nvStatus;

    return NV_ENC_SUCCESS;
}

NVENCSTATUS CNvHWEncoder::NvEncEncodeFrame(EncodeBuffer *pEncodeBuffer, NvEncPictureCommand *encPicCommand,
                                           uint32_t width, uint32_t height, NV_ENC_PIC_STRUCT ePicStruct,
                                           int8_t *qpDeltaMapArray, uint32_t qpDeltaMapArraySize)
{
    NVENCSTATUS nvStatus = NV_ENC_SUCCESS;
    NV_ENC_PIC_PARAMS encPicParams;

    memset(&encPicParams, 0, sizeof(encPicParams));
    SET_VER(encPicParams, NV_ENC_PIC_PARAMS);


    encPicParams.inputBuffer = pEncodeBuffer->stInputBfr.hInputSurface;
    encPicParams.bufferFmt = pEncodeBuffer->stInputBfr.bufferFmt;
    encPicParams.inputWidth = width;
    encPicParams.inputHeight = height;
    encPicParams.outputBitstream = pEncodeBuffer->stOutputBfr.hBitstreamBuffer;
    encPicParams.completionEvent = pEncodeBuffer->stOutputBfr.hOutputEvent;
    encPicParams.inputTimeStamp = m_EncodeIdx;
    encPicParams.pictureStruct = ePicStruct;
    encPicParams.qpDeltaMap = qpDeltaMapArray;
    encPicParams.qpDeltaMapSize = qpDeltaMapArraySize;


    if (encPicCommand)
    {
        if (encPicCommand->bForceIDR)
        {
            encPicParams.encodePicFlags |= NV_ENC_PIC_FLAG_FORCEIDR;
        }

        if (encPicCommand->bForceIntraRefresh)
        {
            if (codecGUID == NV_ENC_CODEC_HEVC_GUID)
            {
                encPicParams.codecPicParams.hevcPicParams.forceIntraRefreshWithFrameCnt = encPicCommand->intraRefreshDuration;
            }
            else
            {
                encPicParams.codecPicParams.h264PicParams.forceIntraRefreshWithFrameCnt = encPicCommand->intraRefreshDuration;
            }
        }
    }

    nvStatus = m_pEncodeAPI->nvEncEncodePicture(m_hEncoder, &encPicParams);
    if (nvStatus != NV_ENC_SUCCESS && nvStatus != NV_ENC_ERR_NEED_MORE_INPUT)
    {
        assert(0);
        return nvStatus;
    }

    m_EncodeIdx++;

    return NV_ENC_SUCCESS;
}

NVENCSTATUS CNvHWEncoder::NvEncFlushEncoderQueue(void *hEOSEvent)
{
    NVENCSTATUS nvStatus = NV_ENC_SUCCESS;
    NV_ENC_PIC_PARAMS encPicParams;
    memset(&encPicParams, 0, sizeof(encPicParams));
    SET_VER(encPicParams, NV_ENC_PIC_PARAMS);
    encPicParams.encodePicFlags = NV_ENC_PIC_FLAG_EOS;
    encPicParams.completionEvent = hEOSEvent;
    nvStatus = m_pEncodeAPI->nvEncEncodePicture(m_hEncoder, &encPicParams);
    if (nvStatus != NV_ENC_SUCCESS)
    {
        assert(0);
    }
    return nvStatus;
}

NVENCSTATUS CNvHWEncoder::ParseArguments(EncodeConfig *encodeConfig, int argc, char *argv[])
{
    for (int i = 1; i < argc; i++)
    {
        if (stricmp(argv[i], "-bmpfilePath") == 0)
        {
            if (++i >= argc)
            {
                PRINTERR("invalid parameter for %s\n", argv[i - 1]);
                return NV_ENC_ERR_INVALID_PARAM;
            }
            encodeConfig->inputFilePath = argv[i];
        }
        else if (stricmp(argv[i], "-i") == 0)
        {
            if (++i >= argc)
            {
                PRINTERR("invalid parameter for %s\n", argv[i - 1]);
                return NV_ENC_ERR_INVALID_PARAM;
            }
            encodeConfig->inputFileName = argv[i];
        }
        else if (stricmp(argv[i], "-o") == 0)
        {
            if (++i >= argc)
            {
                PRINTERR("invalid parameter for %s\n", argv[i - 1]);
                return NV_ENC_ERR_INVALID_PARAM;
            }
            encodeConfig->outputFileName = argv[i];
        }
        else if (stricmp(argv[i], "-size") == 0)
        {
            if (++i >= argc || sscanf(argv[i], "%d", &encodeConfig->width) != 1)
            {
                PRINTERR("invalid parameter for %s\n", argv[i - 1]);
                return NV_ENC_ERR_INVALID_PARAM;
            }

            if (++i >= argc || sscanf(argv[i], "%d", &encodeConfig->height) != 1)
            {
                PRINTERR("invalid parameter for %s\n", argv[i - 2]);
                return NV_ENC_ERR_INVALID_PARAM;
            }
        }
        else if (stricmp(argv[i], "-maxSize") == 0)
        {
            if (++i >= argc || sscanf(argv[i], "%d", &encodeConfig->maxWidth) != 1)
            {
                PRINTERR("invalid parameter for %s\n", argv[i - 1]);
                return NV_ENC_ERR_INVALID_PARAM;
            }

            if (++i >= argc || sscanf(argv[i], "%d", &encodeConfig->maxHeight) != 1)
            {
                PRINTERR("invalid parameter for %s\n", argv[i - 2]);
                return NV_ENC_ERR_INVALID_PARAM;
            }
        }
        else if (stricmp(argv[i], "-bitrate") == 0)
        {
            if (++i >= argc || sscanf(argv[i], "%d", &encodeConfig->bitrate) != 1)
            {
                PRINTERR("invalid parameter for %s\n", argv[i - 1]);
                return NV_ENC_ERR_INVALID_PARAM;
            }
        }
        else if (stricmp(argv[i], "-vbvMaxBitrate") == 0)
        {
            if (++i >= argc || sscanf(argv[i], "%d", &encodeConfig->vbvMaxBitrate) != 1)
            {
                PRINTERR("invalid parameter for %s\n", argv[i - 1]);
                return NV_ENC_ERR_INVALID_PARAM;
            }
        }
        else if (stricmp(argv[i], "-vbvSize") == 0)
        {
            if (++i >= argc || sscanf(argv[i], "%d", &encodeConfig->vbvSize) != 1)
            {
                PRINTERR("invalid parameter for %s\n", argv[i - 1]);
                return NV_ENC_ERR_INVALID_PARAM;
            }
        }
        else if (stricmp(argv[i], "-fps") == 0)
        {
            if (++i >= argc || sscanf(argv[i], "%d", &encodeConfig->fps) != 1)
            {
                PRINTERR("invalid parameter for %s\n", argv[i - 1]);
                return NV_ENC_ERR_INVALID_PARAM;
            }
        }
        else if (stricmp(argv[i], "-startf") == 0)
        {
            if (++i >= argc || sscanf(argv[i], "%d", &encodeConfig->startFrameIdx) != 1)
            {
                PRINTERR("invalid parameter for %s\n", argv[i - 1]);
                return NV_ENC_ERR_INVALID_PARAM;
            }
        }
        else if (stricmp(argv[i], "-endf") == 0)
        {
            if (++i >= argc || sscanf(argv[i], "%d", &encodeConfig->endFrameIdx) != 1)
            {
                PRINTERR("invalid parameter for %s\n", argv[i - 1]);
                return NV_ENC_ERR_INVALID_PARAM;
            }
        }
        else if (stricmp(argv[i], "-rcmode") == 0)
        {
            if (++i >= argc || sscanf(argv[i], "%d", &encodeConfig->rcMode) != 1)
            {
                PRINTERR("invalid parameter for %s\n", argv[i - 1]);
                return NV_ENC_ERR_INVALID_PARAM;
            }
        }
        else if (stricmp(argv[i], "-goplength") == 0)
        {
            if (++i >= argc || sscanf(argv[i], "%d", &encodeConfig->gopLength) != 1)
            {
                PRINTERR("invalid parameter for %s\n", argv[i - 1]);
                return NV_ENC_ERR_INVALID_PARAM;
            }
        }
        else if (stricmp(argv[i], "-numB") == 0)
        {
            if (++i >= argc || sscanf(argv[i], "%d", &encodeConfig->numB) != 1)
            {
                PRINTERR("invalid parameter for %s\n", argv[i - 1]);
                return NV_ENC_ERR_INVALID_PARAM;
            }
        }
        else if (stricmp(argv[i], "-qp") == 0)
        {
            if (++i >= argc || sscanf(argv[i], "%d", &encodeConfig->qp) != 1)
            {
                PRINTERR("invalid parameter for %s\n", argv[i - 1]);
                return NV_ENC_ERR_INVALID_PARAM;
            }
        }
        else if (stricmp(argv[i], "-preset") == 0)
        {
            if (++i >= argc)
            {
                PRINTERR("invalid parameter for %s\n", argv[i - 1]);
                return NV_ENC_ERR_INVALID_PARAM;
            }
            encodeConfig->encoderPreset = argv[i];
        }
        else if (stricmp(argv[i], "-devicetype") == 0)
        {
            if (++i >= argc || sscanf(argv[i], "%d", &encodeConfig->deviceType) != 1)
            {
                PRINTERR("invalid parameter for %s\n", argv[i - 1]);
                return NV_ENC_ERR_INVALID_PARAM;
            }
        }
        else if (stricmp(argv[i], "-codec") == 0)
        {
            if (++i >= argc || sscanf(argv[i], "%d", &encodeConfig->codec) != 1)
            {
                PRINTERR("invalid parameter for %s\n", argv[i - 1]);
                return NV_ENC_ERR_INVALID_PARAM;
            }
        }
        else if (stricmp(argv[i], "-encCmdFile") == 0)
        {
            if (++i >= argc)
            {
                PRINTERR("invalid parameter for %s\n", argv[i - 1]);
                return NV_ENC_ERR_INVALID_PARAM;
            }
            encodeConfig->encCmdFileName = argv[i];
        }
        else if (stricmp(argv[i], "-intraRefresh") == 0)
        {
            if (++i >= argc || sscanf(argv[i], "%d", &encodeConfig->intraRefreshEnableFlag) != 1)
            {
                PRINTERR("invalid parameter for %s\n", argv[i - 1]);
                return NV_ENC_ERR_INVALID_PARAM;
            }
        }
        else if (stricmp(argv[i], "-intraRefreshPeriod") == 0)
        {
            if (++i >= argc || sscanf(argv[i], "%d", &encodeConfig->intraRefreshPeriod) != 1)
            {
                PRINTERR("invalid parameter for %s\n", argv[i - 1]);
                return NV_ENC_ERR_INVALID_PARAM;
            }
        }
        else if (stricmp(argv[i], "-intraRefreshDuration") == 0)
        {
            if (++i >= argc || sscanf(argv[i], "%d", &encodeConfig->intraRefreshDuration) != 1)
            {
                PRINTERR("invalid parameter for %s\n", argv[i - 1]);
                return NV_ENC_ERR_INVALID_PARAM;
            }
        }
        else if (stricmp(argv[i], "-picStruct") == 0)
        {
            if (++i >= argc || sscanf(argv[i], "%d", &encodeConfig->pictureStruct) != 1)
            {
                PRINTERR("invalid parameter for %s\n", argv[i - 1]);
                return NV_ENC_ERR_INVALID_PARAM;
            }
        }
        else if (stricmp(argv[i], "-deviceID") == 0)
        {
            if (++i >= argc || sscanf(argv[i], "%d", &encodeConfig->deviceID) != 1)
            {
                PRINTERR("invalid parameter for %s\n", argv[i - 1]);
                return NV_ENC_ERR_INVALID_PARAM;
            }
        }
        else if (stricmp(argv[i], "-yuv444") == 0)
        {
            if (++i >= argc || sscanf(argv[i], "%d", &encodeConfig->isYuv444) != 1)
            {
                fprintf(stderr, "invalid parameter for %s\n", argv[i - 1]);
                return NV_ENC_ERR_INVALID_PARAM;
            }
        }
        else if (stricmp(argv[i], "-qpDeltaMapFile") == 0)
        {
            if (++i >= argc)
            {
                PRINTERR("invalid parameter for %s\n", argv[i - 1]);
                return NV_ENC_ERR_INVALID_PARAM;
            }
            encodeConfig->qpDeltaMapFile = argv[i];
        }
        else if (stricmp(argv[i], "-help") == 0)
        {
            return NV_ENC_ERR_INVALID_PARAM;
        }
        else
        {
            PRINTERR("invalid parameter  %s\n", argv[i++]);
            return NV_ENC_ERR_INVALID_PARAM;
        }
    }

    return NV_ENC_SUCCESS;
}
