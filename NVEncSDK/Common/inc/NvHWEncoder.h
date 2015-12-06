/*
 * Copyright 1993-2015 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 *
 */

#include <stdlib.h>
#include <stdio.h>
#include <assert.h>

#include <cuda.h>

#include "nvEncodeAPI.h"
#include "nvUtils.h"

#define SET_VER(configStruct, type) {configStruct.version = type##_VER;}

#if defined (NV_WINDOWS)
    #include "d3d9.h"
    #define NVENCAPI __stdcall
    #pragma warning(disable : 4996)
#elif defined (NV_UNIX)
    #include <dlfcn.h>
    #include <string.h>
    #define NVENCAPI
#endif

#define DEFAULT_I_QFACTOR -0.8f
#define DEFAULT_B_QFACTOR 1.25f
#define DEFAULT_I_QOFFSET 0.f
#define DEFAULT_B_QOFFSET 1.25f

typedef struct _EncodeConfig
{
    int              width;
    int              height;
    int              maxWidth;
    int              maxHeight;
    int              fps;
    int              bitrate;
    int              vbvMaxBitrate;
    int              vbvSize;
    int              rcMode;
    int              qp;
    float            i_quant_factor;
    float            b_quant_factor;
    float            i_quant_offset;
    float            b_quant_offset;
    GUID             presetGUID;
    FILE            *fOutput;
    int              codec;
    int              invalidateRefFramesEnableFlag;
    int              intraRefreshEnableFlag;
    int              intraRefreshPeriod;
    int              intraRefreshDuration;
    int              deviceType;
    int              startFrameIdx;
    int              endFrameIdx;
    int              gopLength;
    int              numB;
    int              pictureStruct;
    int              deviceID;
    int              isYuv444;
    char            *qpDeltaMapFile;
    char* inputFileName;
    char* outputFileName;
    char* encoderPreset;
    char* inputFilePath;
    char *encCmdFileName;
    int  enableMEOnly;
    int  preloadedFrameCount;
}EncodeConfig;

typedef struct _EncodeInputBuffer
{
    unsigned int      dwWidth;
    unsigned int      dwHeight;
#if defined (NV_WINDOWS)
    IDirect3DSurface9 *pNV12Surface;
#endif
    CUdeviceptr       pNV12devPtr;
    uint32_t          uNV12Stride;
    CUdeviceptr       pNV12TempdevPtr;
    uint32_t          uNV12TempStride;
    void*             nvRegisteredResource;
    NV_ENC_INPUT_PTR  hInputSurface;
    NV_ENC_BUFFER_FORMAT bufferFmt;
}EncodeInputBuffer;

typedef struct _EncodeOutputBuffer
{
    unsigned int          dwBitstreamBufferSize;
    NV_ENC_OUTPUT_PTR     hBitstreamBuffer;
    HANDLE                hOutputEvent;
    bool                  bWaitOnEvent;
    bool                  bEOSFlag;
}EncodeOutputBuffer;

typedef struct _EncodeBuffer
{
    EncodeOutputBuffer      stOutputBfr;
    EncodeInputBuffer       stInputBfr;
}EncodeBuffer;

typedef struct _NvEncPictureCommand
{
    bool bResolutionChangePending;
    bool bBitrateChangePending;
    bool bForceIDR;
    bool bForceIntraRefresh;
    bool bInvalidateRefFrames;

    uint32_t newWidth;
    uint32_t newHeight;

    uint32_t newBitrate;
    uint32_t newVBVSize;

    uint32_t  intraRefreshDuration;

    uint32_t  numRefFramesToInvalidate;
    uint32_t  refFrameNumbers[16];
}NvEncPictureCommand;

enum
{
    NV_ENC_H264 = 0,
    NV_ENC_HEVC = 1,
};

struct MEOnlyConfig
{
    unsigned char *yuv[2][3];
    unsigned int stride[3];
    unsigned int width;
    unsigned int height;
    unsigned int inputFrameIndex;
    unsigned int referenceFrameIndex;
};

class CNvHWEncoder
{
public:
    uint32_t                                             m_EncodeIdx;
    FILE                                                *m_fOutput;
    uint32_t                                             m_uMaxWidth;
    uint32_t                                             m_uMaxHeight;
    uint32_t                                             m_uCurWidth;
    uint32_t                                             m_uCurHeight;

protected:
    bool                                                 m_bEncoderInitialized;
    GUID                                                 codecGUID;

    NV_ENCODE_API_FUNCTION_LIST*                         m_pEncodeAPI;
    HINSTANCE                                            m_hinstLib;
    void                                                *m_hEncoder;
    NV_ENC_INITIALIZE_PARAMS                             m_stCreateEncodeParams;
    NV_ENC_CONFIG                                        m_stEncodeConfig;

public:
    NVENCSTATUS NvEncOpenEncodeSession(void* device, uint32_t deviceType);
    NVENCSTATUS NvEncGetEncodeGUIDCount(uint32_t* encodeGUIDCount);
    NVENCSTATUS NvEncGetEncodeProfileGUIDCount(GUID encodeGUID, uint32_t* encodeProfileGUIDCount);
    NVENCSTATUS NvEncGetEncodeProfileGUIDs(GUID encodeGUID, GUID* profileGUIDs, uint32_t guidArraySize, uint32_t* GUIDCount);
    NVENCSTATUS NvEncGetEncodeGUIDs(GUID* GUIDs, uint32_t guidArraySize, uint32_t* GUIDCount);
    NVENCSTATUS NvEncGetInputFormatCount(GUID encodeGUID, uint32_t* inputFmtCount);
    NVENCSTATUS NvEncGetInputFormats(GUID encodeGUID, NV_ENC_BUFFER_FORMAT* inputFmts, uint32_t inputFmtArraySize, uint32_t* inputFmtCount);
    NVENCSTATUS NvEncGetEncodeCaps(GUID encodeGUID, NV_ENC_CAPS_PARAM* capsParam, int* capsVal);
    NVENCSTATUS NvEncGetEncodePresetCount(GUID encodeGUID, uint32_t* encodePresetGUIDCount);
    NVENCSTATUS NvEncGetEncodePresetGUIDs(GUID encodeGUID, GUID* presetGUIDs, uint32_t guidArraySize, uint32_t* encodePresetGUIDCount);
    NVENCSTATUS NvEncGetEncodePresetConfig(GUID encodeGUID, GUID  presetGUID, NV_ENC_PRESET_CONFIG* presetConfig);
    NVENCSTATUS NvEncCreateInputBuffer(uint32_t width, uint32_t height, void** inputBuffer, uint32_t isYuv444);
    NVENCSTATUS NvEncDestroyInputBuffer(NV_ENC_INPUT_PTR inputBuffer);
    NVENCSTATUS NvEncCreateBitstreamBuffer(uint32_t size, void** bitstreamBuffer);
    NVENCSTATUS NvEncDestroyBitstreamBuffer(NV_ENC_OUTPUT_PTR bitstreamBuffer);
    NVENCSTATUS NvEncCreateMVBuffer(uint32_t size, void** bitstreamBuffer);
    NVENCSTATUS NvEncDestroyMVBuffer(NV_ENC_OUTPUT_PTR bitstreamBuffer);
    NVENCSTATUS NvRunMotionEstimationOnly(EncodeBuffer *pEncodeBuffer[2], MEOnlyConfig *pMEOnly);
    NVENCSTATUS NvEncLockBitstream(NV_ENC_LOCK_BITSTREAM* lockBitstreamBufferParams);
    NVENCSTATUS NvEncUnlockBitstream(NV_ENC_OUTPUT_PTR bitstreamBuffer);
    NVENCSTATUS NvEncLockInputBuffer(void* inputBuffer, void** bufferDataPtr, uint32_t* pitch);
    NVENCSTATUS NvEncUnlockInputBuffer(NV_ENC_INPUT_PTR inputBuffer);
    NVENCSTATUS NvEncGetEncodeStats(NV_ENC_STAT* encodeStats);
    NVENCSTATUS NvEncGetSequenceParams(NV_ENC_SEQUENCE_PARAM_PAYLOAD* sequenceParamPayload);
    NVENCSTATUS NvEncRegisterAsyncEvent(void** completionEvent);
    NVENCSTATUS NvEncUnregisterAsyncEvent(void* completionEvent);
    NVENCSTATUS NvEncMapInputResource(void* registeredResource, void** mappedResource);
    NVENCSTATUS NvEncUnmapInputResource(NV_ENC_INPUT_PTR mappedInputBuffer);
    NVENCSTATUS NvEncDestroyEncoder();
    NVENCSTATUS NvEncInvalidateRefFrames(const NvEncPictureCommand *pEncPicCommand);
    NVENCSTATUS NvEncOpenEncodeSessionEx(void* device, NV_ENC_DEVICE_TYPE deviceType);
    NVENCSTATUS NvEncRegisterResource(NV_ENC_INPUT_RESOURCE_TYPE resourceType, void* resourceToRegister, uint32_t width, uint32_t height, uint32_t pitch, void** registeredResource);
    NVENCSTATUS NvEncUnregisterResource(NV_ENC_REGISTERED_PTR registeredRes);
    NVENCSTATUS NvEncReconfigureEncoder(const NvEncPictureCommand *pEncPicCommand);
    NVENCSTATUS NvEncFlushEncoderQueue(void *hEOSEvent);

    CNvHWEncoder();
    virtual ~CNvHWEncoder();
    NVENCSTATUS                                          Initialize(void* device, NV_ENC_DEVICE_TYPE deviceType);
    NVENCSTATUS                                          Deinitialize();
    NVENCSTATUS                                          NvEncEncodeFrame(EncodeBuffer *pEncodeBuffer, NvEncPictureCommand *encPicCommand,
                                                                          uint32_t width, uint32_t height,
                                                                          NV_ENC_PIC_STRUCT ePicStruct = NV_ENC_PIC_STRUCT_FRAME,
                                                                          int8_t *qpDeltaMapArray = NULL, uint32_t qpDeltaMapArraySize = 0);
    NVENCSTATUS                                          CreateEncoder(const EncodeConfig *pEncCfg);
    GUID                                                 GetPresetGUID(char* encoderPreset, int codec);
    NVENCSTATUS                                          ProcessOutput(const EncodeBuffer *pEncodeBuffer);
    NVENCSTATUS                                          FlushEncoder();
    NVENCSTATUS                                          ValidateEncodeGUID(GUID inputCodecGuid);
    NVENCSTATUS                                          ValidatePresetGUID(GUID presetCodecGuid, GUID inputCodecGuid);
    static NVENCSTATUS                                   ParseArguments(EncodeConfig *encodeConfig, int argc, char *argv[]);
};

typedef NVENCSTATUS (NVENCAPI *MYPROC)(NV_ENCODE_API_FUNCTION_LIST*); 
