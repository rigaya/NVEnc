/*
 * Copyright 1993-2013 NVIDIA Corporation.  All rights reserved.
 *
 * NOTICE TO LICENSEE:
 *
 * This source code and/or documentation ("Licensed Deliverables") are
 * subject to NVIDIA intellectual property rights under U.S. and
 * international Copyright laws.
 *
 * These Licensed Deliverables contained herein is PROPRIETARY and
 * CONFIDENTIAL to NVIDIA and is being provided under the terms and
 * conditions of a form of NVIDIA software license agreement by and
 * between NVIDIA and Licensee ("License Agreement") or electronically
 * accepted by Licensee.  Notwithstanding any terms or conditions to
 * the contrary in the License Agreement, reproduction or disclosure
 * of the Licensed Deliverables to any third party without the express
 * written consent of NVIDIA is prohibited.
 *
 * ALL NVIDIA DESIGN SPECIFICATIONS, REFERENCE BOARDS, FILES, DRAWINGS,
 * DIAGNOSTICS, LISTS, AND OTHER DOCUMENTS (TOGETHER AND SEPARATELY,
 * “MATERIALS”) ARE BEING PROVIDED “AS IS.” WITHOUT EXPRESS OR IMPLIED
 * WARRANTY OF ANY KIND.  NVIDIA DISCLAIMS ALL WARRANTIES WITH REGARD
 * TO THESE LICENSED DELIVERABLES, INCLUDING ALL IMPLIED WARRANTIES OF
 * MERCHANTABILITY, NONINFRINGEMENT, AND FITNESS FOR A PARTICULAR PURPOSE.
 * NOTWITHSTANDING ANY TERMS OR CONDITIONS TO THE CONTRARY IN THE LICENSE
 * AGREEMENT, IN NO EVENT SHALL NVIDIA BE LIABLE FOR ANY SPECIAL, INDIRECT,
 * INCIDENTAL, OR CONSEQUENTIAL DAMAGES, OR ANY DAMAGES WHATSOEVER RESULTING
 * FROM LOSS OF USE, DATA OR PROFITS, WHETHER IN AN ACTION OF CONTRACT,
 * NEGLIGENCE OR OTHER TORTIOUS ACTION, ARISING OUT OF OR IN CONNECTION
 * WITH THE USE OR PERFORMANCE OF THESE LICENSED DELIVERABLES.
 *
 * Information furnished is believed to be accurate and reliable. However,
 * NVIDIA assumes no responsibility for the consequences of use of such
 * information nor for any infringement of patents or other rights of
 * third parties, which may result from its use.  No License is granted
 * by implication or otherwise under any patent or patent rights of NVIDIA
 * Corporation.  Specifications mentioned in the software are subject to
 * change without notice. This publication supersedes and replaces all
 * other information previously supplied.
 *
 * NVIDIA Corporation products are not authorized for use as critical
 * components in life support devices or systems without express written
 * approval of NVIDIA Corporation.
 *
 * U.S. Government End Users.  These Licensed Deliverables are a
 * "commercial item" as that term is defined at 48 C.F.R. 2.101 (OCT
 * 1995), consisting of "commercial computer software" and "commercial
 * computer software documentation" as such terms are used in 48
 * C.F.R. 12.212 (SEPT 1995) and is provided to the U.S. Government
 * only as a commercial end item.  Consistent with 48 C.F.R.12.212 and
 * 48 C.F.R. 227.7202-1 through 227.7202-4 (JUNE 1995), all
 * U.S. Government End Users acquire the Licensed Deliverables with
 * only those rights set forth herein.
 *
 * Any use of the Licensed Deliverables in individual and commercial
 * software must include, in the user documentation and internal
 * comments to the code, the above Disclaimer and U.S. Government End
 * Users Notice.
 */

/**
* \file CNVEncoder.h
* \brief CNVEncoder is the Class interface for the Hardware Encoder (NV Encode API)
* \date 2011-2013
*  This file contains the CNvEncoder class declaration and data structures
*/

#pragma once

#if defined __linux || defined __APPLE__ || defined __MACOSX
#ifndef NV_UNIX
#define NV_UNIX
#endif
#endif

#if defined(WIN32) || defined(_WIN32) || defined(WIN64)
#ifndef NV_WINDOWS
#define NV_WINDOWS
#endif
#endif

#ifndef _CRT_SECURE_NO_DEPRECATE
#define _CRT_SECURE_NO_DEPRECATE
#endif

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <assert.h>

#if defined (NV_WINDOWS)
#include <windows.h>
#if !defined (_NO_D3D)
#include <d3d9.h>
#include <d3d10_1.h>
#include <d3d11.h>
#endif
#endif

#if defined (NV_UNIX)
#include <stdio.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <errno.h>
#include <string.h>
#include <unistd.h>

#include <dlfcn.h>
#include <pthread.h>
#include "threads/NvPthreadABI.h"
#endif

#include "include/NvTypes.h"

#include "threads/NvThreadingClasses.h"
#include "nvEncodeAPI.h"
#include "defines.h"                    // common headers and definitions
#include <cuda.h>
#include <include/nvFileIO.h>           // helper functions for large file access
#include <include/helper_timer.h>       // helper functions for handling timing

#define MAX_ENCODERS 16
#define MAX_RECONFIGURATION 10

#ifndef max
#define max(a,b) (a > b ? a : b)
#endif

#define MAX_INPUT_QUEUE  32
#define MAX_OUTPUT_QUEUE 32
#define SET_VER(configStruct, type) {configStruct.version = type##_VER;}

// {00000000-0000-0000-0000-000000000000}
static const GUID  NV_ENC_H264_PROFILE_INVALID_GUID =
{ 0x0000000, 0x0000, 0x0000, { 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00 } };

#if defined (NV_WINDOWS)
#define NVENCAPI __stdcall
#elif defined (NV_UNIX)
#define NVENCAPI
typedef void *HINSTANCE;
typedef S32  HRESULT;
typedef void *HANDLE;

#define FILE_CURRENT             SEEK_CUR
#define FILE_BEGIN               SEEK_SET
#define INVALID_SET_FILE_POINTER (-1)
#define S_OK                     (0)
#define E_FAIL                   (-1)
#endif

typedef struct
{
    int param;
    char name[256];
} param_desc;

const param_desc framefieldmode_names[] =
{
    { 0,                                    "Invalid Frame/Field Mode" },
    { NV_ENC_PARAMS_FRAME_FIELD_MODE_FRAME, "Frame Mode"               },
    { NV_ENC_PARAMS_FRAME_FIELD_MODE_FIELD, "Frame Mode"               },
    { NV_ENC_PARAMS_FRAME_FIELD_MODE_MBAFF, "MB adaptive frame/field"  }
};

const param_desc ratecontrol_names[] =
{
    { NV_ENC_PARAMS_RC_CONSTQP,     "Constant QP Mode"                        },
    { NV_ENC_PARAMS_RC_VBR,         "VBR (Variable Bitrate)"                  },
    { NV_ENC_PARAMS_RC_CBR,         "CBR (Constant Bitrate)"                  },
    { 3,                            "Invalid Rate Control Mode"               },
    { NV_ENC_PARAMS_RC_VBR_MINQP,   "VNR_MINQP (Variable Bitrate with MinQP)" },
    { 5,                            "Invalid Rate Control Mode"               },
    { 6,                            "Invalid Rate Control Mode"               },
    { 7,                            "Invalid Rate Control Mode"               },
    { NV_ENC_PARAMS_RC_2_PASS_QUALITY,          "Two-Pass Prefered Quality Bitrate"     },
    { NV_ENC_PARAMS_RC_2_PASS_FRAMESIZE_CAP,    "Two-Pass Prefered Frame Size Bitrate"  },
    { NV_ENC_PARAMS_RC_CBR2,                    "Two-Pass (Constant Bitrate)"           }
};

const param_desc encode_picstruct_names[] =
{
    { 0,                                    "0 = Invalid Picture Struct"                },
    { NV_ENC_PIC_STRUCT_FRAME,              "1 = Progressive Frame"                     },
    { NV_ENC_PIC_STRUCT_FIELD_TOP_BOTTOM,   "2 = Top Field interlaced frame"            },
    { NV_ENC_PIC_STRUCT_FIELD_BOTTOM_TOP,   "3 = Bottom Field first inerlaced frame"    },
};

/**
 *  * Input picture type
 *   */
const param_desc encode_picture_types[] =
{
    { NV_ENC_PIC_TYPE_P,             "0 = Forward predicted"                              },
    { NV_ENC_PIC_TYPE_B,             "1 = Bi-directionally predicted picture"             },
    { NV_ENC_PIC_TYPE_I,             "2 = Intra predicted picture"                        },
    { NV_ENC_PIC_TYPE_IDR,           "3 = IDR picture"                                    },
    { NV_ENC_PIC_TYPE_BI,            "4 = Bi-directionally predicted with only Intra MBs" },
    { NV_ENC_PIC_TYPE_SKIPPED,       "5 = Picture is skipped"                             },
    { NV_ENC_PIC_TYPE_INTRA_REFRESH, "6 = First picture in intra refresh cycle"           },
    { NV_ENC_PIC_TYPE_UNKNOWN,       "0xFF = Picture type unknown"                        }
};

/**
 *  * Input slice type
 *   */
const param_desc encode_slice_type[] =
{
    { NV_ENC_SLICE_TYPE_DEFAULT, "0 = Slice type is same as picture type" },
    { 1   ,                      "1 = Invalid slice type mode"            },
    { NV_ENC_SLICE_TYPE_I,       "2 = Intra predicted slice"              },
    { NV_ENC_SLICE_TYPE_UNKNOWN, "0xFF = Slice type unknown"              }
};

/**
 *  * Motion vector precisions
 *   */
const param_desc encode_precision_mv[] =
{
    { 0,                               "0 = Invalid encode MV precision" },
    { NV_ENC_MV_PRECISION_FULL_PEL,    "1 = Full-Pel    Motion Vector precision" },
    { NV_ENC_MV_PRECISION_HALF_PEL,    "2 = Half-Pel    Motion Vector precision" },
    { NV_ENC_MV_PRECISION_QUARTER_PEL, "3 = Quarter-Pel Motion Vector precision" },
};

typedef struct
{
    GUID id;
    char name[256];
    unsigned int  value;
} guid_desc;

enum
{
    NV_ENC_PRESET_DEFAULT                   =0,
    NV_ENC_PRESET_LOW_LATENCY_DEFAULT       =1,
    NV_ENC_PRESET_HP                        =2,
    NV_ENC_PRESET_HQ                        =3,
    NV_ENC_PRESET_BD                        =4,
    NV_ENC_PRESET_LOW_LATENCY_HQ            =5,
    NV_ENC_PRESET_LOW_LATENCY_HP            =6
};

const guid_desc codec_names[] =
{
    { NV_ENC_CODEC_H264_GUID, "Invalid Codec Setting" , 0},
    { NV_ENC_CODEC_H264_GUID, "Invalid Codec Setting" , 1},
    { NV_ENC_CODEC_H264_GUID, "Invalid Codec Setting" , 2},
    { NV_ENC_CODEC_H264_GUID, "Invalid Codec Setting" , 3},
    { NV_ENC_CODEC_H264_GUID, "H.264 Codec"           , 4}
};

const guid_desc codecprofile_names[] =
{
    { NV_ENC_H264_PROFILE_BASELINE_GUID, "H.264 Baseline", 66 },
    { NV_ENC_H264_PROFILE_MAIN_GUID,     "H.264 Main Profile", 77 },
    { NV_ENC_H264_PROFILE_HIGH_GUID,     "H.264 High Profile", 100 },
    { NV_ENC_H264_PROFILE_STEREO_GUID,   "H.264 Stereo Profile", 128 }
};

const guid_desc preset_names[] =
{
    { NV_ENC_PRESET_DEFAULT_GUID,                               "Default Preset",  0},
    { NV_ENC_PRESET_LOW_LATENCY_DEFAULT_GUID,                   "Low Latancy Default Preset", 1 },
    { NV_ENC_PRESET_HP_GUID,                                    "High Performance (HP) Preset", 2},
    { NV_ENC_PRESET_HQ_GUID,                                    "High Quality (HQ) Preset", 3 },
    { NV_ENC_PRESET_BD_GUID,                                    "Blue Ray Preset", 4 },
    { NV_ENC_PRESET_LOW_LATENCY_HQ_GUID,                        "Low Latancy High Quality (HQ) Preset", 5 },
    { NV_ENC_PRESET_LOW_LATENCY_HP_GUID,                        "Low Latancy High Performance (HP) Preset", 6 }

};

inline bool compareGUIDs(GUID guid1, GUID guid2)
{
    if (guid1.Data1    == guid2.Data1 &&
        guid1.Data2    == guid2.Data2 &&
        guid1.Data3    == guid2.Data3 &&
        guid1.Data4[0] == guid2.Data4[0] &&
        guid1.Data4[1] == guid2.Data4[1] &&
        guid1.Data4[2] == guid2.Data4[2] &&
        guid1.Data4[3] == guid2.Data4[3] &&
        guid1.Data4[4] == guid2.Data4[4] &&
        guid1.Data4[5] == guid2.Data4[5] &&
        guid1.Data4[6] == guid2.Data4[6] &&
        guid1.Data4[7] == guid2.Data4[7])
    {
        return true;
    }

    return false;
}

inline void printGUID(int i, GUID *id)
{
    printf("GUID[%d]: %08X-%04X-%04X-%08X", i, id->Data1, id->Data2, id->Data3, *((unsigned int *)id->Data4));
}

inline void printPresetGUIDName(GUID guid)
{
    int loopCnt = sizeof(preset_names)/ sizeof(guid_desc);

    for (int cnt = 0; cnt < loopCnt; cnt++)
    {
        if (compareGUIDs(preset_names[cnt].id, guid))
        {
            printf(" \"%s\"\n", preset_names[cnt].name);
        }
    }
}

inline void printProfileGUIDName(GUID guid)
{
    int loopCnt = sizeof(codecprofile_names)/ sizeof(guid_desc);

    for (int cnt = 0; cnt < loopCnt; cnt++)
    {
        if (compareGUIDs(codecprofile_names[cnt].id, guid))
        {
            printf(" \"%s\"\n", codecprofile_names[cnt].name);
        }
    }
}

typedef enum _NvEncodeCompressionStd
{
    NV_ENC_Unknown=-1,
    NV_ENC_H264=4      // 14496-10
} NvEncodeCompressionStd;

typedef enum _NvEncodeInterfaceType
{
    NV_ENC_DX9=0,
    NV_ENC_DX11=1,
    NV_ENC_CUDA=2, // On Linux, CUDA is the only NVENC interface available
    NV_ENC_DX10=3,
} NvEncodeInterfaceType;

const param_desc nvenc_interface_names[] =
{
    { NV_ENC_DX9,   "DirectX9"  },
    { NV_ENC_DX11,  "DirectX11" },
    { NV_ENC_CUDA,  "CUDA"      },
    { NV_ENC_DX10,  "DirectX10" }
};

struct EncodeConfig
{
    NvEncodeCompressionStd      codec;
    unsigned int                profile;
    unsigned int                level;
    unsigned int                width;
    unsigned int                height;
    unsigned int                maxWidth;
    unsigned int                maxHeight;
    unsigned int                frameRateNum;
    unsigned int                frameRateDen;
    unsigned int                avgBitRate;
    unsigned int                peakBitRate;
    unsigned int                gopLength;
    unsigned int                enableInitialRCQP;
    NV_ENC_QP                   initialRCQP;
    unsigned int                numBFrames;
    unsigned int                fieldEncoding;
    unsigned int                bottomFieldFrist;
    unsigned int                rateControl; // 0= QP, 1= CBR. 2= VBR
    int                         numSlices;
    unsigned int                vbvBufferSize;
    unsigned int                vbvInitialDelay;
    NV_ENC_MV_PRECISION         mvPrecision;
    unsigned int                enablePTD;
    int                         preset;
    int                         syncMode;
    NvEncodeInterfaceType       interfaceType;
    unsigned int                useMappedResources;
    char                        InputClip[256];
    FILE                        *fOutput;
    unsigned int                endFrame;
};

struct EncodeInputSurfaceInfo
{
    unsigned int      dwWidth;
    unsigned int      dwHeight;
    unsigned int      dwLumaOffset;
    unsigned int      dwChromaOffset;
    void              *hInputSurface;
    unsigned int      lockedPitch;
    NV_ENC_BUFFER_FORMAT bufferFmt;
    void              *pExtAlloc;
    unsigned char     *pExtAllocHost;
    unsigned int      dwCuPitch;
    NV_ENC_INPUT_RESOURCE_TYPE type;
    void              *hRegisteredHandle;
};

struct EncodeOutputBuffer
{
    unsigned int     dwSize;
    unsigned int     dwBitstreamDataSize;
    void             *hBitstreamBuffer;
    HANDLE           hOutputEvent;
    bool             bWaitOnEvent;
    void             *pBitstreamBufferPtr;
    bool             bEOSFlag;
    bool             bReconfiguredflag;
};

struct EncoderThreadData
{
    EncodeOutputBuffer      *pOutputBfr;
    EncodeInputSurfaceInfo  *pInputBfr;
};

#define DYN_DOWNSCALE 1
#define DYN_UPSCALE   2

struct EncodeFrameConfig
{
    unsigned char *yuv[3];
    unsigned int stride[3];
    unsigned int width;
    unsigned int height;
    NV_ENC_PIC_STRUCT picStruct;
    bool         bReconfigured;
};

struct FrameThreadData
{
    HANDLE        hInputYUVFile;
    unsigned int  dwFileWidth;
    unsigned int  dwFileHeight;
    unsigned int  dwSurfWidth;
    unsigned int  dwSurfHeight;
    unsigned int  dwFrmIndex;
    void          *pYUVInputFrame;
};

struct EncoderGPUInfo
{
    char gpu_name[256];
    unsigned int device;
};
struct configs_s
{
    const char *str;
    int type;
    int offset;
};
class CNvEncoderThread;

// The main Encoder Class interface
class CNvEncoder
{
    public:
        CNvEncoder();
        virtual ~CNvEncoder();
    protected:
        void                                                *m_hEncoder;
#if defined (NV_WINDOWS) && !defined(_NO_D3D)
        IDirect3D9                                          *m_pD3D;
        IDirect3DDevice9                                    *m_pD3D9Device;
        ID3D10Device                                        *m_pD3D10Device;
        ID3D11Device                                        *m_pD3D11Device;
#endif
        CUcontext                                            m_cuContext;
        unsigned int                                         m_dwEncodeGUIDCount;
        GUID                                                 m_stEncodeGUID;
        unsigned int                                         m_dwCodecProfileGUIDCount;
        GUID                                                 m_stCodecProfileGUID;
        GUID                                                 m_stPresetGUID;
        unsigned int                                         m_encodersAvailable;
        unsigned int                                         m_dwInputFmtCount;
        NV_ENC_BUFFER_FORMAT                                *m_pAvailableSurfaceFmts;
        NV_ENC_BUFFER_FORMAT                                 m_dwInputFormat;
        NV_ENC_INITIALIZE_PARAMS                             m_stInitEncParams;
        NV_ENC_RECONFIGURE_PARAMS                            m_stReInitEncParams;
        NV_ENC_CONFIG                                        m_stEncodeConfig;
        NV_ENC_PRESET_CONFIG                                 m_stPresetConfig;
        NV_ENC_PIC_PARAMS                                    m_stEncodePicParams;
        bool                                                 m_bEncoderInitialized;
        EncodeConfig                                         m_stEncoderInput[MAX_RECONFIGURATION];
        EncodeInputSurfaceInfo                               m_stInputSurface[MAX_INPUT_QUEUE];
        EncodeOutputBuffer                                   m_stBitstreamBuffer[MAX_OUTPUT_QUEUE];
        CNvQueue<EncodeInputSurfaceInfo *, MAX_INPUT_QUEUE>   m_stInputSurfQueue;
        CNvQueue<EncodeOutputBuffer *, MAX_OUTPUT_QUEUE>      m_stOutputSurfQueue;
        unsigned int                                         m_dwMaxSurfCount;
        unsigned int                                         m_dwCurrentSurfIdx;
        unsigned int                                         m_dwFrameWidth;
        unsigned int                                         m_dwFrameHeight;
        unsigned int                                         m_uMaxHeight;
        unsigned int                                         m_uMaxWidth;

        unsigned int                                         m_uRefCount;
        configs_s                                            m_configs[50];

        FILE                                                *m_fOutput;
        FILE                                                *m_fInput;
        CNvEncoderThread                                    *m_pEncoderThread;
        unsigned char                                       *m_pYUV[3];
        bool                                                 m_bAsyncModeEncoding; // only avialable on Windows Platforms
        unsigned char                                        m_pUserData[128];
        NV_ENC_SEQUENCE_PARAM_PAYLOAD                        m_spspps;
        EncodeOutputBuffer                                   m_stEOSOutputBfr;
        CNvQueue<EncoderThreadData, MAX_OUTPUT_QUEUE>        m_pEncodeFrameQueue;
        unsigned int                                         m_dwReConfigIdx;
        unsigned int                                         m_dwNumReConfig;
        virtual bool                                         ParseConfigFile(const char *file);
        virtual void                                         ParseCommandlineInput(int argc, const char *argv[]);
        virtual bool                                         ParseConfigString(const char *str);
        virtual bool                                         ParseReConfigFile(char *reConfigFile);
        virtual bool                                         ParseReConfigString(const char *str);
        virtual HRESULT                                      LoadCurrentFrame(unsigned char *yuvInput[3] , HANDLE hInputYUVFile,unsigned int dwFrmIndex);
        virtual void                                         DisplayEncodingParams(EncoderAppParams pEncodeAppParams, int numConfigured);
        virtual HRESULT                                      OpenEncodeSession(int argc, const char *argv[],unsigned int deviceID = 0);
        virtual HRESULT                                      LoadCurrentFrame(unsigned char *yuvInput[3] , HANDLE hInputYUVFile, unsigned int dwFrmIndex,
                                                                             unsigned int dwFileWidth, unsigned int dwFileHeight, unsigned int dwSurfWidth, unsigned int dwSurfHeight,
                                                                             bool bFieldPic, bool bTopField, int FrameQueueSize, int chromaFormatIdc = 1) = 0;
        virtual void                                         PreloadFrames(unsigned int frameNumber, unsigned int numFramesToEncode, unsigned int gpuid, HANDLE hInput) = 0;
		virtual int                                          CalculateFramesFromInput(HANDLE hInputFile, const char *filename, int width, int height) = 0;
        virtual HRESULT                                      InitializeEncoder() = 0;
        virtual void                                         PreInitializeEncoder() = 0;
        virtual HRESULT                                      ReconfigureEncoder(EncodeConfig EncoderReConfig)   = 0;
        virtual HRESULT                                      EncodeFrame(EncodeFrameConfig *pEncodeFrame, bool bFlush=false) = 0;
        virtual HRESULT                                      DestroyEncoder() = 0;

    public:
        virtual int                                          EncoderMain(EncoderGPUInfo encoderInfo, EncoderAppParams appParams, int argc,const char *argv[]) =0 ;
        virtual HRESULT                                      CopyBitstreamData(EncoderThreadData stThreadData);
        virtual HRESULT                                      CopyFrameData(FrameThreadData stFrameData);
        virtual HRESULT                                      QueryEncodeCaps(NV_ENC_CAPS caps_type, int *p_nCapsVal);

    protected:
#if defined (NV_WINDOWS) && !defined (_NO_D3D)// Windows uses Direct3D or CUDA to access NVENC
        HRESULT                                              InitD3D9(unsigned int deviceID = 0);
        HRESULT                                              InitD3D10(unsigned int deviceID = 0);
        HRESULT                                              InitD3D11(unsigned int deviceID = 0);
#endif
        HRESULT                                              InitCuda(unsigned int deviceID = 0);
        HRESULT                                              AllocateIOBuffers(unsigned int dwInputWidth, unsigned int dwInputHeight, unsigned int maxFrmCnt);
        HRESULT                                              ReleaseIOBuffers();

        unsigned char                                       *LockInputBuffer(void *hInputSurface, unsigned int *pLockedPitch);
        HRESULT                                              UnlockInputBuffer(void *hInputSurface);
        unsigned int                                         GetCodecType(GUID encodeGUID);
        unsigned int                                         GetCodecProfile(GUID encodeGUID);
        GUID                                                 GetCodecProfileGuid(unsigned int profile);
        HRESULT                                              GetPresetConfig(int iPresetIdx);

        HRESULT                                              FlushEncoder();
        HRESULT                                              ReleaseEncoderResources();
        HRESULT                                              WaitForCompletion();

        NV_ENC_REGISTER_RESOURCE                             m_RegisteredResource;

    public:
        NV_ENCODE_API_FUNCTION_LIST                         *m_pEncodeAPI;
        HINSTANCE                                            m_hinstLib;
        bool                                                 m_bEncodeAPIFound;
        StopWatchInterface                                  *m_timer;
};

class CNvEncoderThread: public CNvThread
{
    public:
        CNvEncoderThread(CNvEncoder *pOwner, U32 dwMaxQueuedSamples)
            :   CNvThread("Encoder Output Thread")
            ,   m_pOwner(pOwner)
            ,   m_dwMaxQueuedSamples(dwMaxQueuedSamples)
        {
            // Empty constructor
        }

        // Overridden virtual functions
        virtual bool ThreadFunc();
        // virtual bool ThreadFini();

        bool QueueSample(EncoderThreadData &sThreadData);
        int GetCurrentQCount()
        {
            return m_pEncoderQueue.GetCount();
        }
        bool IsIdle()
        {
            return m_pEncoderQueue.GetCount() == 0;
        }
        bool IsQueueFull()
        {
            return m_pEncoderQueue.GetCount() >= m_dwMaxQueuedSamples;
        }

    protected:
        CNvEncoder *const m_pOwner;
        CNvQueue<EncoderThreadData, MAX_OUTPUT_QUEUE> m_pEncoderQueue;
        U32 m_dwMaxQueuedSamples;
};

#define QUERY_PRINT_CAPS(pEnc, CAPS, VAL) pEnc->QueryEncodeCaps(CAPS, &VAL); printf("Query %s = %d\n", #CAPS, VAL);

void queryAllEncoderCaps(CNvEncoder *pEncoder);

// NVEncodeAPI entry point
#if defined(NV_WINDOWS)
typedef NVENCSTATUS(__stdcall *MYPROC)(NV_ENCODE_API_FUNCTION_LIST *);
#else
typedef NVENCSTATUS(*MYPROC)(NV_ENCODE_API_FUNCTION_LIST *);
#endif

