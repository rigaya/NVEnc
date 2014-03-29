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

#include <include/videoFormats.h>
#include <CNVEncoderH264.h>
#include <xcodeutil.h>
#include <include/helper_string.h>

#ifndef INFINITE
#define INFINITE UINT_MAX
#endif

#pragma warning (disable:4311)

/**
* \file CNVEncoderH264.cpp
* \brief CNVEncoderH264 is the Class interface for the Hardware Encoder (NV Encode API H.264)
* \date 2011
*  This file contains the CNvEncoderH264 class declaration and data structures
*/


// H264 Encoder
CNvEncoderH264::CNvEncoderH264()
{
    m_uMaxWidth       = 0;
    m_uMaxHeight      = 0;
    m_dwFrameNumInGOP = 0;
}

CNvEncoderH264::~CNvEncoderH264()
{
}

void CNvEncoderH264::InitDefault()
{
    m_stEncoderInput[0].codec                    = NV_ENC_H264;
    m_stEncoderInput[0].rateControl              = NV_ENC_PARAMS_RC_CBR; // Constant Bitrate
    m_stEncoderInput[0].avgBitRate               = 6000000;
    m_stEncoderInput[0].gopLength                = 30;
    m_stEncoderInput[0].frameRateNum             = 30000;
    m_stEncoderInput[0].frameRateDen             = 1001;
    m_stEncoderInput[0].maxWidth                 = 1920;
    m_stEncoderInput[0].maxHeight                = 1080;
    m_stEncoderInput[0].width                    = 1920;
    m_stEncoderInput[0].height                   = 1080;
    m_stEncoderInput[0].preset                   = NV_ENC_PRESET_HQ; // set to high quazlity mode
    m_stEncoderInput[0].profile                  = 100; // 66=baseline, 77=main, 100=high, 128=stereo (need to also set stereo3d bits below)
    m_stEncoderInput[0].mvPrecision              = NV_ENC_MV_PRECISION_QUARTER_PEL;
    m_stEncoderInput[0].numSlices                = 1;   // 1 slice per frame
    m_stEncoderInput[0].level                    = 40;
    m_stEncoderInput[0].enablePTD                = 1;
    m_stEncoderInput[0].useMappedResources       = 1;
    m_stEncoderInput[0].interfaceType            = NV_ENC_CUDA;  // Windows R304 (DX9 only), Windows R310 (DX10/DX11/CUDA), Linux R310 (CUDA only)
    m_stEncoderInput[0].syncMode                 = 1;
    m_stEncoderInput[0].endFrame                 = 0;
}

int CNvEncoderH264::CalculateFramesFromInput(HANDLE hInputFile, const char *filename, int width, int height)
{
    nvGetFileSize(hInputFile, NULL);
    // Let's compute how many YUV input frames (based on the file size and resolution)
    int numberOfFrames = 0;
    {
#if defined(NV_WINDOWS)
        LARGE_INTEGER fileSize;
        fileSize.LowPart = GetFileSize(hInputFile, (LPDWORD)&fileSize.HighPart);
        numberOfFrames = (unsigned int)(fileSize.QuadPart/(LONGLONG)(width*height*3/2));
#else
        fseek((FILE *)hInputFile, 0, SEEK_END);
        FILE_SIZE fileSize = (FILE_SIZE)ftell((FILE *)hInputFile);
        numberOfFrames = (unsigned int)(fileSize/(LONGLONG)(width*height*3/2));
#endif
    }
    NvPrintf("[ Source Input File ] = \"%s\"\n", filename);
    NvPrintf("[ # of Input Frames ] = %d\n", numberOfFrames);

    return numberOfFrames;
}

HRESULT CNvEncoderH264::LoadCurrentFrame(unsigned char *yuvInput[3] , HANDLE hInputYUVFile, unsigned int dwFrmIndex,
                         unsigned int dwFileWidth, unsigned int dwFileHeight, unsigned int dwSurfWidth, unsigned int dwSurfHeight,
                         bool bFieldPic, bool bTopField, int FrameQueueSize, int chromaFormatIdc)
{
    U64 fileOffset;
    U32 numBytesRead = 0;
    U32 result;
    unsigned int dwFrameStrideY    = (dwFrmIndex % FrameQueueSize) *  dwFileWidth * dwFileHeight;
    unsigned int dwFrameStrideCbCr = (dwFrmIndex % FrameQueueSize) * (dwFileWidth * dwFileHeight)/4;
    unsigned int dwInFrameSize = dwFileWidth*dwFileHeight + (dwFileWidth*dwFileHeight)/2;

    if (chromaFormatIdc == 3)
    {
        dwInFrameSize = dwFileWidth * dwFileHeight * 3;
    }
    else
    {
        dwInFrameSize = dwFileWidth * dwFileHeight + (dwFileWidth * dwFileHeight) / 2;
    }

    fileOffset = ((U64)dwInFrameSize * (U64)dwFrmIndex);
    result = nvSetFilePointer64(hInputYUVFile, fileOffset, NULL, FILE_BEGIN);

    if (result == INVALID_SET_FILE_POINTER)
    {
        return E_FAIL;
    }

    if (chromaFormatIdc == 3)
    {
        for (unsigned int i = 0 ; i < dwFileHeight; i++)
        {
            nvReadFile(hInputYUVFile, yuvInput[0] + dwFrameStrideY    + i * dwSurfWidth, dwFileWidth, &numBytesRead, NULL);
        }

        // read U
        for (unsigned int i = 0 ; i < dwFileHeight; i++)
        {
            nvReadFile(hInputYUVFile, yuvInput[1] + dwFrameStrideCbCr + i * dwSurfWidth, dwFileWidth, &numBytesRead, NULL);
        }

        // read V
        for (unsigned int i = 0 ; i < dwFileHeight; i++)
        {
            nvReadFile(hInputYUVFile, yuvInput[2] + dwFrameStrideCbCr + i * dwSurfWidth, dwFileWidth, &numBytesRead, NULL);
        }
    }
    else if (bFieldPic)
    {
        if (!bTopField)
        {
            // skip the first line
            fileOffset  = (U64)dwFileWidth;
            result = nvSetFilePointer64(hInputYUVFile, fileOffset, NULL, FILE_CURRENT);

            if (result == INVALID_SET_FILE_POINTER)
            {
                return E_FAIL;
            }
        }

        // read Y
        for (unsigned int i = 0 ; i < dwFileHeight/2; i++)
        {
            nvReadFile(hInputYUVFile, yuvInput[0] + dwFrameStrideY + i*dwSurfWidth, dwFileWidth, &numBytesRead, NULL);
            // skip the next line
            fileOffset  = (U64)dwFileWidth;
            result = nvSetFilePointer64(hInputYUVFile, fileOffset, NULL, FILE_CURRENT);

            if (result == INVALID_SET_FILE_POINTER)
            {
                return E_FAIL;
            }
        }

        // read U,V
        for (int cbcr = 0; cbcr < 2; cbcr++)
        {
            //put file pointer at the beginning of chroma
            fileOffset  = ((U64)dwInFrameSize*dwFrmIndex + (U64)dwFileWidth*dwFileHeight + (U64)cbcr*((dwFileWidth*dwFileHeight)/4));
            result = nvSetFilePointer64(hInputYUVFile, fileOffset, NULL, FILE_BEGIN);

            if (result == INVALID_SET_FILE_POINTER)
            {
                return E_FAIL;
            }

            if (!bTopField)
            {
                fileOffset  = (U64)dwFileWidth/2;
                result = nvSetFilePointer64(hInputYUVFile, fileOffset, NULL, FILE_CURRENT);

                if (result == INVALID_SET_FILE_POINTER)
                {
                    return E_FAIL;
                }
            }

            for (unsigned int i = 0 ; i < dwFileHeight/4; i++)
            {
                nvReadFile(hInputYUVFile, yuvInput[cbcr + 1] + dwFrameStrideCbCr + i*(dwSurfWidth/2), dwFileWidth/2, &numBytesRead, NULL);
                fileOffset = (U64)dwFileWidth/2;
                result = nvSetFilePointer64(hInputYUVFile, fileOffset, NULL, FILE_CURRENT);

                if (result == INVALID_SET_FILE_POINTER)
                {
                    return E_FAIL;
                }
            }
        }
    }
    else if (dwFileWidth != dwSurfWidth)
    {
        // load the whole frame
        // read Y
        for (unsigned int i = 0 ; i < dwFileHeight; i++)
        {
            nvReadFile(hInputYUVFile, yuvInput[0] + dwFrameStrideY + i*dwSurfWidth, dwFileWidth, &numBytesRead, NULL);
        }

        // read U,V
        for (int cbcr = 0; cbcr < 2; cbcr++)
        {
            // move in front of chroma
            fileOffset = (U32)(dwInFrameSize*dwFrmIndex + dwFileWidth*dwFileHeight + cbcr*((dwFileWidth* dwFileHeight)/4));
            result = nvSetFilePointer64(hInputYUVFile, fileOffset, NULL, FILE_BEGIN);

            if (result == INVALID_SET_FILE_POINTER)
            {
                return E_FAIL;
            }

            for (unsigned int i = 0 ; i < dwFileHeight/2; i++)
            {
                nvReadFile(hInputYUVFile, yuvInput[cbcr + 1] + dwFrameStrideCbCr + i*dwSurfWidth/2, dwFileWidth/2, &numBytesRead, NULL);
            }
        }
    }
    else
    {
        // direct file read
        nvReadFile(hInputYUVFile, &yuvInput[0][dwFrameStrideY   ], dwFileWidth * dwFileHeight, &numBytesRead, NULL);
        nvReadFile(hInputYUVFile, &yuvInput[1][dwFrameStrideCbCr], dwFileWidth * dwFileHeight/4, &numBytesRead, NULL);
        nvReadFile(hInputYUVFile, &yuvInput[2][dwFrameStrideCbCr], dwFileWidth * dwFileHeight/4, &numBytesRead, NULL);
    }

    return S_OK;
}



#define USE_PRECACHE 1
#if USE_PRECACHE
#define FRAME_QUEUE 240     // Maximum of 240 frames that we will use as an array to buffering frames
#else
#define FRAME_QUEUE 1       // Maximum of 240 frames that we will use as an array to buffering frames
#endif


// Allocate a buffer in system memory and preload 
void CNvEncoderH264::PreloadFrames(unsigned int frameNumber, unsigned int numFramesToEncode, unsigned int reconfigIdx, HANDLE hInput)
{
    unsigned int picHeight = (m_stEncoderInput[reconfigIdx].fieldEncoding == 2) ? m_stEncoderInput[reconfigIdx].height >> 1 : m_stEncoderInput[reconfigIdx].height;
    int lumaPlaneSize      = (m_stEncoderInput[reconfigIdx].width * m_stEncoderInput[reconfigIdx].height);
    int chromaPlaneSize    = (m_stEncoderInput[reconfigIdx].width * m_stEncoderInput[reconfigIdx].height) >> 2;
//    int chromaPlaneSize    = (m_stEncoderInput[reconfigIdx].chromaFormatIDC == 3) ? lumaPlaneSize : ((m_stEncoderInput[reconfigIdx].width * m_stEncoderInput[reconfigIdx].height) >> 2);

    int viewId = 0;
    int  botFieldFirst         = 0;
    bool bTopField             = !botFieldFirst;
    unsigned int curEncWidth   = m_stEncoderInput[reconfigIdx].width;
    unsigned int curEncHeight  = m_stEncoderInput[reconfigIdx].height;

    for (unsigned int frameCount = frameNumber ; frameCount < MIN(frameNumber+FRAME_QUEUE, numFramesToEncode); frameCount++)
    {
        EncodeFrameConfig stEncodeFrame;
        memset(&stEncodeFrame, 0, sizeof(stEncodeFrame));

        // Interlaced Source Input (Field based)
        if (m_stEncoderInput[reconfigIdx].fieldEncoding == 2)
        {
            for (int field = 0; field < 2; field++)
            {
                LoadCurrentFrame(m_pYUV, hInput, frameCount,
                                 m_stEncoderInput[reconfigIdx].width, m_stEncoderInput[reconfigIdx].height,
                                 m_stEncoderInput[reconfigIdx].width, picHeight,
                                !m_stEncoderInput[reconfigIdx].fieldEncoding, bTopField, FRAME_QUEUE, 
                                 0 ); // m_stEncoderInput[reconfigIdx].chromaFormatIDC);
            }
        }
        else // Progressive Source Input (Frame Based)
        {
            LoadCurrentFrame(m_pYUV, hInput, frameCount,
                             m_stEncoderInput[reconfigIdx].width, m_stEncoderInput[reconfigIdx].height,
                             m_stEncoderInput[reconfigIdx].width, m_stEncoderInput[reconfigIdx].height,
                             false, false, FRAME_QUEUE, 0); //  m_stEncoderInput[reconfigIdx].chromaFormatIDC);
        }

        viewId = viewId ^ 1 ;
    }
}

int CNvEncoderH264::EncoderMain(EncoderGPUInfo encoderInfo, 
                                EncoderAppParams appParams, 
                                int argc, const char *argv[])
{
    int encoderID = appParams.nDeviceID;
    InitDefault();
    char *outputExt = NULL;
    int filename_length = (int)strlen(appParams.outputFile);
    int extension_index = getFileExtension(appParams.outputFile, &outputExt);

    EncodeFrameConfig stEncodeFrame;

    strncpy(m_outputFilename, appParams.outputFile, extension_index-1);
    m_outputFilename[extension_index-1] = '\0';
    sprintf(m_outputFilename, "%s.gpu%d.%s", m_outputFilename, encoderID, outputExt);
    m_stEncoderInput[0].fOutput  = fopen(m_outputFilename, "wb+");
    m_stEncoderInput[0].endFrame = appParams.numFramesToEncode;
    if(appParams.configFile !=NULL)
    {
        ParseConfigFile(appParams.configFile);
    }
    getCmdLineArgumentValue(argc, (const char **)argv, "interfaceType",         &m_stEncoderInput[0].interfaceType);
    getCmdLineArgumentValue(argc, (const char **)argv, "preset",                &m_stEncoderInput[0].preset);

    OpenEncodeSession(argc, (const char **)argv, encoderInfo.device);
    PreInitializeEncoder();

    if (S_OK != InitializeEncoder())
    {
        nvPrintf(stderr, NV_LOG_ERROR, "\n nvEncoder Error: NVENC H.264 encoder initialization failure! Check input params!\n");
        return 1;
    }
    if ((appParams.showCaps == 1) && (encoderID == 0))
    {
        queryAllEncoderCaps(this);
    }
    for(int j=1; j < MAX_RECONFIGURATION; j++)
    {
        memcpy(&m_stEncoderInput[j], &m_stEncoderInput[0], sizeof(EncodeConfig));
    }
    if(appParams.reConfigFile !=NULL)
    {
        ParseReConfigFile(appParams.reConfigFile);
    }

    for(int i = 0; i <= (int)m_dwNumReConfig; i++)
        DisplayEncodingParams(appParams, i);

    int lumaPlaneSize    =  m_stEncoderInput[0].width * m_stEncoderInput[0].height;
    int chromaPlaneSize  = (m_stEncoderInput[0].width * m_stEncoderInput[0].height) >> 2;
    
    HANDLE hInput = nvOpenFile( m_stEncoderInput[0].InputClip );
    unsigned int numSourceFrames = CalculateFramesFromInput(hInput, m_stEncoderInput[0].InputClip, 
                                                            m_stEncoderInput[0].width, m_stEncoderInput[0].height);
    if (appParams.numFramesToEncode == 0) {
        appParams.numFramesToEncode = (unsigned int)numSourceFrames;
    } else {
        appParams.numFramesToEncode = MIN(appParams.numFramesToEncode, numSourceFrames);
    }

    NvPrintf("\n ** Start Encode <%s>, Frames [0,%d] ** \n", m_stEncoderInput[0].InputClip, appParams.numFramesToEncode);

    m_dwReConfigIdx = 0;
    int curYuvframecnt  = 0;

    m_pYUV[0] = new unsigned char[FRAME_QUEUE * lumaPlaneSize  ];
    m_pYUV[1] = new unsigned char[FRAME_QUEUE * chromaPlaneSize];
    m_pYUV[2] = new unsigned char[FRAME_QUEUE * chromaPlaneSize];

    sdkCreateTimer(&m_timer);
    sdkResetTimer(&m_timer);

    double total_encode_time = 0.0, sum = 0.0;

    // This is the main loop that will send frames to the NVENC hardware encoder
    for (unsigned int frameNumber = 0 ; frameNumber < appParams.numFramesToEncode; frameNumber += FRAME_QUEUE)
    {
#if USE_PRECACHE
        NvPrintf("\nLoading Frames [%d,%d] into system memory queue (%d frames)\n", frameNumber,
                (MIN(frameNumber+FRAME_QUEUE, appParams.numFramesToEncode))-1,
                (MIN(frameNumber+FRAME_QUEUE, appParams.numFramesToEncode))-frameNumber);
#endif
        // Step #1, if USE_PRECACHE is defined as 1, then buffer FRAME_QUEUE # of frames.
        // We can send to the HW encoder through system memory to minimize the impact that disk I/O has on performance
        for (unsigned int frameCount = frameNumber ; frameCount < MIN(frameNumber+FRAME_QUEUE, appParams.numFramesToEncode); )
        {
            memset(&stEncodeFrame, 0, sizeof(stEncodeFrame));
            stEncodeFrame.yuv[0] = m_pYUV[0];
            stEncodeFrame.yuv[1] = m_pYUV[1];
            stEncodeFrame.yuv[2] = m_pYUV[2];

            if( frameNumber == m_stEncoderInput[m_dwReConfigIdx].endFrame)
            {
                m_dwReConfigIdx++;
                nvCloseFile(hInput);
                // If there is a failure to open the file, this function will quit and print an error message
                hInput = nvOpenFile(m_stEncoderInput[m_dwReConfigIdx].InputClip);
                if (S_OK != ReconfigureEncoder( m_stEncoderInput[m_dwReConfigIdx]))
                {
                    nvPrintf(stderr, NV_LOG_ERROR, "\n nvEncoder.exe Error: NVENC HW encoder initialization failure!  Please check input params for encoder %d!\n",encoderID);
                    return 1;
                }
                stEncodeFrame.bReconfigured = 1;
                curYuvframecnt              = 0;
            }

            stEncodeFrame.stride[0] = m_stEncoderInput[m_dwReConfigIdx].width;
            stEncodeFrame.stride[1] = m_stEncoderInput[m_dwReConfigIdx].width/2;
            stEncodeFrame.stride[2] = m_stEncoderInput[m_dwReConfigIdx].width/2;

    #if USE_PRECACHE
            // Precache frames in the YUV (we load up to FRAME_QUEUE # of frames at a time
            PreloadFrames(frameCount, 
                          MIN(frameNumber+FRAME_QUEUE, appParams.numFramesToEncode) - frameNumber, 
                          m_dwReConfigIdx, hInput);
            frameCount += MIN(FRAME_QUEUE, (appParams.numFramesToEncode-frameNumber));
    #else
            // Frame Based source inputs
            LoadCurrentFrame(m_pYUV, hInput, curYuvframecnt);
            frameCount++;
    #endif

        }

        // Step #2, send frames from system memory directly to the HW encoder

        // We are only timing the Encoding time (YUV420->NV12 Tiled and the Encoding)
        // Now start the encoding process
#if USE_PRECACHE
        NvPrintf("Encoding Frames [%d,%d]\n", frameNumber, (MIN(frameNumber+FRAME_QUEUE, appParams.numFramesToEncode)-1));
#endif
        sdkStartTimer(&m_timer);

        for (unsigned int frameCount = frameNumber ; frameCount < MIN(frameNumber+FRAME_QUEUE, appParams.numFramesToEncode); frameCount++)
        {
            stEncodeFrame.width  = m_stEncoderInput[m_dwReConfigIdx].width;
            stEncodeFrame.height = m_stEncoderInput[m_dwReConfigIdx].height;

            stEncodeFrame.yuv[0] = &m_pYUV[0][(frameCount % FRAME_QUEUE)*m_stEncoderInput[m_dwReConfigIdx].width*m_stEncoderInput[m_dwReConfigIdx].height];
            stEncodeFrame.yuv[1] = &m_pYUV[1][(frameCount % FRAME_QUEUE)*m_stEncoderInput[m_dwReConfigIdx].width*m_stEncoderInput[m_dwReConfigIdx].height/4];
            stEncodeFrame.yuv[2] = &m_pYUV[2][(frameCount % FRAME_QUEUE)*m_stEncoderInput[m_dwReConfigIdx].width*m_stEncoderInput[m_dwReConfigIdx].height/4];

            stEncodeFrame.stride[0] = m_stEncoderInput[m_dwReConfigIdx].width;
            stEncodeFrame.stride[1] = m_stEncoderInput[m_dwReConfigIdx].width/2;
            stEncodeFrame.stride[2] = m_stEncoderInput[m_dwReConfigIdx].width/2;
            
            if (m_stEncoderInput[m_dwReConfigIdx].fieldEncoding == 1)
            {
                stEncodeFrame.picStruct = m_stEncoderInput[m_dwReConfigIdx].bottomFieldFrist ==1 ?NV_ENC_PIC_STRUCT_FIELD_BOTTOM_TOP : NV_ENC_PIC_STRUCT_FIELD_TOP_BOTTOM;
            }
            else
            {
                stEncodeFrame.picStruct = NV_ENC_PIC_STRUCT_FRAME;
            }

            EncodeFrame(&stEncodeFrame, false);
            if (frameCount == appParams.numFramesToEncode-1)
            {
                NvPrintf("%d. ", frameCount);
                NvPrintf("\n\n>> Last Encoded Frame completed <<\n", encoderID);
                EncodeFrame(NULL,true);
            }
            curYuvframecnt++;
            stEncodeFrame.bReconfigured = 0;

            if (appParams.maxNumberEncoders > 1)
            {
                NvPrintf("[%d]%d, ", encoderID, frameCount);
            }
            else
            {
                if (frameCount < appParams.numFramesToEncode-1)
                    NvPrintf("%d. ", frameCount);
            }
        }
        sdkStopTimer(&m_timer);
        sum = sdkGetTimerValue(&m_timer);
        total_encode_time += sum;

        NvPrintf("\nencodeID[%d], Frames [%d,%d] Encode Time = %6.2f (ms)\n", encoderID, frameNumber,
                MIN(frameNumber+FRAME_QUEUE, appParams.numFramesToEncode)-1, sum);
    }

    // Encoding Complete, now print statistics
    DestroyEncoder();

    NvPrintf("  Frames Encoded     : %6d\n", appParams.numFramesToEncode);
    NvPrintf("  Total Encode Time  : %6.2f (sec)\n", total_encode_time / 1000.0f);
    NvPrintf("  Average Time/Frame : %6.2f (ms)\n",  total_encode_time / appParams.numFramesToEncode);
    NvPrintf("  Average Frame Rate : %6.2f (fps)\n", appParams.numFramesToEncode * 1000.0f / total_encode_time);

    if (m_stEncoderInput[0].fOutput)
    {
        fclose(m_stEncoderInput[0].fOutput);

        FILE *fOutput = fopen(m_outputFilename, "rb");
        fseek(fOutput, 0, SEEK_END);
        long int file_size = ftell(fOutput);
        fclose(fOutput);
        NvPrintf("  OutputFile[%d] = %s\n", encoderID, m_outputFilename);
        NvPrintf("  Filesize[%d] = %ld\n", encoderID, file_size);
        NvPrintf("  Average Bitrate[%d] (%4.2f seconds) %4.3f (bps)\n", encoderID,
                (float)appParams.numFramesToEncode / ((float)m_stEncoderInput[0].frameRateNum / (float)m_stEncoderInput[0].frameRateDen),
                (float)file_size*8.0f * (float)m_stEncoderInput[0].frameRateNum / ((float)appParams.numFramesToEncode * (float)m_stEncoderInput[0].frameRateDen));
    }

    nvCloseFile(hInput);

    for (unsigned int i = 0; i < 3; i++)
    {
        if (m_pYUV[i])
        {
            delete [] m_pYUV[i];
            m_pYUV[i] = NULL;
        }
    }

    sdkDeleteTimer(&m_timer);
    //printf("\n* NVENC completed encoding H.264 video saved to: %s \n", m_outputFilename);

    return 0;
}

HRESULT CNvEncoderH264::InitializeEncoder()
{
    HRESULT hr           = S_OK;
    NVENCSTATUS nvStatus = NV_ENC_SUCCESS;

    // Initialize the Encoder
    nvStatus = m_pEncodeAPI->nvEncInitializeEncoder(m_hEncoder, &m_stInitEncParams);

    if (nvStatus == NV_ENC_SUCCESS)
    {
        // Allocate IO buffers
        int numMBs = ((m_uMaxWidth + 15)/16) * ((m_uMaxHeight + 15)/16);
        int NumIOBuffers = (numMBs >= 8160) ? 16 : 32;
        AllocateIOBuffers(m_uMaxWidth, m_uMaxHeight, NumIOBuffers);
        hr = S_OK;
    }
    else
        hr = E_FAIL;

    // intialize output thread
    if (hr == S_OK && !m_pEncoderThread)
    {
        m_pEncoderThread = new CNvEncoderThread(reinterpret_cast<CNvEncoder *>(this), MAX_OUTPUT_QUEUE);

        if (!m_pEncoderThread)
        {
            hr = E_FAIL;
        }
        else
        {
            m_pEncoderThread->ThreadStart();
        }
    }

    if (hr == S_OK)
        m_bEncoderInitialized = true;

    return hr;
}

void CNvEncoderH264::PreInitializeEncoder()
{
    m_bAsyncModeEncoding = (bool)!m_stEncoderInput[0].syncMode;

    m_uMaxHeight         = m_stEncoderInput[0].maxHeight;
    m_uMaxWidth          = m_stEncoderInput[0].maxWidth;
    m_dwFrameWidth       = m_stEncoderInput[0].width;
    m_dwFrameHeight      = m_stEncoderInput[0].height;
    memset(&m_stInitEncParams, 0, sizeof(NV_ENC_INITIALIZE_PARAMS));
    SET_VER(m_stInitEncParams, NV_ENC_INITIALIZE_PARAMS);
    m_stInitEncParams.encodeConfig = &m_stEncodeConfig;
    SET_VER(m_stEncodeConfig, NV_ENC_CONFIG);
    m_stInitEncParams.darHeight           = m_dwFrameHeight;
    m_stInitEncParams.darWidth            = m_dwFrameWidth;
    m_stInitEncParams.encodeHeight        = m_dwFrameHeight;
    m_stInitEncParams.encodeWidth         = m_dwFrameWidth;

    m_stInitEncParams.maxEncodeHeight     = m_uMaxHeight;
    m_stInitEncParams.maxEncodeWidth      = m_uMaxWidth;

    m_stInitEncParams.frameRateNum        = m_stEncoderInput[0].frameRateNum;
    m_stInitEncParams.frameRateDen        = m_stEncoderInput[0].frameRateDen;
    //Fix me add theading model
    m_stInitEncParams.enableEncodeAsync   = m_bAsyncModeEncoding;
    m_stInitEncParams.enablePTD           = m_stEncoderInput[0].enablePTD;
    m_stInitEncParams.encodeGUID          = m_stEncodeGUID;
    m_stInitEncParams.presetGUID          = m_stPresetGUID;

    //m_stInitEncParams.encodeConfig->encodeCodecConfig.h264Config.entropyCodingMode          = ((m_stEncoderInput[0].profile > 66) && (m_stEncoderInput[0].vle_cabac_enable == 1 )) ? NV_ENC_H264_ENTROPY_CODING_MODE_CABAC : NV_ENC_H264_ENTROPY_CODING_MODE_CAVLC;

    m_stInitEncParams.encodeConfig->encodeCodecConfig.h264Config.idrPeriod      = m_stInitEncParams.encodeConfig->gopLength ;
    m_stInitEncParams.encodeConfig->encodeCodecConfig.h264Config.bdirectMode    = m_stEncoderInput[0].numBFrames > 0 ? NV_ENC_H264_BDIRECT_MODE_TEMPORAL : NV_ENC_H264_BDIRECT_MODE_DISABLE;
    m_stInitEncParams.encodeConfig->frameFieldMode                              = m_stEncoderInput[0].fieldEncoding ? NV_ENC_PARAMS_FRAME_FIELD_MODE_FIELD : NV_ENC_PARAMS_FRAME_FIELD_MODE_FRAME ;

    m_stInitEncParams.encodeConfig->profileGUID                  = m_stCodecProfileGUID;
    m_stInitEncParams.encodeConfig->monoChromeEncoding   = 0;
    m_stInitEncParams.encodeConfig->encodeCodecConfig.h264Config.disableDeblockingFilterIDC = 0;
    m_stInitEncParams.encodeConfig->encodeCodecConfig.h264Config.disableSPSPPS  = 0;
    m_stInitEncParams.encodeConfig->encodeCodecConfig.h264Config.sliceMode      = 3;
    m_stInitEncParams.encodeConfig->encodeCodecConfig.h264Config.sliceModeData  = m_stEncoderInput[0].numSlices;
}

HRESULT CNvEncoderH264::ReconfigureEncoder(EncodeConfig EncoderReConfig)
{
    // Initialize the Encoder
    memcpy(&m_stEncoderInput ,&EncoderReConfig, sizeof(EncoderReConfig));
    m_stInitEncParams.encodeHeight        =  EncoderReConfig.height;
    m_stInitEncParams.encodeWidth         =  EncoderReConfig.width;
    m_stInitEncParams.darWidth            =  EncoderReConfig.width;
    m_stInitEncParams.darHeight           =  EncoderReConfig.height;

    m_stInitEncParams.frameRateNum        =  EncoderReConfig.frameRateNum;
    m_stInitEncParams.frameRateDen        =  EncoderReConfig.frameRateDen;
    //m_stInitEncParams.presetGUID          = m_stPresetGUID;
    m_stInitEncParams.encodeConfig->rcParams.maxBitRate         = EncoderReConfig.peakBitRate;
    m_stInitEncParams.encodeConfig->rcParams.averageBitRate     = EncoderReConfig.avgBitRate;
    m_stInitEncParams.encodeConfig->frameFieldMode              = EncoderReConfig.fieldEncoding ? NV_ENC_PARAMS_FRAME_FIELD_MODE_FIELD : NV_ENC_PARAMS_FRAME_FIELD_MODE_FRAME ;
    m_stInitEncParams.encodeConfig->rcParams.vbvBufferSize      = EncoderReConfig.vbvBufferSize;
    m_stInitEncParams.encodeConfig->rcParams.vbvInitialDelay    = EncoderReConfig.vbvInitialDelay;
    m_stInitEncParams.encodeConfig->encodeCodecConfig.h264Config.disableSPSPPS = 0;
    memcpy( &m_stReInitEncParams.reInitEncodeParams, &m_stInitEncParams, sizeof(m_stInitEncParams));
    SET_VER(m_stReInitEncParams, NV_ENC_RECONFIGURE_PARAMS);
    m_stReInitEncParams.resetEncoder    = true;
    NVENCSTATUS nvStatus = m_pEncodeAPI->nvEncReconfigureEncoder(m_hEncoder, &m_stReInitEncParams);
    return nvStatus;
}

HRESULT CNvEncoderH264::EncodeFrame(EncodeFrameConfig *pEncodeFrame, bool bFlush)
{
    NVENCSTATUS nvStatus = NV_ENC_SUCCESS;
    HRESULT hr = S_OK;
    NV_ENC_MAP_INPUT_RESOURCE mapRes = {0};
    unsigned int dwCurWidth;
    unsigned int dwCurHeight;

    if (bFlush)
    {
        FlushEncoder();
        return S_OK;
    }

    if (!pEncodeFrame)
    {
        return E_FAIL;
    }

    EncodeInputSurfaceInfo  *pInput;
    EncodeOutputBuffer      *pOutputBitstream;

    if (!m_stInputSurfQueue.Remove(pInput, INFINITE))
    {
        assert(0);
    }


    if (!m_stOutputSurfQueue.Remove(pOutputBitstream, INFINITE))
    {
        assert(0);
    }

    unsigned int lockedPitch = 0;
    // encode width and height
    unsigned int dwWidth =  m_uMaxWidth; //m_stEncoderInput.width;
    unsigned int dwHeight = m_uMaxHeight;//m_stEncoderInput.height;
    // Align 32 as driver does the same
    unsigned int dwSurfWidth  = (dwWidth + 0x1f) & ~0x1f;
    unsigned int dwSurfHeight = (dwHeight + 0x1f) & ~0x1f;
    unsigned char *pLuma    = pEncodeFrame->yuv[0];
    unsigned char *pChromaU = pEncodeFrame->yuv[1];
    unsigned char *pChromaV = pEncodeFrame->yuv[2];
    unsigned char *pInputSurface = NULL;
    unsigned char *pInputSurfaceCh = NULL;
    dwCurWidth  = pEncodeFrame->width;
    dwCurHeight = pEncodeFrame->height;

    if (m_stEncoderInput[m_dwReConfigIdx].useMappedResources)
    {
        if (m_stEncoderInput[m_dwReConfigIdx].interfaceType == NV_ENC_CUDA)
        {
            lockedPitch = pInput->dwCuPitch;
            pInputSurface = pInput->pExtAllocHost;
            pInputSurfaceCh = pInputSurface + (dwSurfHeight*pInput->dwCuPitch);
        }

#if defined(NV_WINDOWS) && !defined (_NO_D3D)
        if (m_stEncoderInput[m_dwReConfigIdx].interfaceType == NV_ENC_DX9)
        {
            D3DLOCKED_RECT lockRect = {0};
            IDirect3DSurface9 *pSurf = (IDirect3DSurface9 *)(pInput->pExtAlloc);
            HRESULT hr1 = pSurf->LockRect(&lockRect, NULL, 0);

            if (hr1 == S_OK)
            {
                pInputSurface = (unsigned char *)lockRect.pBits;
                lockedPitch = lockRect.Pitch;
            }
            else
            {
                return hr1;
            }
        }
#endif
    }
    else
    {
        pInputSurface = LockInputBuffer(pInput->hInputSurface, &lockedPitch);
        pInputSurfaceCh = pInputSurface + (dwSurfHeight*lockedPitch);
    }

    if (IsYV12PLFormat(pInput->bufferFmt))
    {
        for (unsigned int h = 0; h < dwHeight; h++)
        {
            memcpy(&pInputSurface[lockedPitch * h], &pLuma[pEncodeFrame->stride[0] * h], dwWidth);
        }

        for (unsigned int h = 0; h < dwHeight/2; h++)
        {
            memcpy(&pInputSurfaceCh[h * (lockedPitch/2)] , &pChromaV[h * pEncodeFrame->stride[2]], dwWidth/2);
        }

        pInputSurfaceCh = pInputSurface + (dwSurfHeight*lockedPitch) + ((dwSurfHeight * lockedPitch)>>2);

        for (unsigned int h = 0; h < dwHeight/2; h++)
        {
            memcpy(&pInputSurfaceCh[h * (lockedPitch/2)] , &pChromaU[h * pEncodeFrame->stride[1]], dwWidth/2);
        }
    }
    else if (IsNV12Tiled16x16Format(pInput->bufferFmt))
    {
        convertYUVpitchtoNV12tiled16x16(pLuma, pChromaU, pChromaV,pInputSurface, pInputSurfaceCh, dwWidth, dwHeight, dwWidth, lockedPitch);
    }
    else if (IsNV12PLFormat(pInput->bufferFmt))
    {
        pInputSurfaceCh = pInputSurface + (pInput->dwHeight*lockedPitch);
        convertYUVpitchtoNV12(pLuma, pChromaU, pChromaV,pInputSurface, pInputSurfaceCh, dwCurWidth, dwCurHeight, dwCurWidth, lockedPitch);
    }
    else if (IsYUV444Tiled16x16Format(pInput->bufferFmt))
    {
        unsigned char *pInputSurfaceCb = pInputSurface   + (dwSurfHeight * lockedPitch);
        unsigned char *pInputSurfaceCr = pInputSurfaceCb + (dwSurfHeight * lockedPitch);
        convertYUVpitchtoYUV444tiled16x16(pLuma, pChromaU, pChromaV, pInputSurface, pInputSurfaceCb, pInputSurfaceCr, dwWidth, dwHeight, dwWidth, lockedPitch);
    }
    else if (IsYUV444PLFormat(pInput->bufferFmt))
    {
        unsigned char *pInputSurfaceCb = pInputSurface   + (pInput->dwHeight * lockedPitch);
        unsigned char *pInputSurfaceCr = pInputSurfaceCb + (pInput->dwHeight * lockedPitch);
        convertYUVpitchtoYUV444(pLuma, pChromaU, pChromaV, pInputSurface, pInputSurfaceCb, pInputSurfaceCr, dwWidth, dwHeight, dwWidth, lockedPitch);
    }

    // CUDA or DX9 interop with NVENC
    if (m_stEncoderInput[m_dwReConfigIdx].useMappedResources)
    {
        // Here we copy from Host to Device Memory (CUDA)
        if (m_stEncoderInput[m_dwReConfigIdx].interfaceType == NV_ENC_CUDA)
        {
            cuCtxPushCurrent(m_cuContext); // Necessary to bind the
            CUcontext cuContextCurr;

            CUresult result = cuMemcpyHtoD((CUdeviceptr)pInput->pExtAlloc, pInput->pExtAllocHost, pInput->dwCuPitch*pInput->dwHeight*3/2);
            cuCtxPopCurrent(&cuContextCurr);
        }

#if defined(NV_WINDOWS) && !defined (_NO_D3D)
        // TODO: Grab a pointer GPU Device Memory (DX9) and then copy the result
        if (m_stEncoderInput[m_dwReConfigIdx].interfaceType == NV_ENC_DX9)
        {
            IDirect3DSurface9 *pSurf = (IDirect3DSurface9 *)pInput->pExtAlloc;
            pSurf->UnlockRect();
        }
#endif
        SET_VER(mapRes, NV_ENC_MAP_INPUT_RESOURCE);
        mapRes.registeredResource  = pInput->hRegisteredHandle;
        nvStatus = m_pEncodeAPI->nvEncMapInputResource(m_hEncoder, &mapRes);
        pInput->hInputSurface = mapRes.mappedResource;
    }
    else // here we just pass the frame in system memory to NVENC
    {
        UnlockInputBuffer(pInput->hInputSurface);
    }

    memset(&m_stEncodePicParams, 0, sizeof(m_stEncodePicParams));
    SET_VER(m_stEncodePicParams, NV_ENC_PIC_PARAMS);
    m_stEncodePicParams.inputBuffer     = pInput->hInputSurface;
    m_stEncodePicParams.bufferFmt       = pInput->bufferFmt;
    m_stEncodePicParams.inputWidth      = pInput->dwWidth;
    m_stEncodePicParams.inputHeight     = pInput->dwHeight;
    m_stEncodePicParams.outputBitstream = pOutputBitstream->hBitstreamBuffer;
    m_stEncodePicParams.completionEvent = m_bAsyncModeEncoding == true ? pOutputBitstream->hOutputEvent : NULL;
    m_stEncodePicParams.pictureStruct   = pEncodeFrame->picStruct;
    m_stEncodePicParams.encodePicFlags  = 0;
    m_stEncodePicParams.inputTimeStamp  = 0;
    m_stEncodePicParams.inputDuration   = 0;
    m_stEncodePicParams.codecPicParams.h264PicParams.sliceMode = m_stEncodeConfig.encodeCodecConfig.h264Config.sliceMode;
    m_stEncodePicParams.codecPicParams.h264PicParams.sliceModeData = m_stEncodeConfig.encodeCodecConfig.h264Config.sliceModeData;
    memcpy(&m_stEncodePicParams.rcParams,&m_stEncodeConfig.rcParams, sizeof(m_stEncodePicParams.rcParams));

    if (!m_stInitEncParams.enablePTD)
    {
        m_stEncodePicParams.codecPicParams.h264PicParams.refPicFlag = 1;
        m_stEncodePicParams.codecPicParams.h264PicParams.displayPOCSyntax = 2*m_dwFrameNumInGOP;
        m_stEncodePicParams.pictureType = ((m_dwFrameNumInGOP % m_stEncoderInput[m_dwReConfigIdx].gopLength) == 0) ? NV_ENC_PIC_TYPE_IDR : NV_ENC_PIC_TYPE_P;
        if(pEncodeFrame->bReconfigured == 1)
        {
             m_stEncodePicParams.pictureType = NV_ENC_PIC_TYPE_IDR;
        }
    }

    if ((m_bAsyncModeEncoding == false) &&
        (m_stInitEncParams.enablePTD == 1))
    {
        EncoderThreadData stThreadData;
        stThreadData.pOutputBfr = pOutputBitstream;
        stThreadData.pInputBfr = pInput;
        stThreadData.pOutputBfr->bReconfiguredflag = pEncodeFrame->bReconfigured;
        pOutputBitstream->bWaitOnEvent = false;
        m_pEncodeFrameQueue.Add(stThreadData);
    }

    nvStatus = m_pEncodeAPI->nvEncEncodePicture(m_hEncoder, &m_stEncodePicParams);

    m_dwFrameNumInGOP++;

    if ((m_bAsyncModeEncoding == false) &&
        (m_stInitEncParams.enablePTD == 1))
    {
        if (nvStatus == NV_ENC_SUCCESS)
        {
            EncoderThreadData stThreadData;

            while (m_pEncodeFrameQueue.Remove(stThreadData, 0))
            {
                m_pEncoderThread->QueueSample(stThreadData);
            }
        }
        else
        {
            assert(nvStatus == NV_ENC_ERR_NEED_MORE_INPUT);
        }
    }
    else
    {
        if (nvStatus == NV_ENC_SUCCESS)
        {
            EncoderThreadData stThreadData;
            stThreadData.pOutputBfr = pOutputBitstream;
            stThreadData.pInputBfr = pInput;
            pOutputBitstream->bWaitOnEvent = true;
            stThreadData.pOutputBfr->bReconfiguredflag = pEncodeFrame->bReconfigured;

            // Queue o/p Sample
            if (!m_pEncoderThread->QueueSample(stThreadData))
            {
                assert(0);
            }
        }
        else
        {
            assert(0);
        }
    }

    return hr;
}

HRESULT CNvEncoderH264::DestroyEncoder()
{
    HRESULT hr = S_OK;
    // common
    hr = ReleaseEncoderResources();
    return hr;
}
