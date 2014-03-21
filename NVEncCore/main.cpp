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
 * ÅgMATERIALSÅh) ARE BEING PROVIDED ÅgAS IS.Åh WITHOUT EXPRESS OR IMPLIED
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

////////////////////////////////////////////////////////////
// This is the main application for using the NV Encode API
//

#if defined(LINUX) || defined (NV_LINUX)
// This is required so that fopen will use the 64-bit equivalents for large file access
#define _FILE_OFFSET_BITS 64
#define _LARGEFILE_SOURCE
#define _LARGEFILE64_SOURCE
#endif

#include <nvEncodeAPI.h>                // the NVENC common API header
#include "CNVEncoderH264.h"             // class definition for the H.264 encoding class
#include "xcodeutil.h"                  // class helper functions for video encoding
#include <platform/NvTypes.h>           // type definitions

#include <cuda.h>                       // include CUDA header for CUDA/NVENC interop
#include <include/helper_cuda_drvapi.h> // helper functions for CUDA driver API
#include <include/helper_string.h>      // helper functions for string parsing
#include <include/helper_timer.h>       // helper functions for timing
#include <include/nvFileIO.h>           // helper functions for large file access

#pragma warning (disable:4189)

const char *sAppName = "nvEncoder";

StopWatchInterface *timer[MAX_ENCODERS];

// Utilities.cpp
//
// printHelp()       - prints all the command options for the NVENC sample application
extern "C" void    printHelp();
// parseCmdLineArguments - parsing command line arguments for EncodeConfig Struct
extern "C" void    parseCmdLineArguments(int argc, const char *argv[], EncoderAppParams *pEncodeAppParamsS);

// Initialization code that checks the GPU encoders available and fills a table
unsigned int checkNumberEncoders(EncoderGPUInfo *encoderInfo)
{
    CUresult cuResult = CUDA_SUCCESS;
    CUdevice cuDevice = 0;

    char gpu_name[100];
    int  deviceCount = 0;
    int  SMminor = 0, SMmajor = 0;
    int  NVENC_devices = 0;

    NvPrintf("\n");

    // CUDA interfaces
    cuResult = cuInit(0);

    if (cuResult != CUDA_SUCCESS)
    {
        fprintf(stderr, ">> GetNumberEncoders() - cuInit() failed error:0x%x\n", cuResult);
        exit(EXIT_FAILURE);
    }

    checkCudaErrors(cuDeviceGetCount(&deviceCount));

    if (deviceCount == 0)
    {
        fprintf(stderr, ">> GetNumberEncoders() - reports no devices available that support CUDA\n");
        exit(EXIT_FAILURE);
    }
    else
    {
        NvPrintf(">> GetNumberEncoders() has detected %d CUDA capable GPU device(s) <<\n", deviceCount);

        for (int currentDevice=0; currentDevice < deviceCount; currentDevice++)
        {
            checkCudaErrors(cuDeviceGet(&cuDevice, currentDevice));
            checkCudaErrors(cuDeviceGetName(gpu_name, 100, cuDevice));
            checkCudaErrors(cuDeviceComputeCapability(&SMmajor, &SMminor, currentDevice));
            NvPrintf("  [ GPU #%d - < %s > has Compute SM %d.%d, NVENC %s ]\n",
                   currentDevice, gpu_name, SMmajor, SMminor,
                   (((SMmajor << 4) + SMminor) >= 0x30) ? "Available" : "Not Available");

            if (((SMmajor << 4) + SMminor) >= 0x30)
            {
                encoderInfo[NVENC_devices].device = currentDevice;
                strcpy(encoderInfo[NVENC_devices].gpu_name, gpu_name);
                NVENC_devices++;
            }
        }
    }

    return NVENC_devices;
}

// Main Console Application for NVENC
int main(int argc, char *argv[])
{
    int retval        = 1;

    CNvEncoder      *pEncoder[MAX_ENCODERS];
    EncoderGPUInfo  encoderInfo[MAX_ENCODERS];
    EncoderAppParams appParams;

    memset(&appParams, 0,sizeof(appParams));
    appParams.maxNumberEncoders    = 1;

#if defined __linux || defined __APPLE_ || defined __MACOSX
    U32 num_bytes_read;
    NvPthreadABIInit();
#endif
 
    // Parse the command line & config file parameters for the application and NVENC
    parseCmdLineArguments(argc, (const char **)argv, &appParams);

    // Query the number of GPUs that have NVENC encoders
    unsigned int numEncoders = MIN(checkNumberEncoders(encoderInfo), appParams.maxNumberEncoders);

    // Depending on the number of available encoders and the maximum encoders, we open multiple FILEs (one per GPU)
    for (unsigned int encoderID=0; encoderID < numEncoders; encoderID++)
    {
        // Create H.264 based encoder
        pEncoder[encoderID] = new CNvEncoderH264();
        appParams.nDeviceID    = encoderID;
        pEncoder[encoderID]->EncoderMain( encoderInfo[encoderID], appParams);
    }


    for (unsigned int i=0; i < numEncoders; i++)
    {
        if (pEncoder[i])
        {
            delete pEncoder[i];
            pEncoder[i] = NULL;
        }
    }

    return 0;
}
