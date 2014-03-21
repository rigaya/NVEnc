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

#include <nvEncodeAPI.h>
#include "CNVEncoderH264.h"
#include "xcodeutil.h"
#include <platform/NvTypes.h>

#include <include/videoFormats.h>
#include <include/helper_string.h>
#include <include/helper_timer.h>
#include <include/nvFileIO.h>

#include <stddef.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <ctype.h>

#pragma warning (disable:4189)
#define MAX_LINE_SIZE   2000
extern "C"
void printHelp()
{
    printf("[ NVENC 3.0 Command Line Encoder]\n");

    printf("Usage: nvEncoder -configFile=config.txt -numFramesToEncode=numFramesToEncode -outFile=output.264 <optional params>\n\n");

    printf("> Encoder Test Application Parameters\n");
    printf("   [maxNumberEncoders=n] n=number of encoders to use when multiple GPUs are detected\n");
    printf("> Optional Parameters to be Set from ConfigFile Or from Command Line\n");
    printf("   [frameNum]           Start Frame (within input file)\n");
    printf("   [bitrate]            Video Bitrate of output file (eg: n=6000000 for 6 mbps)\n");
    printf("   [maxbitrate]         Video Bitrate of output file (eg: n=6000000 for 6 mbps)\n");
    printf("   [rcMode]             Rate Control Mode (0=Constant QP, 1=VBR, 2=CBR, 4=VBR_MinQP, 8=Two-Pass CBR\n");
    printf("   [enableInitialRCQP]  Enable Initial Frame RC QP mode setting\n");
    printf("   [initialQPI]         Initial Frame QP for Intra Frame\n");
    printf("   [initialQPP]         Initial Frame QP for Inter P-Frame\n");
    printf("   [initialQPB]         Initial Frame QP for Inter B-Frame\n");
    printf("   [frameRateNum]       Frame Rate numerator   (default = 30000)  (numerator/denominator = 29.97fps)\n");
    printf("   [frameRateDen]       Frame Rate denominator (default =  1001)\n");
    printf("   [gopLength]          Specify GOP length (N=distance between I-Frames)\n");
    printf("   [profile]            H.264 Codec Profile (n=profile #)\n");
    printf("                           66  = (Baseline)\n");
    printf("                           77  = (Main Profile)\n");
    printf("                           100 = (High Profile)\n");
    printf("   [numSlices]          Specify Number of Slices to be encoded per Frame\n");
    printf("   [preset]             Specify the encoding preset\n");
    printf("                         -1 = No preset\n");
    printf("                          0 = NV_ENC_PRESET_DEFAULT\n");
    printf("                          1 = NV_ENC_PRESET_LOW_LATENCY_DEFAULT\n");
    printf("                          2 = NV_ENC_PRESET_HP\n");
    printf("                          3 = NV_ENC_PRESET_HQ\n");
    printf("                          4 = NV_ENC_PRESET_BD\n");
    printf("                          5 = NV_ENC_PRESET_LOW_LATENCY_HQ\n");
    printf("                          6 = NV_ENC_PRESET_LOW_LATENCY_HP\n");
    printf("   [numBFrames]         Number fo B frames between P successive frames\n");
    printf("   [syncMode]           Run NvEnc in sync Mode if set\n");
    printf("   [interfaceType]      Run NvEnc at specified Interface\n");
    printf("                           0  = (DX9)\n");
    printf("                           1  = (DX11)\n");
    printf("                           2  = (cuda)\n");
    printf("                           3  = (DX10)\n");
    printf("   [vbvBufferSize]      HRD buffer size. For low latancy it should be less or equal to single frame size\n");
    printf("   [vbvInitialDelay]    Initial HRD bufffer Fullness\n");
    printf("   [fieldMode]          Field Encoding Mode (0=Frame, 1=Field)\n");
    printf("   [level])             Codec Level value \n");
    printf("   [inFile])            InputClip \n");
    printf("   [enablePtd])         if set picture type decision will be taken by EncodeAPI \n");
    printf("   [reconfigFile]       Reconfiguration will occur after the frameNum mentioned in reconfig file with specified parameter in reconfig file \n");
}

// These are the initial parameters for the NVENC encoder
extern "C"
void parseCmdLineArguments(int argc, const char *argv[], EncoderAppParams *pEncodeAppParams)
{
    if (!checkCmdLineFlag(argc, (const char **)argv, "outFile"))
    {
        printf("Error!  Command line paramters -outFile is required in order to run this application\n");
        printHelp();
        exit(EXIT_FAILURE);
    }
    if (argc > 1)
    {
        // These are parameters specific to the encoding application
        getCmdLineArgumentString(argc, (const char **)argv, "configFile",           &pEncodeAppParams->configFile);
        getCmdLineArgumentString(argc, (const char **)argv, "outFile",              &pEncodeAppParams->outputFile);
        getCmdLineArgumentValue(argc, (const char **)argv, "device",                &pEncodeAppParams->nDeviceID);
        getCmdLineArgumentValue(argc, (const char **)argv, "numFramesToEncode",     &pEncodeAppParams->numFramesToEncode);
        getCmdLineArgumentValue(argc, (const char **)argv, "maxNumberEncoders",     &pEncodeAppParams->maxNumberEncoders);
        if (checkCmdLineFlag(argc, (const char **)argv, "reconfigFile"))
        {
            getCmdLineArgumentString(argc, (const char **)argv, "reconfigFile",     &pEncodeAppParams->reConfigFile);
        }
        pEncodeAppParams->showCaps             = checkCmdLineFlag(argc, (const char **)argv, "showCaps");
    }
}

