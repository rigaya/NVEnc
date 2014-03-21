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

//---------------------------------------------------------------------------
//
// encodeUtils.h
//
//---------------------------------------------------------------------------

#ifndef _ENCODE_UTILS_H
#define _ENCODE_UTILS_H

#include "nvEncodeAPI.h"

#if defined __linux
#include <sys/stat.h>
#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#endif

#include <string.h>
#include "NvTypes.h"

#define PRINTERR(message) \
    fprintf(stderr, (message)); \
    fprintf(stderr, "\n-> @ %s, line %d\n", __FILE__, __LINE__);

#if defined (WIN32) || defined (_WIN32)
#define LICENSE_FILE "C:\\NvEncodeSDKLicense.bin"
#else
#define LICENSE_FILE "~/NvEncodeSDKLicense.bin"
#endif

inline const char *getVideoFormatString(unsigned int dwFormat)
{
    switch (dwFormat)
    {
        case NV_ENC_BUFFER_FORMAT_NV12_PL:
            return "NV12 (Semi-Planar UV Interleaved) Pitch Linear";
            break;

        case NV_ENC_BUFFER_FORMAT_NV12_TILED16x16:
            return "NV12 (Semi-Planar UV Interleaved) Tiled 16x16";
            break;

        case NV_ENC_BUFFER_FORMAT_NV12_TILED64x16:
            return "NV12 (Semi-Planar UV Interleaved) Tiled 64x16";
            break;

        case NV_ENC_BUFFER_FORMAT_YV12_PL:
            return "YV12 (Planar YUV) Pitch Linear";
            break;

        case NV_ENC_BUFFER_FORMAT_YV12_TILED16x16:
            return "YV12 (Planar YUV) Tiled 16x16";
            break;

        case NV_ENC_BUFFER_FORMAT_YV12_TILED64x16:
            return "YV12 (Planar YUV) Tiled 64x16";
            break;

        case NV_ENC_BUFFER_FORMAT_IYUV_PL:
            return "IYUV (Planar YUV) Pitch Linear";
            break;

        case NV_ENC_BUFFER_FORMAT_IYUV_TILED16x16:
            return "IYUV (Planar YUV) Tiled 16x16";
            break;

        case NV_ENC_BUFFER_FORMAT_IYUV_TILED64x16:
            return "IYUV (Planar YUV) Tiled 64x16";
            break;

        case NV_ENC_BUFFER_FORMAT_YUV444_PL:
            return "YUV444 (Planar YUV) Pitch Linear";
            break;

        case NV_ENC_BUFFER_FORMAT_YUV444_TILED16x16:
            return "YUV444 (Planar YUV) Tiled 16x16";
            break;

        case NV_ENC_BUFFER_FORMAT_YUV444_TILED64x16:
            return "YUV444 (Planar YUV) Tiled 64x16";
            break;

        default:
            return "Unknown Video Format";
            break;
    }
}



inline bool IsYV12PLFormat(NV_ENC_BUFFER_FORMAT dwFormat)
{
    if (dwFormat == NV_ENC_BUFFER_FORMAT_YV12_PL)
    {
        return true;
    }
    else
        return false;
}

inline bool IsNV12PLFormat(NV_ENC_BUFFER_FORMAT dwFormat)
{
    if (dwFormat == NV_ENC_BUFFER_FORMAT_NV12_PL)
    {
        return true;
    }
    else
    {
        return false;
    }
}

inline bool IsNV12Tiled16x16Format(NV_ENC_BUFFER_FORMAT dwFormat)
{
    if (dwFormat == NV_ENC_BUFFER_FORMAT_NV12_TILED16x16)
    {
        return true;
    }
    else
    {
        return false;
    }
}

inline bool IsNV12Tiled64x16Format(NV_ENC_BUFFER_FORMAT dwFormat)
{
    if (dwFormat == NV_ENC_BUFFER_FORMAT_NV12_TILED64x16)
    {
        return true;
    }
    else
    {
        return false;
    }
}

inline bool IsYUV444PLFormat(NV_ENC_BUFFER_FORMAT dwFormat)
{
    if (dwFormat == NV_ENC_BUFFER_FORMAT_YUV444_PL)
    {
        return true;
    }
    else
    {
        return false;
    }
}

inline bool IsYUV444Tiled16x16Format(NV_ENC_BUFFER_FORMAT dwFormat)
{
    if (dwFormat == NV_ENC_BUFFER_FORMAT_YUV444_TILED16x16)
    {
        return true;
    }
    else
    {
        return false;
    }
}
inline bool IsYUV444Tiled64x16Format(NV_ENC_BUFFER_FORMAT dwFormat)
{
    if (dwFormat == NV_ENC_BUFFER_FORMAT_YUV444_TILED64x16)
    {
        return true;
    }
    else
    {
        return false;
    }
}

inline bool IsNV12Format(NV_ENC_BUFFER_FORMAT dwFormat)
{
    if ((dwFormat == NV_ENC_BUFFER_FORMAT_NV12_PL) ||
        (dwFormat == NV_ENC_BUFFER_FORMAT_NV12_TILED16x16) ||
        (dwFormat == NV_ENC_BUFFER_FORMAT_NV12_TILED64x16))
    {
        return true;
    }
    else
        return false;
}

inline bool IsYV12Format(NV_ENC_BUFFER_FORMAT dwFormat)
{
    if ((dwFormat == NV_ENC_BUFFER_FORMAT_YV12_PL) ||
        (dwFormat == NV_ENC_BUFFER_FORMAT_YV12_TILED16x16) ||
        (dwFormat == NV_ENC_BUFFER_FORMAT_YV12_TILED64x16))
    {
        return true;
    }
    else
        return false;
}

inline bool IsTiled16x16Format(NV_ENC_BUFFER_FORMAT dwFormat)
{
    if ((dwFormat == NV_ENC_BUFFER_FORMAT_NV12_TILED16x16) ||
        (dwFormat == NV_ENC_BUFFER_FORMAT_YV12_TILED16x16) ||
        (dwFormat == NV_ENC_BUFFER_FORMAT_IYUV_TILED16x16))
    {
        return true;
    }
    else
    {
        return false;
    }
}

inline void CvtToTiled16x16(unsigned char *tile, unsigned char *src,
                            unsigned int width, unsigned int height,
                            unsigned int srcStride, unsigned int dstStride)
{
    unsigned int tileNb = 0;
    unsigned int x,y;
    unsigned int offs;

    for (y = 0 ; y < height ; y++)
    {
        for (x = 0 ; x < width; x++)
        {
            tileNb = x/16 + (y/16) * dstStride/16;

            offs = tileNb * 256;
            offs += (y % 16) * 16 + (x % 16);
            tile[offs]   =  src[srcStride*y + x];
        }
    }
}

inline void getTiled16x16Sizes(int width, int height, bool frame , int &luma_size, int &chroma_size)
{
    int  bl_width, bl_height;

    if (frame)
    {
        bl_width  = (width + 15)/16;
        bl_height = (height + 15)/16;
        luma_size =  bl_width * bl_height * 16*16;

        bl_height = (height/2 + 15)/16;
        chroma_size =   bl_width * bl_height * 16*16;
    }
    else
    {
        bl_width  = (width + 15)/16;
        bl_height = (height/2 + 15)/16;
        luma_size =  bl_width * bl_height * 16*16;

        bl_height = (height/4 + 15)/16;
        chroma_size =  bl_width * bl_height * 16*16;
    }
}

inline void convertYUVpitchtoNV12tiled16x16(unsigned char *yuv_luma, unsigned char *yuv_cb, unsigned char *yuv_cr,
                                            unsigned char *tiled_luma, unsigned char *tiled_chroma,
                                            int width, int height, int srcStride, int dstStride)
{
    int tileNb, offs;
    int x,y;

    if (srcStride<0) srcStride = width;

    if (dstStride<0) dstStride = width;

    for (y = 0 ; y < height ; y++)
    {
        for (x= 0 ; x < width; x++)
        {
            tileNb = x/16 + (y/16) * dstStride/16;

            offs = tileNb * 256;
            offs += (y % 16) * 16 + (x % 16);
            tiled_luma[offs]   =  yuv_luma[(srcStride*y) + x];
        }
    }

    for (y = 0 ; y < height/2 ; y++)
    {
        for (x= 0 ; x < width; x = x+2)
        {
            tileNb = x/16 + (y/16) * dstStride/16;

            offs = tileNb * 256;
            offs += (y % 16) * 16 + (x % 16);
            tiled_chroma[offs]   =  yuv_cb[(srcStride/2)*y + x/2];
            tiled_chroma[offs+1] =  yuv_cr[(srcStride/2)*y + x/2];
        }
    }
}

inline void convertYUVpitchtoNV12(unsigned char *yuv_luma, unsigned char *yuv_cb, unsigned char *yuv_cr,
                                  unsigned char *nv12_luma, unsigned char *nv12_chroma,
                                  int width, int height , int srcStride, int dstStride)
{
    int y;
    int x;

    if (srcStride == 0)
        srcStride = width;

    if (dstStride == 0)
        dstStride = width;

    for (y = 0 ; y < height ; y++)
    {
        memcpy(nv12_luma + (dstStride*y), yuv_luma + (srcStride*y) , width);
    }

    for (y = 0 ; y < height/2 ; y++)
    {
        for (x= 0 ; x < width; x=x+2)
        {
            nv12_chroma[(y*dstStride) + x] =    yuv_cb[((srcStride/2)*y) + (x >>1)];
            nv12_chroma[(y*dstStride) +(x+1)] = yuv_cr[((srcStride/2)*y) + (x >>1)];
        }
    }
}

inline
void convertYUVpitchtoYUV444tiled16x16(unsigned char *yuv_luma, unsigned char *yuv_cb, unsigned char *yuv_cr,
                                       unsigned char *tiled_luma, unsigned char *tiled_cb, unsigned char *tiled_cr,
                                       int width, int height, int srcStride, int dstStride)
{
    int tileNb, offs;
    int x,y;

    if (srcStride<0) srcStride = width;

    if (dstStride<0) dstStride = width;

    for (y = 0; y < height; y++)
    {
        for (x = 0; x < width; x++)
        {
            tileNb = x/16 + (y/16) * dstStride/16;
            offs = tileNb * 256;
            offs += (y % 16) * 16 + (x % 16);
            tiled_luma[offs]   =  yuv_luma[(srcStride*y) + x];
            tiled_cb  [offs]   =  yuv_cb  [(srcStride*y) + x];
            tiled_cr  [offs]   =  yuv_cr  [(srcStride*y) + x];
        }
    }
}

inline
void convertYUVpitchtoYUV444(unsigned char *yuv_luma, unsigned char *yuv_cb, unsigned char *yuv_cr,
                             unsigned char *surf_luma, unsigned char *surf_cb, unsigned char *surf_cr,
                             int width, int height, int srcStride, int dstStride)
{
    if (srcStride<0)
        srcStride = width;

    if (dstStride<0)
        dstStride = width;

    for (int h = 0; h < height; h++)
    {
        memcpy(surf_luma + dstStride * h, yuv_luma + srcStride * h, width);
        memcpy(surf_cb   + dstStride * h, yuv_cb   + srcStride * h, width);
        memcpy(surf_cr   + dstStride * h, yuv_cr   + srcStride * h, width);
    }
}

#endif
