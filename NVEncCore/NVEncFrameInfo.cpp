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

#pragma once

#include "NVEncFrameInfo.h"

FrameInfoExtra getFrameInfoExtra(const FrameInfo *pFrameInfo) {
    FrameInfoExtra exinfo = { 0 };
    switch (pFrameInfo->csp) {
    case RGY_CSP_NV12:
        exinfo.width_byte = pFrameInfo->width;
        exinfo.height_total = pFrameInfo->height * 3 / 2;
        break;
    case RGY_CSP_YV12:
        exinfo.width_byte = pFrameInfo->width;
        exinfo.height_total = (pFrameInfo->deivce_mem) ? pFrameInfo->height * 2 : pFrameInfo->height * 3 / 2;
        break;
    case RGY_CSP_YUY2:
        exinfo.width_byte = pFrameInfo->width * 2;
        exinfo.height_total = pFrameInfo->height;
        break;
    case RGY_CSP_YUV422:
        exinfo.width_byte = pFrameInfo->width;
        exinfo.height_total = pFrameInfo->height * 2;
        break;
    case RGY_CSP_YUV444:
        exinfo.width_byte = pFrameInfo->width;
        exinfo.height_total = pFrameInfo->height * 3;
        break;
    case RGY_CSP_YV12_09:
    case RGY_CSP_YV12_10:
    case RGY_CSP_YV12_12:
    case RGY_CSP_YV12_14:
    case RGY_CSP_YV12_16:
        exinfo.width_byte = pFrameInfo->width * 2;
        exinfo.height_total = (pFrameInfo->deivce_mem) ? pFrameInfo->height * 2 : pFrameInfo->height * 3 / 2;
        break;
    case RGY_CSP_P010:
        exinfo.width_byte = pFrameInfo->width * 2;
        exinfo.height_total = pFrameInfo->height * 3 / 2;
        break;
    case RGY_CSP_YUV444_09:
    case RGY_CSP_YUV444_10:
    case RGY_CSP_YUV444_12:
    case RGY_CSP_YUV444_14:
    case RGY_CSP_YUV444_16:
        exinfo.width_byte = pFrameInfo->width * 2;
        exinfo.height_total = pFrameInfo->height * 3;
        break;
    case RGY_CSP_RGB3:
        exinfo.width_byte = (pFrameInfo->width * 3 + 3) & (~3);
        exinfo.height_total = pFrameInfo->height;
        break;
    case RGY_CSP_RGB4:
        exinfo.width_byte = pFrameInfo->width * 4;
        exinfo.height_total = pFrameInfo->height;
        break;
    case RGY_CSP_YC48:
        exinfo.width_byte = pFrameInfo->width * 6;
        exinfo.height_total = pFrameInfo->height;
        break;
    default:
        break;
    }
    exinfo.frame_size = pFrameInfo->pitch * exinfo.height_total;
    return exinfo;
}
