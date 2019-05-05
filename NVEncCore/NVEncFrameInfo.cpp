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
    case RGY_CSP_NV16:
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
    case RGY_CSP_YUV422_09:
    case RGY_CSP_YUV422_10:
    case RGY_CSP_YUV422_12:
    case RGY_CSP_YUV422_14:
    case RGY_CSP_YUV422_16:
        exinfo.width_byte = pFrameInfo->width * 2;
        exinfo.height_total = pFrameInfo->height * 3;
        break;
    case RGY_CSP_P210:
        exinfo.width_byte = pFrameInfo->width * 2;
        exinfo.height_total = pFrameInfo->height * 2;
        break;
    case RGY_CSP_YUV444_09:
    case RGY_CSP_YUV444_10:
    case RGY_CSP_YUV444_12:
    case RGY_CSP_YUV444_14:
    case RGY_CSP_YUV444_16:
        exinfo.width_byte = pFrameInfo->width * 2;
        exinfo.height_total = pFrameInfo->height * 3;
        break;
    case RGY_CSP_RGB24:
    case RGY_CSP_RGB24R:
    case RGY_CSP_BGR24:
        exinfo.width_byte = (pFrameInfo->width * 3 + 3) & (~3);
        exinfo.height_total = pFrameInfo->height;
        break;
    case RGY_CSP_RGB32:
    case RGY_CSP_RGB32R:
    case RGY_CSP_BGR32:
        exinfo.width_byte = pFrameInfo->width * 4;
        exinfo.height_total = pFrameInfo->height;
        break;
    case RGY_CSP_RGB:
        exinfo.width_byte = pFrameInfo->width;
        exinfo.height_total = pFrameInfo->height * 3;
        break;
    case RGY_CSP_RGBA:
        exinfo.width_byte = pFrameInfo->width;
        exinfo.height_total = pFrameInfo->height * 4;
        break;
    case RGY_CSP_YC48:
        exinfo.width_byte = pFrameInfo->width * 6;
        exinfo.height_total = pFrameInfo->height;
        break;
    case RGY_CSP_Y8:
        exinfo.width_byte = pFrameInfo->width;
        exinfo.height_total = pFrameInfo->height;
        break;
    case RGY_CSP_Y16:
        exinfo.width_byte = pFrameInfo->width * 2;
        exinfo.height_total = pFrameInfo->height;
        break;
    default:
        break;
    }
    exinfo.frame_size = pFrameInfo->pitch * exinfo.height_total;
    return exinfo;
}

FrameInfo getPlane(const FrameInfo *frameInfo, const RGY_PLANE plane) {
    FrameInfo planeInfo = *frameInfo;
    if (frameInfo->csp == RGY_CSP_GBR || frameInfo->csp == RGY_CSP_GBRA) {
        switch (plane) {
        case RGY_PLANE_G: break;
        case RGY_PLANE_B: planeInfo.ptr += frameInfo->pitch * frameInfo->height; break;
        case RGY_PLANE_R: planeInfo.ptr += frameInfo->pitch * frameInfo->height * 2; break;
        case RGY_PLANE_A: planeInfo.ptr += frameInfo->pitch * frameInfo->height * 3; break;
        default: break;
        }
    } else {
        switch (plane) {
        case RGY_PLANE_U:
        case RGY_PLANE_V:
            //case RGY_PLANE_G:
            //case RGY_PLANE_B:
            if (frameInfo->csp == RGY_CSP_YUY2
                || RGY_CSP_CHROMA_FORMAT[frameInfo->csp] == RGY_CHROMAFMT_RGB_PACKED
                || RGY_CSP_CHROMA_FORMAT[frameInfo->csp] == RGY_CHROMAFMT_MONOCHROME) {
                ; //なにもしない
            } else if (frameInfo->csp == RGY_CSP_NV12 || frameInfo->csp == RGY_CSP_P010) {
                planeInfo.ptr += frameInfo->pitch * frameInfo->height;
                planeInfo.height >>= 1;
            } else if (frameInfo->csp == RGY_CSP_NV16 || frameInfo->csp == RGY_CSP_P210) {
                planeInfo.ptr += frameInfo->pitch * frameInfo->height;
            } else if (RGY_CSP_CHROMA_FORMAT[frameInfo->csp] == RGY_CHROMAFMT_YUV420) {
                planeInfo.ptr += frameInfo->pitch * frameInfo->height;
                planeInfo.width >>= 1;
                planeInfo.height >>= 1;
                if (plane == RGY_PLANE_V) {
                    planeInfo.ptr += planeInfo.pitch * planeInfo.height;
                }
            } else if (RGY_CSP_CHROMA_FORMAT[frameInfo->csp] == RGY_CHROMAFMT_YUV422) {
                planeInfo.ptr += plane * frameInfo->pitch * frameInfo->height;
                planeInfo.width >>= 1;
                if (plane == RGY_PLANE_V) {
                    planeInfo.ptr += planeInfo.pitch * planeInfo.height;
                }
            } else { //RGY_CHROMAFMT_YUV444 & RGY_CHROMAFMT_RGB
                planeInfo.ptr += plane * planeInfo.pitch * planeInfo.height;
            }
            break;
        case RGY_PLANE_Y:
            //case RGY_PLANE_R:
        default:
            break;
        }
    }
    return planeInfo;
}
