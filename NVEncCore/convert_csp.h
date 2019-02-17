// -----------------------------------------------------------------------------------------
// QSVEnc/NVEnc by rigaya
// -----------------------------------------------------------------------------------------
// The MIT License
//
// Copyright (c) 2011-2016 rigaya
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

#ifndef _CONVERT_CSP_H_
#define _CONVERT_CSP_H_

#include <cstdint>
#include "rgy_tchar.h"

enum RGY_PLANE {
    RGY_PLANE_Y,
    RGY_PLANE_U,
    RGY_PLANE_V,
};

typedef void (*funcConvertCSP) (void **dst, const void **src, int width, int src_y_pitch_byte, int src_uv_pitch_byte, int dst_y_pitch_byte, int height, int dst_height, int *crop);

enum RGY_CSP {
    RGY_CSP_NA = 0,
    RGY_CSP_NV12,
    RGY_CSP_YV12,
    RGY_CSP_YUY2,
    RGY_CSP_YUV422,
    RGY_CSP_NV16,
    RGY_CSP_YUV444,
    RGY_CSP_YV12_09,
    RGY_CSP_YV12_10,
    RGY_CSP_YV12_12,
    RGY_CSP_YV12_14,
    RGY_CSP_YV12_16,
    RGY_CSP_P010,
    RGY_CSP_YUV422_09,
    RGY_CSP_YUV422_10,
    RGY_CSP_YUV422_12,
    RGY_CSP_YUV422_14,
    RGY_CSP_YUV422_16,
    RGY_CSP_P210,
    RGY_CSP_YUV444_09,
    RGY_CSP_YUV444_10,
    RGY_CSP_YUV444_12,
    RGY_CSP_YUV444_14,
    RGY_CSP_YUV444_16,
    RGY_CSP_RGB24R,
    RGY_CSP_RGB32R,
    RGY_CSP_RGB24,
    RGY_CSP_RGB32,
    RGY_CSP_YC48,
    RGY_CSP_Y8,
    RGY_CSP_Y16
};

static const TCHAR *RGY_CSP_NAMES[] = {
    _T("Invalid"),
    _T("nv12"),
    _T("yv12"),
    _T("yuy2"),
    _T("yuv422"),
    _T("nv16"),
    _T("yuv444"),
    _T("yv12(9bit)"),
    _T("yv12(10bit)"),
    _T("yv12(12bit)"),
    _T("yv12(14bit)"),
    _T("yv12(16bit)"),
    _T("p010"),
    _T("yuv422(9bit)"),
    _T("yuv422(10bit)"),
    _T("yuv422(12bit)"),
    _T("yuv422(14bit)"),
    _T("yuv422(16bit)"),
    _T("p210"),
    _T("yuv444(9bit)"),
    _T("yuv444(10bit)"),
    _T("yuv444(12bit)"),
    _T("yuv444(14bit)"),
    _T("yuv444(16bit)"),
    _T("rgb24r"),
    _T("rgb32r"),
    _T("rgb24"),
    _T("rgb32"),
    _T("yc48"),
    _T("y8"),
    _T("yc16")
};

static const uint8_t RGY_CSP_BIT_DEPTH[] = {
     0, //RGY_CSP_NA
     8, //RGY_CSP_NV12
     8, //RGY_CSP_YV12
     8, //RGY_CSP_YUY2
     8, //RGY_CSP_YUV422
     8, //RGY_CSP_NV16
     8, //RGY_CSP_YUV444
     9, //RGY_CSP_YV12_09
    10,
    12,
    14,
    16, //RGY_CSP_YV12_16
    16, //RGY_CSP_P010
     9, //RGY_CSP_YUV422_09
    10,
    12,
    14,
    16, //RGY_CSP_YUV422_16
    16, //RGY_CSP_P210
     9, //RGY_CSP_YUV444_09
    10,
    12,
    14,
    16, //RGY_CSP_YUV444_16
     8, //RGY_CSP_RGB24R
     8, //RGY_CSP_RGB32R
     8, //RGY_CSP_RGB24
     8, //RGY_CSP_RGB32
    10, //RGY_CSP_YC48
     8, //RGY_CSP_Y8
    16, //RGY_CSP_Y16
};

enum RGY_CHROMAFMT {
    RGY_CHROMAFMT_UNKNOWN = 0,
    RGY_CHROMAFMT_MONOCHROME = 0,
    RGY_CHROMAFMT_YUV420,
    RGY_CHROMAFMT_YUV422,
    RGY_CHROMAFMT_YUV444,
    RGY_CHROMAFMT_RGB,
};

static const RGY_CHROMAFMT RGY_CSP_CHROMA_FORMAT[] = {
    RGY_CHROMAFMT_UNKNOWN, //RGY_CSP_NA
    RGY_CHROMAFMT_YUV420, //RGY_CSP_NV12
    RGY_CHROMAFMT_YUV420, //RGY_CSP_YV12
    RGY_CHROMAFMT_YUV422, //RGY_CSP_YUY2
    RGY_CHROMAFMT_YUV422, //RGY_CSP_YUV422
    RGY_CHROMAFMT_YUV422, //RGY_CSP_NV16
    RGY_CHROMAFMT_YUV444, //RGY_CSP_YUV444
    RGY_CHROMAFMT_YUV420, //RGY_CSP_YV12_09
    RGY_CHROMAFMT_YUV420,
    RGY_CHROMAFMT_YUV420,
    RGY_CHROMAFMT_YUV420,
    RGY_CHROMAFMT_YUV420, //RGY_CSP_YV12_16
    RGY_CHROMAFMT_YUV420, //RGY_CSP_P010
    RGY_CHROMAFMT_YUV422, //RGY_CSP_YUV422_09
    RGY_CHROMAFMT_YUV422,
    RGY_CHROMAFMT_YUV422,
    RGY_CHROMAFMT_YUV422,
    RGY_CHROMAFMT_YUV422, //RGY_CSP_YUV422_16
    RGY_CHROMAFMT_YUV422, //RGY_CSP_P210
    RGY_CHROMAFMT_YUV444, //RGY_CSP_YUV444_09
    RGY_CHROMAFMT_YUV444,
    RGY_CHROMAFMT_YUV444,
    RGY_CHROMAFMT_YUV444,
    RGY_CHROMAFMT_YUV444, //RGY_CSP_YUV444_16
    RGY_CHROMAFMT_RGB,
    RGY_CHROMAFMT_RGB,
    RGY_CHROMAFMT_RGB,
    RGY_CHROMAFMT_RGB,
    RGY_CHROMAFMT_YUV444, //RGY_CSP_YC48
    RGY_CHROMAFMT_MONOCHROME,
    RGY_CHROMAFMT_MONOCHROME,
};

static const uint8_t RGY_CSP_BIT_PER_PIXEL[] = {
     0, //RGY_CSP_NA
    12, //RGY_CSP_NV12
    12, //RGY_CSP_YV12
    16, //RGY_CSP_YUY2
    16, //RGY_CSP_YUV422
    16, //RGY_CSP_NV16
    24, //RGY_CSP_YUV444
    24, //RGY_CSP_YV12_09
    24,
    24,
    24,
    24, //RGY_CSP_YV12_16
    24, //RGY_CSP_P010
    32, //RGY_CSP_YUV422_09
    32,
    32,
    32,
    32, //RGY_CSP_YUV422_16
    32, //RGY_CSP_P210
    48, //RGY_CSP_YUV444_09
    48,
    48,
    48,
    48, //RGY_CSP_YUV444_16
    24, //RGY_CSP_RGB24R
    24, //RGY_CSP_RGB32R
    24, //RGY_CSP_RGB24
    32, //RGY_CSP_RGB32
    48, //RGY_CSP_YC48
     8, //RGY_CSP_Y8
    16, //RGY_CSP_Y16
};

enum RGY_PICSTRUCT : uint32_t {
    RGY_PICSTRUCT_UNKNOWN      = 0x00,
    RGY_PICSTRUCT_FRAME        = 0x01, //フレームとして符号化されている
    RGY_PICSTRUCT_TFF          = 0x02,
    RGY_PICSTRUCT_BFF          = 0x04,
    RGY_PICSTRUCT_FIELD        = 0x08, //フィールドとして符号化されている
    RGY_PICSTRUCT_INTERLACED   = 0x00 | RGY_PICSTRUCT_TFF | RGY_PICSTRUCT_BFF,   //インタレ
    RGY_PICSTRUCT_FRAME_TFF    = 0x00 | RGY_PICSTRUCT_TFF | RGY_PICSTRUCT_FRAME, //フレームとして符号化されているインタレ (TFF)
    RGY_PICSTRUCT_FRAME_BFF    = 0x00 | RGY_PICSTRUCT_BFF | RGY_PICSTRUCT_FRAME, //フレームとして符号化されているインタレ (BFF)
    RGY_PICSTRUCT_FIELD_TOP    = 0x00 | RGY_PICSTRUCT_TFF | RGY_PICSTRUCT_FIELD, //フィールドとして符号化されている (Topフィールド)
    RGY_PICSTRUCT_FIELD_BOTTOM = 0x00 | RGY_PICSTRUCT_BFF | RGY_PICSTRUCT_FIELD, //フィールドとして符号化されている (Bottomフィールド)
};

static RGY_PICSTRUCT operator|(RGY_PICSTRUCT a, RGY_PICSTRUCT b) {
    return (RGY_PICSTRUCT)((uint8_t)a | (uint8_t)b);
}

static RGY_PICSTRUCT operator|=(RGY_PICSTRUCT& a, RGY_PICSTRUCT b) {
    a = a | b;
    return a;
}

static RGY_PICSTRUCT operator&(RGY_PICSTRUCT a, RGY_PICSTRUCT b) {
    return (RGY_PICSTRUCT)((uint8_t)a & (uint8_t)b);
}

static RGY_PICSTRUCT operator&=(RGY_PICSTRUCT& a, RGY_PICSTRUCT b) {
    a = (RGY_PICSTRUCT)((uint8_t)a & (uint8_t)b);
    return a;
}

typedef struct ConvertCSP {
    RGY_CSP csp_from, csp_to;
    bool uv_only;
    funcConvertCSP func[2];
    unsigned int simd;
} ConvertCSP;

const ConvertCSP *get_convert_csp_func(RGY_CSP csp_from, RGY_CSP csp_to, bool uv_only);
const TCHAR *get_simd_str(unsigned int simd);

enum RGY_FRAME_FLAGS : uint64_t {
    RGY_FRAME_FLAG_NONE     = 0x00u,
    RGY_FRAME_FLAG_RFF      = 0x01u,
    RGY_FRAME_FLAG_RFF_COPY = 0x02u,
    RGY_FRAME_FLAG_RFF_TFF  = 0x04u,
    RGY_FRAME_FLAG_RFF_BFF  = 0x08u,
};

static RGY_FRAME_FLAGS operator|(RGY_FRAME_FLAGS a, RGY_FRAME_FLAGS b) {
    return (RGY_FRAME_FLAGS)((uint64_t)a | (uint64_t)b);
}

static RGY_FRAME_FLAGS operator|=(RGY_FRAME_FLAGS& a, RGY_FRAME_FLAGS b) {
    a = a | b;
    return a;
}

static RGY_FRAME_FLAGS operator&(RGY_FRAME_FLAGS a, RGY_FRAME_FLAGS b) {
    return (RGY_FRAME_FLAGS)((uint64_t)a & (uint64_t)b);
}

static RGY_FRAME_FLAGS operator&=(RGY_FRAME_FLAGS& a, RGY_FRAME_FLAGS b) {
    a = a & b;
    return a;
}

static RGY_FRAME_FLAGS operator~(RGY_FRAME_FLAGS a) {
    return (RGY_FRAME_FLAGS)(~((uint64_t)a));
}

struct FrameInfo {
    uint8_t *ptr;
    RGY_CSP csp;
    int width, height, pitch;
    int64_t timestamp;
    int64_t duration;
    bool deivce_mem;
    RGY_PICSTRUCT picstruct;
    RGY_FRAME_FLAGS flags;
};

static bool cmpFrameInfoCspResolution(const FrameInfo *pA, const FrameInfo *pB) {
    return pA->csp != pB->csp
        || pA->width != pB->width
        || pA->height != pB->height
        || pA->deivce_mem != pB->deivce_mem
        || pA->pitch != pB->pitch;
}

struct FrameInfoExtra {
    int width_byte, height_total, frame_size;
};

static bool interlaced(const FrameInfo& FrameInfo) {
    return (FrameInfo.picstruct & RGY_PICSTRUCT_INTERLACED) != 0;
}

FrameInfoExtra getFrameInfoExtra(const FrameInfo *pFrameInfo);

FrameInfo getPlane(const FrameInfo *frameInfo, const RGY_PLANE plane);

#endif //_CONVERT_CSP_H_
