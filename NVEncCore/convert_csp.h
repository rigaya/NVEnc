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
#include <vector>
#include <memory>
#include <string>
#include "rgy_tchar.h"
#include "rgy_simd.h"

#if defined(_MSC_VER)
#ifndef RGY_FORCEINLINE
#define RGY_FORCEINLINE __forceinline
#endif
#ifndef RGY_NOINLINE
#define RGY_NOINLINE __declspec(noinline)
#endif
#else
#ifndef RGY_FORCEINLINE
#define RGY_FORCEINLINE inline
#endif
#ifndef RGY_NOINLINE
#define RGY_NOINLINE __attribute__ ((noinline))
#endif
#endif

enum RGY_PLANE {
    RGY_PLANE_Y,
    RGY_PLANE_R = RGY_PLANE_Y,
    RGY_PLANE_C,
    RGY_PLANE_U = RGY_PLANE_C,
    RGY_PLANE_G = RGY_PLANE_U,
    RGY_PLANE_V,
    RGY_PLANE_B = RGY_PLANE_V,
    RGY_PLANE_A
};

typedef void (*funcConvertCSP) (void **dst, const void **src, int width, int src_y_pitch_byte, int src_uv_pitch_byte, int dst_y_pitch_byte, int height, int dst_height, int thread_id, int thread_n, int *crop);

enum RGY_CSP {
    RGY_CSP_NA = 0,
    RGY_CSP_NV12,
    RGY_CSP_YV12,
    RGY_CSP_YUY2,
    RGY_CSP_YUV422,
    RGY_CSP_NV16,
    RGY_CSP_NV24,
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
    RGY_CSP_YUV444_32,
    RGY_CSP_YUVA444,
    RGY_CSP_YUVA444_16,
    RGY_CSP_AYUV,
    RGY_CSP_AYUV_16,
    RGY_CSP_Y210,   //YUY2 layout
    RGY_CSP_Y216,   //YUY2 layout
    RGY_CSP_Y410,   //packed
    RGY_CSP_Y416,   //packed
    RGY_CSP_RGB24R, //packed
    RGY_CSP_RGB32R, //packed
    RGY_CSP_RGB24, //packed
    RGY_CSP_RGB32, //packed
    RGY_CSP_BGR24, //packed
    RGY_CSP_BGR32, //packed
    RGY_CSP_RGB,   //planar
    RGY_CSP_RGBA,  //planar
    RGY_CSP_GBR,   //planar
    RGY_CSP_GBRA,  //planar
    RGY_CSP_RGB_16,   //planar
    RGY_CSP_RGBA_16,  //planar
    RGY_CSP_BGR_16,   //planar
    RGY_CSP_BGRA_16,  //planar
    RGY_CSP_RGB_F32,  //planar
    RGY_CSP_RGBA_F32, //planar
    RGY_CSP_BGR_F32,  //planar
    RGY_CSP_BGRA_F32, //planar
    RGY_CSP_YC48,
    RGY_CSP_Y8,
    RGY_CSP_Y16,
    RGY_CSP_COUNT
};

static const TCHAR *RGY_CSP_NAMES[] = {
    _T("Invalid"),
    _T("nv12"),
    _T("yv12"),
    _T("yuy2"),
    _T("yuv422"),
    _T("nv16"),
    _T("nv24"),
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
    _T("yuv444(32bit)"),
    _T("yuva444"),
    _T("yuva444(16bit)"),
    _T("ayuv444"),
    _T("ayuv444(16bit)"),
    _T("y210"),
    _T("y216"),
    _T("y410"),
    _T("y416"),
    _T("rgb24r"),
    _T("rgb32r"),
    _T("rgb24"),
    _T("rgb32"),
    _T("bgr24"),
    _T("bgr32"),
    _T("rgb"),
    _T("rgba"),
    _T("gbr"),
    _T("gbra"),
    _T("rgb(16bit)"),
    _T("rgba(16bit)"),
    _T("bgr(16bit)"),
    _T("bgra(16bit)"),
    _T("rgb(fp32)"),
    _T("rgba(fp32)"),
    _T("bgr(fp32)"),
    _T("bgra(fp32)"),
    _T("yc48"),
    _T("y8"),
    _T("yc16")
};
static_assert(sizeof(RGY_CSP_NAMES)/sizeof(RGY_CSP_NAMES[0]) == RGY_CSP_COUNT, "_countof(RGY_CSP_NAMES) == RGY_CSP_COUNT");

static const uint8_t RGY_CSP_BIT_DEPTH[] = {
     0, //RGY_CSP_NA
     8, //RGY_CSP_NV12
     8, //RGY_CSP_YV12
     8, //RGY_CSP_YUY2
     8, //RGY_CSP_YUV422
     8, //RGY_CSP_NV16
     8, //RGY_CSP_NV24
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
    32, //RGY_CSP_YUV444_32
     8, //RGY_CSP_YUVA444
    16, //RGY_CSP_YUVA444_16
     8, //RGY_CSP_AYUV444
    16, //RGY_CSP_AYUV444_16
    10, //RGY_CSP_Y210
    16, //RGY_CSP_Y216
    10, //RGY_CSP_Y410
    16, //RGY_CSP_Y416
     8, //RGY_CSP_RGB24R
     8, //RGY_CSP_RGB32R
     8, //RGY_CSP_RGB24
     8, //RGY_CSP_RGB32
     8, //RGY_CSP_BGR24
     8, //RGY_CSP_BGR32
     8, //RGY_CSP_RGB
     8, //RGY_CSP_RGBA
     8, //RGY_CSP_GBR
     8, //RGY_CSP_GBRA
    16, //RGY_CSP_RGB_16
    16, //RGY_CSP_RGBA_16
    16, //RGY_CSP_BGR_16
    16, //RGY_CSP_BGRA_16
    32, //RGY_CSP_RGB_F32
    32, //RGY_CSP_RGBA_F32
    32, //RGY_CSP_BGR_F32
    32, //RGY_CSP_BGRA_F32
    10, //RGY_CSP_YC48
     8, //RGY_CSP_Y8
    16, //RGY_CSP_Y16
};
static_assert(sizeof(RGY_CSP_BIT_DEPTH) / sizeof(RGY_CSP_BIT_DEPTH[0]) == RGY_CSP_COUNT, "_countof(RGY_CSP_BIT_DEPTH) == RGY_CSP_COUNT");

static const uint8_t RGY_CSP_PLANES[] = {
     0, //RGY_CSP_NA
     2, //RGY_CSP_NV12
     3, //RGY_CSP_YV12
     1, //RGY_CSP_YUY2
     3, //RGY_CSP_YUV422
     2, //RGY_CSP_NV16
     2, //RGY_CSP_NV24
     3, //RGY_CSP_YUV444
     3, //RGY_CSP_YV12_09
     3,
     3,
     3,
     3, //RGY_CSP_YV12_16
     2, //RGY_CSP_P010
     3, //RGY_CSP_YUV422_09
     3,
     3,
     3,
     3, //RGY_CSP_YUV422_16
     2, //RGY_CSP_P210
     3, //RGY_CSP_YUV444_09
     3,
     3,
     3,
     3, //RGY_CSP_YUV444_16
     3, //RGY_CSP_YUV444_32
     4, //RGY_CSP_YUVA444
     4, //RGY_CSP_YUVA444_16
     1, //RGY_CSP_AYUV444
     1, //RGY_CSP_AYUV444_16
     1, //RGY_CSP_Y210
     1, //RGY_CSP_Y216
     1, //RGY_CSP_Y410
     1, //RGY_CSP_Y416
     1, //RGY_CSP_RGB24R
     1, //RGY_CSP_RGB32R
     1, //RGY_CSP_RGB24
     1, //RGY_CSP_RGB32
     1, //RGY_CSP_BGR24
     1, //RGY_CSP_BGR32
     3, //RGY_CSP_RGB
     4, //RGY_CSP_RGBA
     3, //RGY_CSP_GBR
     4, //RGY_CSP_GBRA
     3, //RGY_CSP_RGB_16
     4, //RGY_CSP_RGBA_16
     3, //RGY_CSP_BGR_16
     4, //RGY_CSP_BGRA_16
     3, //RGY_CSP_RGB_F32
     4, //RGY_CSP_RGBA_F32
     3, //RGY_CSP_BGR_F32
     4, //RGY_CSP_BGRA_F32
     1, //RGY_CSP_YC48
     1, //RGY_CSP_Y8
     1, //RGY_CSP_Y16
};
static_assert(sizeof(RGY_CSP_PLANES) / sizeof(RGY_CSP_PLANES[0]) == RGY_CSP_COUNT, "_countof(RGY_CSP_PLANES) == RGY_CSP_COUNT");

enum RGY_CHROMAFMT {
    RGY_CHROMAFMT_UNKNOWN = 0,
    RGY_CHROMAFMT_MONOCHROME = 0,
    RGY_CHROMAFMT_YUV420,
    RGY_CHROMAFMT_YUV422,
    RGY_CHROMAFMT_YUV444,
    RGY_CHROMAFMT_YUVA444,
    RGY_CHROMAFMT_RGB_PACKED,
    RGY_CHROMAFMT_RGB,
    RGY_CHROMAFMT_COUNT,
};

static const TCHAR *RGY_CHROMAFMT_NAMES[] = {
    _T("Monochrome"),
    _T("yuv420"),
    _T("yuv422"),
    _T("yuv444"),
    _T("yuva444"),
    _T("rgbp"),
    _T("rgb")
};
static_assert(sizeof(RGY_CHROMAFMT_NAMES) / sizeof(RGY_CHROMAFMT_NAMES[0]) == RGY_CHROMAFMT_COUNT, "_countof(RGY_CHROMAFMT_NAMES) == RGY_CHROMAFMT_COUNT");

static const RGY_CHROMAFMT RGY_CSP_CHROMA_FORMAT[] = {
    RGY_CHROMAFMT_UNKNOWN, //RGY_CSP_NA
    RGY_CHROMAFMT_YUV420, //RGY_CSP_NV12
    RGY_CHROMAFMT_YUV420, //RGY_CSP_YV12
    RGY_CHROMAFMT_YUV422, //RGY_CSP_YUY2
    RGY_CHROMAFMT_YUV422, //RGY_CSP_YUV422
    RGY_CHROMAFMT_YUV422, //RGY_CSP_NV16
    RGY_CHROMAFMT_YUV444, //RGY_CSP_NV24
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
    RGY_CHROMAFMT_YUV444, //RGY_CSP_YUV444_32
    RGY_CHROMAFMT_YUVA444, //RGY_CSP_YUVA444
    RGY_CHROMAFMT_YUVA444, //RGY_CSP_YUVA444_16
    RGY_CHROMAFMT_YUVA444, //RGY_CSP_YUVA444
    RGY_CHROMAFMT_YUVA444, //RGY_CSP_YUVA444_16
    RGY_CHROMAFMT_YUV422, //RGY_CSP_Y210
    RGY_CHROMAFMT_YUV422, //RGY_CSP_Y216
    RGY_CHROMAFMT_YUV444, //RGY_CSP_Y410
    RGY_CHROMAFMT_YUV444, //RGY_CSP_Y416
    RGY_CHROMAFMT_RGB_PACKED,
    RGY_CHROMAFMT_RGB_PACKED,
    RGY_CHROMAFMT_RGB_PACKED,
    RGY_CHROMAFMT_RGB_PACKED,
    RGY_CHROMAFMT_RGB_PACKED,
    RGY_CHROMAFMT_RGB_PACKED,
    RGY_CHROMAFMT_RGB,
    RGY_CHROMAFMT_RGB,
    RGY_CHROMAFMT_RGB,
    RGY_CHROMAFMT_RGB,
    RGY_CHROMAFMT_RGB,
    RGY_CHROMAFMT_RGB,
    RGY_CHROMAFMT_RGB,
    RGY_CHROMAFMT_RGB,
    RGY_CHROMAFMT_RGB,
    RGY_CHROMAFMT_RGB,
    RGY_CHROMAFMT_RGB,
    RGY_CHROMAFMT_RGB,
    RGY_CHROMAFMT_YUV444, //RGY_CSP_YC48
    RGY_CHROMAFMT_MONOCHROME,
    RGY_CHROMAFMT_MONOCHROME,
};
static_assert(sizeof(RGY_CSP_CHROMA_FORMAT)/sizeof(RGY_CSP_CHROMA_FORMAT[0]) == RGY_CSP_COUNT, "_countof(RGY_CSP_CHROMA_FORMAT) == RGY_CSP_COUNT");

static const uint8_t RGY_CSP_BIT_PER_PIXEL[] = {
     0, //RGY_CSP_NA
    12, //RGY_CSP_NV12
    12, //RGY_CSP_YV12
    16, //RGY_CSP_YUY2
    16, //RGY_CSP_YUV422
    16, //RGY_CSP_NV16
    24, //RGY_CSP_NV24
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
    96, //RGY_CSP_YUV444_32
    32, //RGY_CSP_YUVA444
    64, //RGY_CSP_YUVA444_16
    32, //RGY_CSP_AYUV444
    64, //RGY_CSP_AYUV444_16
    32, //RGY_CSP_Y210
    32, //RGY_CSP_Y216
    48, //RGY_CSP_Y410
    48, //RGY_CSP_Y416
    24, //RGY_CSP_RGB24R
    24, //RGY_CSP_RGB32R
    24, //RGY_CSP_RGB24
    32, //RGY_CSP_RGB32
    24, //RGY_CSP_BGR24
    32, //RGY_CSP_BGR32
    24, //RGY_CSP_RGB
    32, //RGY_CSP_RGBA
    24, //RGY_CSP_GBR
    32, //RGY_CSP_GBRA
    48, //RGY_CSP_RGB_16
    64, //RGY_CSP_RGBA_16
    48, //RGY_CSP_BGR_16
    64, //RGY_CSP_BGRA_16
    96, //RGY_CSP_RGB_F32
   128, //RGY_CSP_RGBA_F32
    96, //RGY_CSP_BGR_F32
   128, //RGY_CSP_BGRA_F32
    48, //RGY_CSP_YC48
     8, //RGY_CSP_Y8
    16, //RGY_CSP_Y16
};
static_assert(sizeof(RGY_CSP_BIT_PER_PIXEL) / sizeof(RGY_CSP_BIT_PER_PIXEL[0]) == RGY_CSP_COUNT, "_countof(RGY_CSP_BIT_PER_PIXEL) == RGY_CSP_COUNT");

static bool cspShiftUsed(const RGY_CSP csp) {
    return csp == RGY_CSP_P010
        || csp == RGY_CSP_P210
        || csp == RGY_CSP_Y210
        || csp == RGY_CSP_Y416;
}

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
    RGY_PICSTRUCT_AUTO         = 0xffffffff,
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

const TCHAR *picstrcut_to_str(RGY_PICSTRUCT picstruct);

typedef struct ConvertCSP {
    RGY_CSP csp_from, csp_to;
    bool uv_only;
    funcConvertCSP func[2];
    RGY_SIMD simd;
} ConvertCSP;

const ConvertCSP *get_convert_csp_func(RGY_CSP csp_from, RGY_CSP csp_to, bool uv_only, RGY_SIMD simd);
const TCHAR *get_simd_str(RGY_SIMD simd);

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

static const int RGY_MAX_PLANES = 4;

enum RGY_MEM_TYPE {
    RGY_MEM_TYPE_CPU = 0,
    RGY_MEM_TYPE_GPU,
    RGY_MEM_TYPE_GPU_IMAGE,            // VCEのImage
    RGY_MEM_TYPE_GPU_IMAGE_NORMALIZED, // QSVのImage
    RGY_MEM_TYPE_MPP,
};
const TCHAR *get_memtype_str(RGY_MEM_TYPE type);

class RGYFrameData;

struct RGYFrameInfo {
    uint8_t *ptr;
    RGY_CSP csp;
    int width, height, pitch;
    int bitdepth;
    int64_t timestamp;
    int64_t duration;
    bool deivce_mem;
    RGY_PICSTRUCT picstruct;
    RGY_FRAME_FLAGS flags;
    int inputFrameId;
    std::vector<std::shared_ptr<RGYFrameData>> dataList;

    RGYFrameInfo() :
        ptr(nullptr),
        csp(RGY_CSP_NA),
        width(0),
        height(0),
        pitch(0),
        bitdepth(0),
        timestamp(0),
        duration(0),
        deivce_mem(false),
        picstruct(RGY_PICSTRUCT_UNKNOWN),
        flags(RGY_FRAME_FLAG_NONE),
        inputFrameId(-1),
        dataList() {};

    std::basic_string<TCHAR> print() const;
};

static bool cmpFrameInfoCspResolution(const RGYFrameInfo *pA, const RGYFrameInfo *pB) {
    return pA->csp != pB->csp
        || pA->width != pB->width
        || pA->height != pB->height
        || pA->deivce_mem != pB->deivce_mem;
}

static void copyFrameProp(RGYFrameInfo *dst, const RGYFrameInfo *src) {
    dst->width = src->width;
    dst->height = src->height;
    dst->csp = src->csp;
    dst->picstruct = src->picstruct;
    dst->timestamp = src->timestamp;
    dst->duration = src->duration;
    dst->inputFrameId = src->inputFrameId;
    dst->flags = src->flags;
}

struct FrameInfoExtra {
    int width_byte, height_total, frame_size;
};

static bool interlaced(const RGYFrameInfo& RGYFrameInfo) {
    return (RGYFrameInfo.picstruct & RGY_PICSTRUCT_INTERLACED) != 0;
}

FrameInfoExtra getFrameInfoExtra(const RGYFrameInfo *pFrameInfo);

RGYFrameInfo getPlane(const RGYFrameInfo *frameInfo, const RGY_PLANE plane);

#pragma warning(push)
#pragma warning(disable: 4201)
typedef union sInputCrop {
    struct {
        int left, up, right, bottom;
    } e;
    int c[4];
} sInputCrop;
#pragma warning(pop)

static inline bool cropEnabled(const sInputCrop &crop) {
    return 0 != (crop.c[0] | crop.c[1] | crop.c[2] | crop.c[3]);
}

static inline sInputCrop initCrop() {
    sInputCrop crop;
    for (int i = 0; i < 4; i++) {
        crop.c[i] = 0;
    }
    return crop;
}

static sInputCrop getPlane(const sInputCrop *crop, const RGY_CSP csp, const RGY_PLANE plane) {
    sInputCrop planeCrop = *crop;
    if (plane == RGY_PLANE_Y
        || csp == RGY_CSP_YUY2
        || csp == RGY_CSP_Y210 || csp == RGY_CSP_Y216
        || csp == RGY_CSP_Y410 || csp == RGY_CSP_Y416
        || RGY_CSP_CHROMA_FORMAT[csp] == RGY_CHROMAFMT_RGB
        || RGY_CSP_CHROMA_FORMAT[csp] == RGY_CHROMAFMT_RGB_PACKED
        || RGY_CSP_CHROMA_FORMAT[csp] == RGY_CHROMAFMT_YUV444
        || RGY_CSP_CHROMA_FORMAT[csp] == RGY_CHROMAFMT_MONOCHROME
        || RGY_CSP_PLANES[csp] == 1) {
        return planeCrop;
    }
    if (csp == RGY_CSP_NV12 || csp == RGY_CSP_P010) {
        planeCrop.e.up >>= 1;
        planeCrop.e.bottom >>= 1;
    } else if (csp == RGY_CSP_NV16 || csp == RGY_CSP_P210) {
        ;
    } else if (RGY_CSP_CHROMA_FORMAT[csp] == RGY_CHROMAFMT_YUV420) {
        planeCrop.e.up >>= 1;
        planeCrop.e.bottom >>= 1;
        planeCrop.e.left >>= 1;
        planeCrop.e.right >>= 1;
    } else if (RGY_CSP_CHROMA_FORMAT[csp] == RGY_CHROMAFMT_YUV422) {
        planeCrop.e.left >>= 1;
        planeCrop.e.right >>= 1;
    }
    return planeCrop;
}

struct THREAD_Y_RANGE {
    int start_src;
    int start_dst;
    int len;
};

static inline THREAD_Y_RANGE thread_y_range(int y_start, int y_end, int thread_id, int thread_n) {
    const int h = y_end - y_start;
    THREAD_Y_RANGE y_range;
    int y0 = ((((h *  thread_id)    / thread_n) + 3) & ~3);
    int y1 = ((((h * (thread_id+1)) / thread_n) + 3) & ~3);
    if (y1 > h) {
        y1 = h;
    }
    y_range.start_src = y_start + y0;
    y_range.start_dst = y0;
    y_range.len = y1 - y0;
    return y_range;
}

#ifdef __CUDACC__
#define CU_DEV_HOST_CODE __device__ __host__
#elif defined(_MSC_VER)
#define CU_DEV_HOST_CODE __forceinline
#else
#define CU_DEV_HOST_CODE inline
#endif

template<int in_bit_depth, int out_bit_depth, int shift_offset>
CU_DEV_HOST_CODE
static int conv_bit_depth_lsft() {
    const int lsft = out_bit_depth - (in_bit_depth + shift_offset);
    return lsft < 0 ? 0 : lsft;
}

template<int in_bit_depth, int out_bit_depth, int shift_offset>
CU_DEV_HOST_CODE
static int conv_bit_depth_rsft() {
    const int rsft = in_bit_depth + shift_offset - out_bit_depth;
    return rsft < 0 ? 0 : rsft;
}

template<int in_bit_depth, int out_bit_depth, int shift_offset>
CU_DEV_HOST_CODE
static int conv_bit_depth_rsft_add() {
    const int rsft = conv_bit_depth_rsft<in_bit_depth, out_bit_depth, shift_offset>();
    return (rsft - 1 >= 0) ? 1 << (rsft - 1) : 0;
}

template<int in_bit_depth, int out_bit_depth, int shift_offset>
CU_DEV_HOST_CODE
static int conv_bit_depth(int c) {
    if (out_bit_depth > in_bit_depth + shift_offset) {
        return c << conv_bit_depth_lsft<in_bit_depth, out_bit_depth, shift_offset>();
    } else if (out_bit_depth < in_bit_depth + shift_offset) {
        const int x = (c + conv_bit_depth_rsft_add<in_bit_depth, out_bit_depth, shift_offset>()) >> conv_bit_depth_rsft<in_bit_depth, out_bit_depth, shift_offset>();
        const int low = 0;
        const int high = (1 << out_bit_depth) - 1;
        return (((x) <= (high)) ? (((x) >= (low)) ? (x) : (low)) : (high));
    } else {
        return c;
    }
}

#endif //_CONVERT_CSP_H_
