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

#include "NVEncUtil.h"

static const auto RGY_CODEC_TO_NVENC = make_array<std::pair<RGY_CODEC, cudaVideoCodec>>(
    std::make_pair(RGY_CODEC_H264,  cudaVideoCodec_H264),
    std::make_pair(RGY_CODEC_HEVC,  cudaVideoCodec_HEVC),
    std::make_pair(RGY_CODEC_MPEG1, cudaVideoCodec_MPEG1),
    std::make_pair(RGY_CODEC_MPEG2, cudaVideoCodec_MPEG2),
    std::make_pair(RGY_CODEC_MPEG4, cudaVideoCodec_MPEG4),
    std::make_pair(RGY_CODEC_VP8,   cudaVideoCodec_VP8),
    std::make_pair(RGY_CODEC_VP9,   cudaVideoCodec_VP9),
    std::make_pair(RGY_CODEC_VC1,   cudaVideoCodec_VC1)
    );

MAP_PAIR_0_1(codec, rgy, RGY_CODEC, enc, cudaVideoCodec, RGY_CODEC_TO_NVENC, RGY_CODEC_UNKNOWN, cudaVideoCodec_NumCodecs);

static const auto RGY_CHROMAFMT_TO_NVENC = make_array<std::pair<RGY_CHROMAFMT, cudaVideoChromaFormat>>(
    std::make_pair(RGY_CHROMAFMT_MONOCHROME, cudaVideoChromaFormat_Monochrome),
    std::make_pair(RGY_CHROMAFMT_YUV420,     cudaVideoChromaFormat_420),
    std::make_pair(RGY_CHROMAFMT_YUV422,     cudaVideoChromaFormat_422),
    std::make_pair(RGY_CHROMAFMT_YUV444,     cudaVideoChromaFormat_444)
    );

MAP_PAIR_0_1(chromafmt, rgy, RGY_CHROMAFMT, enc, cudaVideoChromaFormat, RGY_CHROMAFMT_TO_NVENC, RGY_CHROMAFMT_UNKNOWN, cudaVideoChromaFormat_Monochrome);

NV_ENC_PIC_STRUCT picstruct_rgy_to_enc(RGY_PICSTRUCT picstruct) {
    if (picstruct & RGY_PICSTRUCT_TFF) return NV_ENC_PIC_STRUCT_FIELD_TOP_BOTTOM;
    if (picstruct & RGY_PICSTRUCT_BFF) return NV_ENC_PIC_STRUCT_FIELD_BOTTOM_TOP;
    return NV_ENC_PIC_STRUCT_FRAME;
}

RGY_CSP getEncCsp(NV_ENC_BUFFER_FORMAT enc_buffer_format) {
    switch (enc_buffer_format) {
    case NV_ENC_BUFFER_FORMAT_NV12:
        return RGY_CSP_NV12;
    case NV_ENC_BUFFER_FORMAT_YV12:
    case NV_ENC_BUFFER_FORMAT_IYUV:
        return RGY_CSP_YV12;
    case NV_ENC_BUFFER_FORMAT_YUV444:
        return RGY_CSP_YUV444;
    case NV_ENC_BUFFER_FORMAT_YUV420_10BIT:
        return RGY_CSP_P010;
    case NV_ENC_BUFFER_FORMAT_YUV444_10BIT:
        return RGY_CSP_YUV444_16;
    case NV_ENC_BUFFER_FORMAT_UNDEFINED:
    default:
        return RGY_CSP_NA;
    }
}
