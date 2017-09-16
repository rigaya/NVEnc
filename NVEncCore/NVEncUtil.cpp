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
#include "NvHWEncoder.h"

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

static const GUID GUID_EMPTY = { 0 };

static const auto RGY_CODEC_TO_GUID = make_array<std::pair<RGY_CODEC, GUID>>(
    std::make_pair(RGY_CODEC_H264, NV_ENC_CODEC_H264_GUID),
    std::make_pair(RGY_CODEC_HEVC, NV_ENC_CODEC_HEVC_GUID)
    );

MAP_PAIR_0_1(codec_guid, rgy, RGY_CODEC, enc, GUID, RGY_CODEC_TO_GUID, RGY_CODEC_UNKNOWN, GUID_EMPTY);

static const auto RGY_CODEC_PROFILE_TO_GUID = make_array<std::pair<RGY_CODEC_DATA, GUID>>(
    std::make_pair(RGY_CODEC_DATA(RGY_CODEC_H264, 77),  NV_ENC_H264_PROFILE_BASELINE_GUID),
    std::make_pair(RGY_CODEC_DATA(RGY_CODEC_H264, 88),  NV_ENC_H264_PROFILE_MAIN_GUID),
    std::make_pair(RGY_CODEC_DATA(RGY_CODEC_H264, 100), NV_ENC_H264_PROFILE_HIGH_GUID),
    std::make_pair(RGY_CODEC_DATA(RGY_CODEC_H264, 144), NV_ENC_H264_PROFILE_HIGH_444_GUID),
    std::make_pair(RGY_CODEC_DATA(RGY_CODEC_HEVC, 1),   NV_ENC_HEVC_PROFILE_MAIN_GUID),
    std::make_pair(RGY_CODEC_DATA(RGY_CODEC_HEVC, 2),   NV_ENC_HEVC_PROFILE_MAIN10_GUID),
    std::make_pair(RGY_CODEC_DATA(RGY_CODEC_HEVC, 3),   NV_ENC_HEVC_PROFILE_FREXT_GUID)
    );

MAP_PAIR_0_1(codec_guid_profile, rgy, RGY_CODEC_DATA, enc, GUID, RGY_CODEC_PROFILE_TO_GUID, RGY_CODEC_DATA(), GUID_EMPTY);

static const auto RGY_CHROMAFMT_TO_NVENC = make_array<std::pair<RGY_CHROMAFMT, cudaVideoChromaFormat>>(
    std::make_pair(RGY_CHROMAFMT_MONOCHROME, cudaVideoChromaFormat_Monochrome),
    std::make_pair(RGY_CHROMAFMT_YUV420,     cudaVideoChromaFormat_420),
    std::make_pair(RGY_CHROMAFMT_YUV422,     cudaVideoChromaFormat_422),
    std::make_pair(RGY_CHROMAFMT_YUV444,     cudaVideoChromaFormat_444)
    );

MAP_PAIR_0_1(chromafmt, rgy, RGY_CHROMAFMT, enc, cudaVideoChromaFormat, RGY_CHROMAFMT_TO_NVENC, RGY_CHROMAFMT_UNKNOWN, cudaVideoChromaFormat_Monochrome);

static const auto RGY_CSP_TO_NVENC = make_array<std::pair<RGY_CSP, NV_ENC_BUFFER_FORMAT>>(
    std::make_pair(RGY_CSP_NA,        NV_ENC_BUFFER_FORMAT_UNDEFINED),
    std::make_pair(RGY_CSP_NV12,      NV_ENC_BUFFER_FORMAT_NV12),
    std::make_pair(RGY_CSP_YV12,      NV_ENC_BUFFER_FORMAT_YV12),
    std::make_pair(RGY_CSP_YUY2,      NV_ENC_BUFFER_FORMAT_UNDEFINED),
    std::make_pair(RGY_CSP_YUV422,    NV_ENC_BUFFER_FORMAT_UNDEFINED),
    std::make_pair(RGY_CSP_YUV444,    NV_ENC_BUFFER_FORMAT_UNDEFINED),
    std::make_pair(RGY_CSP_YV12_09,   NV_ENC_BUFFER_FORMAT_UNDEFINED),
    std::make_pair(RGY_CSP_YV12_10,   NV_ENC_BUFFER_FORMAT_YUV420_10BIT),
    std::make_pair(RGY_CSP_YV12_12,   NV_ENC_BUFFER_FORMAT_UNDEFINED),
    std::make_pair(RGY_CSP_YV12_14,   NV_ENC_BUFFER_FORMAT_UNDEFINED),
    std::make_pair(RGY_CSP_YV12_16,   NV_ENC_BUFFER_FORMAT_UNDEFINED),
    std::make_pair(RGY_CSP_P010,      NV_ENC_BUFFER_FORMAT_YUV420_10BIT),
    std::make_pair(RGY_CSP_YUV422_09, NV_ENC_BUFFER_FORMAT_UNDEFINED),
    std::make_pair(RGY_CSP_YUV422_10, NV_ENC_BUFFER_FORMAT_UNDEFINED),
    std::make_pair(RGY_CSP_YUV422_12, NV_ENC_BUFFER_FORMAT_UNDEFINED),
    std::make_pair(RGY_CSP_YUV422_14, NV_ENC_BUFFER_FORMAT_UNDEFINED),
    std::make_pair(RGY_CSP_YUV422_16, NV_ENC_BUFFER_FORMAT_UNDEFINED),
    std::make_pair(RGY_CSP_P210,      NV_ENC_BUFFER_FORMAT_UNDEFINED),
    std::make_pair(RGY_CSP_YUV444_09, NV_ENC_BUFFER_FORMAT_UNDEFINED),
    std::make_pair(RGY_CSP_YUV444_10, NV_ENC_BUFFER_FORMAT_YUV444_10BIT),
    std::make_pair(RGY_CSP_YUV444_12, NV_ENC_BUFFER_FORMAT_UNDEFINED),
    std::make_pair(RGY_CSP_YUV444_14, NV_ENC_BUFFER_FORMAT_UNDEFINED),
    std::make_pair(RGY_CSP_YUV444_16, NV_ENC_BUFFER_FORMAT_UNDEFINED),
    std::make_pair(RGY_CSP_RGB24R,    NV_ENC_BUFFER_FORMAT_UNDEFINED),
    std::make_pair(RGY_CSP_RGB32R,    NV_ENC_BUFFER_FORMAT_UNDEFINED),
    std::make_pair(RGY_CSP_RGB24,     NV_ENC_BUFFER_FORMAT_UNDEFINED),
    std::make_pair(RGY_CSP_RGB32,     NV_ENC_BUFFER_FORMAT_ARGB),
    std::make_pair(RGY_CSP_YC48,      NV_ENC_BUFFER_FORMAT_UNDEFINED)
    );

MAP_PAIR_0_1(csp, rgy, RGY_CSP, enc, NV_ENC_BUFFER_FORMAT, RGY_CSP_TO_NVENC, RGY_CSP_NA, NV_ENC_BUFFER_FORMAT_UNDEFINED);

__declspec(noinline)
NV_ENC_PIC_STRUCT picstruct_rgy_to_enc(RGY_PICSTRUCT picstruct) {
    if (picstruct & RGY_PICSTRUCT_TFF) return NV_ENC_PIC_STRUCT_FIELD_TOP_BOTTOM;
    if (picstruct & RGY_PICSTRUCT_BFF) return NV_ENC_PIC_STRUCT_FIELD_BOTTOM_TOP;
    return NV_ENC_PIC_STRUCT_FRAME;
}

__declspec(noinline)
RGY_PICSTRUCT picstruct_enc_to_rgy(NV_ENC_PIC_STRUCT picstruct) {
    if (picstruct == NV_ENC_PIC_STRUCT_FIELD_TOP_BOTTOM) return RGY_PICSTRUCT_FRAME_TFF;
    if (picstruct == NV_ENC_PIC_STRUCT_FIELD_BOTTOM_TOP) return RGY_PICSTRUCT_FRAME_BFF;
    return RGY_PICSTRUCT_FRAME;
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
    case NV_ENC_BUFFER_FORMAT_ARGB:
        return RGY_CSP_RGB32;
    case NV_ENC_BUFFER_FORMAT_UNDEFINED:
    default:
        return RGY_CSP_NA;
    }
}

__declspec(noinline)
VideoInfo videooutputinfo(
    const GUID& encCodecGUID,
    NV_ENC_BUFFER_FORMAT buffer_fmt,
    int nEncWidth,
    int nEncHeight,
    const NV_ENC_CONFIG *pEncConfig,
    NV_ENC_PIC_STRUCT nPicStruct,
    std::pair<int, int> sar,
    std::pair<int, int> outFps) {

    VideoInfo info;
    memset(&info, 0, sizeof(info));
    info.codec = codec_guid_enc_to_rgy(encCodecGUID);
    info.codecLevel = (info.codec == RGY_CODEC_H264) ? pEncConfig->encodeCodecConfig.h264Config.level : pEncConfig->encodeCodecConfig.hevcConfig.level;
    info.codecProfile = codec_guid_profile_enc_to_rgy(pEncConfig->profileGUID).codecProfile;
    info.videoDelay = ((pEncConfig->frameIntervalP - 2) > 0) + (((pEncConfig->frameIntervalP - 2) > 2));
    info.dstWidth = nEncWidth;
    info.dstHeight = nEncHeight;
    info.fpsN = outFps.first;
    info.fpsD = outFps.second;
    info.sar[0] = sar.first;
    info.sar[1] = sar.second;
    info.picstruct = picstruct_enc_to_rgy(nPicStruct);
    info.csp = csp_enc_to_rgy(buffer_fmt);

    const NV_ENC_CONFIG_H264_VUI_PARAMETERS& videoSignalInfo = (info.codec == RGY_CODEC_H264)
        ? pEncConfig->encodeCodecConfig.h264Config.h264VUIParameters
        : pEncConfig->encodeCodecConfig.hevcConfig.hevcVUIParameters;

    info.vui.descriptpresent = videoSignalInfo.colourDescriptionPresentFlag;
    info.vui.colorprim = videoSignalInfo.colourPrimaries;
    info.vui.matrix = videoSignalInfo.colourMatrix;
    info.vui.transfer = videoSignalInfo.transferCharacteristics;
    info.vui.fullrange = videoSignalInfo.videoFullRangeFlag;
    info.vui.format = videoSignalInfo.videoFormat;
    return info;
}

#if !ENABLE_AVSW_READER
#define TTMATH_NOASM
#include "ttmath/ttmath.h"

int64_t rational_rescale(int64_t v, rgy_rational<int> from, rgy_rational<int> to) {
    auto mul = rgy_rational<int64_t>((int64_t)from.n() * (int64_t)to.d(), (int64_t)from.d() * (int64_t)to.n());

#if _M_IX86
#define RESCALE_INT_SIZE 4
#else
#define RESCALE_INT_SIZE 2
#endif
    ttmath::Int<RESCALE_INT_SIZE> tmp1 = v;
    tmp1 *= mul.n();
    ttmath::Int<RESCALE_INT_SIZE> tmp2 = mul.d();

    tmp1 = (tmp1 + tmp2 - 1) / tmp2;
    int64_t ret;
    tmp1.ToInt(ret);
    return ret;
}

#endif