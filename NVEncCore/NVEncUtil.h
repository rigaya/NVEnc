// -----------------------------------------------------------------------------------------
// NVEnc by rigaya
// -----------------------------------------------------------------------------------------
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
#ifndef __NVENC_UTIL_H__
#define __NVENC_UTIL_H__

#include <utility>
#include <array>
#include "nvEncodeAPI.h"
#include "cuviddec.h"
#include "ConvertCsp.h"
#include "rgy_util.h"

#ifndef cudaVideoCodec_VP8
#define cudaVideoCodec_VP8 (cudaVideoCodec)(cudaVideoCodec_HEVC+1)
#endif
#ifndef cudaVideoCodec_VP9
#define cudaVideoCodec_VP9 (cudaVideoCodec)(cudaVideoCodec_HEVC+2)
#endif

#define MAP_PAIR_0_1(prefix, name0, type0, name1, type1, map_pair, default0, default1) \
    __declspec(noinline) \
    type1 prefix ## _ ## name0 ## _to_ ## name1(type0 var0) {\
        auto ret = std::find_if(map_pair.begin(), map_pair.end(), [var0](std::pair<type0, type1> a) { \
            return a.first == var0; \
        }); \
        return (ret == map_pair.end()) ? default1 : ret->second; \
    } \
    __declspec(noinline)  \
    type0 prefix ## _ ## name1 ## _to_ ## name0(type1 var1) {\
        auto ret = std::find_if(map_pair.begin(), map_pair.end(), [var1](std::pair<type0, type1> a) { \
            return a.second == var1; \
        }); \
        return (ret == map_pair.end()) ? default0 : ret->first; \
    }

#define MAP_PAIR_0_1_PROTO(prefix, name0, type0, name1, type1) \
    type1 prefix ## _ ## name0 ## _to_ ## name1(type0 var0); \
    type0 prefix ## _ ## name1 ## _to_ ## name0(type1 var1);

MAP_PAIR_0_1_PROTO(codec, rgy, RGY_CODEC, enc, cudaVideoCodec);
MAP_PAIR_0_1_PROTO(chromafmt, rgy, RGY_CHROMAFMT, enc, cudaVideoChromaFormat);
MAP_PAIR_0_1_PROTO(csp, rgy, RGY_CSP, enc, NV_ENC_BUFFER_FORMAT);
MAP_PAIR_0_1_PROTO(codec_guid, rgy, RGY_CODEC, enc, GUID);
MAP_PAIR_0_1_PROTO(codec_guid_profile, rgy, RGY_CODEC_DATA, enc, GUID);

NV_ENC_PIC_STRUCT picstruct_rgy_to_enc(RGY_PICSTRUCT picstruct);
RGY_PICSTRUCT picstruct_enc_to_rgy(NV_ENC_PIC_STRUCT picstruct);

RGY_CSP getEncCsp(NV_ENC_BUFFER_FORMAT enc_buffer_format);

VideoInfo videooutputinfo(
    const GUID& encCodecGUID,
    NV_ENC_BUFFER_FORMAT buffer_fmt,
    int nEncWidth,
    int nEncHeight,
    const NV_ENC_CONFIG *pEncConfig,
    NV_ENC_PIC_STRUCT nPicStruct,
    std::pair<int, int> sar,
    std::pair<int, int> outFps);

#endif //__NVENC_UTIL_H__
