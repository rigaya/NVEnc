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
// --------------------------------------------------------------------------------------------

#include "rgy_level_h264.h"
#include "rgy_level_hevc.h"
#include "rgy_level_av1.h"
#if ENCODER_NVENC
#include "NVEncParam.h"
#endif

std::unique_ptr<RGYCodecLevel> createCodecLevel(RGY_CODEC codec) {
    switch (codec) {
    case RGY_CODEC_H264:
        return std::unique_ptr<RGYCodecLevel>(new RGYCodecLevelH264());
    case RGY_CODEC_HEVC:
        return std::unique_ptr<RGYCodecLevel>(new RGYCodecLevelHEVC());
    case RGY_CODEC_AV1:
        return std::unique_ptr<RGYCodecLevel>(new RGYCodecLevelAV1());
    default:
        return nullptr;
    }
}

int RGYCodecLevel::level_auto() {
    return get_cx_value(get_level_list(m_codec), _T("auto"));
}
