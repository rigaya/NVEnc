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

#ifndef __RGY_LEVEL_H__
#define __RGY_LEVEL_H__

#include <memory>
#include "rgy_def.h"


class RGYCodecLevel {
public:
    RGYCodecLevel() : m_codec(RGY_CODEC_UNKNOWN) {};
    virtual ~RGYCodecLevel() {};
    virtual int calc_auto_level(int width, int height, int ref, bool interlaced, int fps_num, int fps_den, int profile, bool high_tier, int max_bitrate, int vbv_buf, int tile_col, int tile_row) = 0;
    virtual int get_max_bitrate(int level, int profile, bool high_tier = false) = 0;
    virtual int get_max_vbv_buf(int level, int profile) = 0;
    virtual int get_max_ref(int width, int height, int level, bool interlaced) = 0;
    virtual int level_auto();
protected:
    RGY_CODEC m_codec;
};

std::unique_ptr<RGYCodecLevel> createCodecLevel(RGY_CODEC codec);

#endif //__RGY_LEVEL_AV1_H__
