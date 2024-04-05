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
// --------------------------------------------------------------------------------------------

#pragma once
#ifndef __RGY_ASPECT_RATIO_H__
#define __RGY_ASPECT_RATIO_H__

#include <utility>
#include "rgy_def.h"

void get_dar_pixels(unsigned int* width, unsigned int* height, int sar_w, int sar_h);
std::pair<int, int> get_sar(unsigned int width, unsigned int height, unsigned int darWidth, unsigned int darHeight);
void adjust_sar(int *sar_w, int *sar_h, int width, int height);
void set_auto_resolution(int& dst_w, int& dst_h, int dst_sar_w, int dst_sar_h, int src_w, int src_h, int src_sar_w, int src_sar_h, const int mod_w, const int mod_h, const RGYResizeResMode mode, const bool ignoreSAR, const sInputCrop& crop);
int get_h264_sar_idx(std::pair<int, int>sar);
std::pair<int, int> get_h264_sar(int idx);

#endif //__RGY_ASPECT_RATIO_H__
