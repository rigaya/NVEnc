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

#include <map>
#include <cstdint>
#include <algorithm>
#include "rgy_osdep.h"
#include "h264_level.h"

const int MAX_REF_FRAMES = 16;
const int PROGRESSIVE    = 1;
const int INTERLACED     = 2;
const int LEVEL_COLUMNS  = 7;
const int COLUMN_VBVMAX  = 4;
const int COLUMN_VBVBUF  = 5;

static const int H264_LEVEL_LIMITS[][LEVEL_COLUMNS] =
{   //interlaced, MaxMBpsec, MaxMBpframe, MaxDpbMbs, MaxVBVMaxrate, MaxVBVBuf,    end,  level
    { PROGRESSIVE,        -1,          -1,        -1,             0,         0,  0}, // auto
    { PROGRESSIVE,      1485,          99,       396,            64,       175,  0}, // 1
    { PROGRESSIVE,      1485,          99,       396,           128,       350,  0}, // 1b
    { PROGRESSIVE,      3000,         396,       900,           192,       500,  0}, // 1.1
    { PROGRESSIVE,      6000,         396,      2376,           384,      1000,  0}, // 1.2
    { PROGRESSIVE,     11880,         396,      2376,           768,      2000,  0}, // 1.3
    { PROGRESSIVE,     11880,         396,      2376,          2000,      2000,  0}, // 2
    {  INTERLACED,     19800,         792,      4752,          4000,      4000,  0}, // 2.1
    {  INTERLACED,     20250,        1620,      8100,          4000,      4000,  0}, // 2.2
    {  INTERLACED,     40500,        1620,      8100,         10000,     10000,  0}, // 3
    {  INTERLACED,    108000,        3600,     18000,         14000,     14000,  0}, // 3.1
    {  INTERLACED,    216000,        5120,     20480,         20000,     20000,  0}, // 3.2
    {  INTERLACED,    245760,        8192,     32768,         20000,     25000,  0}, // 4
    {  INTERLACED,    245760,        8192,     32768,         50000,     62500,  0}, // 4.1
    {  INTERLACED,    522240,        8704,     34816,         50000,     62500,  0}, // 4.2
    { PROGRESSIVE,    589824,       22080,    110400,        135000,    135000,  0}, // 5
    { PROGRESSIVE,    983040,       36864,    184320,        240000,    240000,  0}, // 5.1
    { PROGRESSIVE,   2073600,       36864,    184320,        240000,    240000,  0}, // 5.2
    {           0,         0,           0,         0,             0,         0,  0}, // end
};

const int H264_LEVEL_INDEX[] = {
    0,
    10,
    9,
    11,
    12,
    13,
    20,
    21,
    22,
    30,
    31,
    32,
    40,
    41,
    42,
    50,
    51,
    52
};

static const std::map<int, double> H264_PROFILE_VBV_MULTI = {
    {   0, 1.25 }, //auto = high
    {  66, 1.00 }, //baseline
    {  77, 1.00 }, //main
    { 100, 1.25 }, //high
    { 244, 3.00 }, //high444
};

static int ceil_div_int(int value, int div) {
    return (value + div - 1) / div;
}
static int64_t ceil_div_int64(int64_t value, int64_t div) {
    return (value + div - 1) / div;
}

//必要なLevelを計算する, 適合するLevelがなければ 0 を返す
int calc_h264_auto_level(int width, int height, int ref, bool interlaced, int fps_num, int fps_den, int profile, int vbv_max, int vbv_buf) {
    double profile_vbv_multi = H264_PROFILE_VBV_MULTI.at((H264_PROFILE_VBV_MULTI.count(profile) == 0) ? 0 : profile);
    int i, j = (interlaced) ? INTERLACED : PROGRESSIVE;
    int MB_frame = ceil_div_int(width, 16) * (j * ceil_div_int(height, 16*j));
    int data[LEVEL_COLUMNS] = {
        j,
        (int)ceil_div_int64((int64_t)MB_frame * fps_num, fps_den),
        MB_frame,
        MB_frame * ref,
        int(vbv_max * profile_vbv_multi + 0.5),
        int(vbv_buf * profile_vbv_multi + 0.5),
        0
    };

    //あとはひたすら比較
    i = 1, j = 0; // i -> 行(Level), j -> 列(項目)
    while (H264_LEVEL_LIMITS[i][j])
        (data[j] > H264_LEVEL_LIMITS[i][j]) ? i++ : j++;
    //一番右の列まで行き着いてればそれが求めるレベル 一応インターレースについても再チェック
    int level_idx = (j == (LEVEL_COLUMNS-1) && data[0] <= H264_LEVEL_LIMITS[i][0]) ? i : 0;
    return H264_LEVEL_INDEX[level_idx];
}

//vbv値を求める *vbv_max と *vbv_buf はNULLでもOK
void get_h264_vbv_value(int *vbv_max, int *vbv_buf, int level, int profile) {
    int level_idx = (int)(std::find(H264_LEVEL_INDEX, H264_LEVEL_INDEX + _countof(H264_LEVEL_INDEX), level) - H264_LEVEL_INDEX);
    if (level_idx == _countof(H264_LEVEL_INDEX)) {
        level_idx = 0;
    }
    double profile_vbv_multi = H264_PROFILE_VBV_MULTI.at((H264_PROFILE_VBV_MULTI.count(profile) == 0) ? 0 : profile);
    if (level_idx > 0 && H264_LEVEL_LIMITS[level_idx][1] > 0) {
        if (vbv_max)
            *vbv_max = (int)(H264_LEVEL_LIMITS[level_idx][COLUMN_VBVMAX] * profile_vbv_multi);
        if (vbv_buf)
            *vbv_buf = (int)(H264_LEVEL_LIMITS[level_idx][COLUMN_VBVBUF] * profile_vbv_multi);
    } else {
        if (vbv_max)
            *vbv_max = 0;
        if (vbv_buf)
            *vbv_buf = 0;
    }
    return;
}
