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
#include "rgy_level_av1.h"

const int MAX_REF_FRAMES = 16;
const int LEVEL_COLUMNS  = 12;
const int COLUMN_MAX_BITRATE_MAIN = 6;
const int COLUMN_MAX_BITRATE_HIGH = 7;

const int LEVEL_INDEX_AV1[] = {
    -1,  //auto
     0,
     1,
     2,
     3,
     4,
     5,
     6,
     7,
     8,
     9,
     10,
     11,
     12,
     13,
     14,
     15,
     16,
     17,
     18,
     19,
     20,
     21,
     22,
     23
};

static const int64_t LEVEL_LIMITS_AV1[_countof(LEVEL_INDEX_AV1)+1][LEVEL_COLUMNS] =
{   //MaxPicSize, MaxHSize, MaxVSize, MaxDisplayRate, MaxDecodeRate, MaxHeaderRate, MainKbps, HighKbps, MainCR HighCR MaxTiles MaxTileCols
    {          0,        0,        0,            0,            0,    0,        0,        0,   0,   0,     0,    0,  }, // auto
    {     147456,     2048,     1152,      4423680,      5529600,  150,     1500,        0,   2,   0,     8,    4   }, // 2.0
    {     278784,     2816,     1584,      8363520,     10454400,  150,     3000,        0,   2,   0,     8,    4   }, // 2.1
    {     278784,     2816,     1584,      8363520,     10454400,  150,     3000,        0,   2,   0,     8,    4   }, // 2.2 dummy
    {     278784,     2816,     1584,      8363520,     10454400,  150,     3000,        0,   2,   0,     8,    4   }, // 2.3 dummy
    {     665856,     4352,     2448,     19975680,     24969600,  150,     6000,        0,   2,   0,    16,    6   }, // 3.0
    {    1065024,     5504,     3096,     31950720,     39938400,  150,    10000,        0,   2,   0,    16,    6   }, // 3.1
    {    1065024,     5504,     3096,     31950720,     39938400,  150,    10000,        0,   2,   0,    16,    6   }, // 3.2 dummy
    {    1065024,     5504,     3096,     31950720,     39938400,  150,    10000,        0,   2,   0,    16,    6   }, // 3.3 dummy
    {    2359296,     6144,     3456,     70778880,     77856768,  300,    12000,    30000,   4,   4,    32,    8   }, // 4.0
    {    2359296,     6144,     3456,    141557760,    155713536,  300,    20000,    50000,   4,   4,    32,    8   }, // 4.1
    {    2359296,     6144,     3456,    141557760,    155713536,  300,    20000,    50000,   4,   4,    32,    8   }, // 4.2 dummy
    {    2359296,     6144,     3456,    141557760,    155713536,  300,    20000,    50000,   4,   4,    32,    8   }, // 4.3 dummy
    {    8912896,     8192,     4352,    267386880,    273715200,  300,    30000,   100000,   6,   4,    64,    8   }, // 5.0
    {    8912896,     8192,     4352,    534773760,    547430400,  300,    40000,   160000,   8,   4,    64,    8   }, // 5.1
    {    8912896,     8192,     4352,   1069547520,   1094860800,  300,    60000,   240000,   8,   4,    64,    8   }, // 5.2
    {    8912896,     8192,     4352,   1069547520,   1176502272,  300,    60000,   240000,   8,   4,    64,    8   }, // 5.3
    {   35651584,    16384,     8704,   1069547520,   1176502272,  300,    60000,   240000,   8,   4,   128,   16   }, // 6.0
    {   35651584,    16384,     8704,   2139095040,   2189721600,  300,   100000,   480000,   8,   4,   128,   16   }, // 6.1
    {   35651584,    16384,     8704,   4278190080,   4379443200,  300,   160000,   800000,   8,   4,   128,   16   }, // 6.2
    {   35651584,    16384,     8704,   4278190080,   4706009088,  300,   160000,   800000,   8,   4,   128,   16   }, // 6.3
    {          0,        0,        0,            0,            0,    0,        0,        0,   0,   0,     0,    0,  }, // end
};

//必要なLevelを計算する, 適合するLevelがなければ 0 を返す
int calc_auto_level_av1(int width, int height, int ref, int fps_num, int fps_den, int profile, int max_bitrate, int tile_col, int tile_row) {
    int ref_mul_x3 = 3; //refのためにsample数にかけたい数を3倍したもの(あとで3で割る)
    if (ref > 12) {
        ref_mul_x3 = 4*3;
    } else if (ref > 8) {
        ref_mul_x3 = 2*3;
    } else if (ref > 6) {
        ref_mul_x3 = 4;
    } else {
        ref_mul_x3 = 3;
    }
    const int64_t data[] = {
        (int64_t)width * height, //MaxPicSize
        (int64_t)width,          //MaxHSize
        (int64_t)height,         //MaxVSize
        (int64_t)width * height * fps_num / fps_den, // MaxDisplayRate
        (int64_t)width * height * fps_num / fps_den, // MaxDecodeRate
        0, //MaxHeaderRate
        (profile == 0) ? 0 : max_bitrate, //MainKbps
        (profile  > 0) ? max_bitrate : 0, //HighKbps
        0, //MainCR
        0, //HighCR
        tile_col * tile_row, //MaxTiles
        tile_col          //MaxTileCols
    };
    static_assert(_countof(data) == LEVEL_COLUMNS);

    //あとはひたすら比較
    int i = 1, j = 0; // i -> 行(Level), j -> 列(項目)
    while (LEVEL_LIMITS_AV1[i][j])
        (data[j] > LEVEL_LIMITS_AV1[i][j]) ? i++ : j++;
    //一番右の列まで行き着いてればそれが求めるレベル
    int level_idx = (j == (LEVEL_COLUMNS-1)) ? i : _countof(LEVEL_INDEX_AV1)-1;
    return LEVEL_INDEX_AV1[level_idx];
}

int get_max_bitrate_av1(int level, int profile) {
    int level_idx = (int)(std::find(LEVEL_INDEX_AV1, LEVEL_INDEX_AV1 + _countof(LEVEL_INDEX_AV1), level) - LEVEL_INDEX_AV1);
    if (level_idx == _countof(LEVEL_INDEX_AV1)) {
        level_idx = 0;
    }
    auto bitrate = (level_idx > 0 && LEVEL_LIMITS_AV1[level_idx][0] > 0) ? LEVEL_LIMITS_AV1[level_idx][(profile > 0 && is_avail_high_profile_av1(level)) ? COLUMN_MAX_BITRATE_HIGH : COLUMN_MAX_BITRATE_MAIN] : 0;
    return (int)bitrate;
}

bool is_avail_high_profile_av1(int level) {
    int level_idx = (int)(std::find(LEVEL_INDEX_AV1, LEVEL_INDEX_AV1 + _countof(LEVEL_INDEX_AV1), level) - LEVEL_INDEX_AV1);
    if (level_idx == _countof(LEVEL_INDEX_AV1)) {
        level_idx = 0;
    }
    return (level_idx > 0 && LEVEL_LIMITS_AV1[level_idx][0] > 0) ? (LEVEL_LIMITS_AV1[level_idx][COLUMN_MAX_BITRATE_HIGH] > 1) : false;
}