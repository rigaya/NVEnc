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

#include "rgy_aspect_ratio.h"
#include "rgy_util.h"

struct sar_option_t {
    int key;
    int sar[2];
};

static const sar_option_t SAR_LIST[] = {
    {  0, {  0,  0 } },
    {  1, {  1,  1 } },
    {  2, { 12, 11 } },
    {  3, { 10, 11 } },
    {  4, { 16, 11 } },
    {  5, { 40, 33 } },
    {  6, { 24, 11 } },
    {  7, { 20, 11 } },
    {  8, { 32, 11 } },
    {  9, { 80, 33 } },
    { 10, { 18, 11 } },
    { 11, { 15, 11 } },
    { 12, { 64, 33 } },
    { 13, {160, 99 } },
    { 14, {  4,  3 } },
    { 15, {  3,  2 } },
    { 16, {  2,  1 } }
};

std::pair<int, int> get_h264_sar(int idx) {
    for (int i = 0; i < _countof(SAR_LIST); i++) {
        if (SAR_LIST[i].key == idx)
            return std::make_pair(SAR_LIST[i].sar[0], SAR_LIST[i].sar[1]);
    }
    return std::make_pair(0, 0);
}

int get_h264_sar_idx(std::pair<int, int> sar) {

    if (0 != sar.first && 0 != sar.second) {
        rgy_reduce(sar);
    }

    for (int i = 0; i < _countof(SAR_LIST); i++) {
        if (SAR_LIST[i].sar[0] == sar.first && SAR_LIST[i].sar[1] == sar.second) {
            return SAR_LIST[i].key;
        }
    }
    return -1;
}

void adjust_sar(int *sar_w, int *sar_h, int width, int height) {
    int aspect_w = *sar_w;
    int aspect_h = *sar_h;
    //正負チェック
    if (aspect_w * aspect_h <= 0)
        aspect_w = aspect_h = 0;
    else if (aspect_w < 0) {
        //負で与えられている場合はDARでの指定
        //SAR比に変換する
        int dar_x = -1 * aspect_w;
        int dar_y = -1 * aspect_h;
        int x = dar_x * height;
        int y = dar_y * width;
        //多少のづれは容認する
        if (abs(y - x) > 16 * dar_y) {
            //gcd
            int a = x, b = y, c;
            while ((c = a % b) != 0)
                a = b, b = c;
            *sar_w = x / b;
            *sar_h = y / b;
        } else {
            *sar_w = *sar_h = 1;
        }
    } else {
        //sarも一応gcdをとっておく
        int a = aspect_w, b = aspect_h, c;
        while ((c = a % b) != 0)
            a = b, b = c;
        *sar_w = aspect_w / b;
        *sar_h = aspect_h / b;
    }
}

void get_dar_pixels(unsigned int* width, unsigned int* height, int sar_w, int sar_h) {
    int w = *width;
    int h = *height;
    if (0 != (w * h * sar_w * sar_h)) {
        int x = w * sar_w;
        int y = h * sar_h;
        int a = x, b = y, c;
        while ((c = a % b) != 0)
            a = b, b = c;
        x /= b;
        y /= b;
        const double ratio = (sar_w >= sar_h) ? h / (double)y : w / (double)x;
        *width  = (int)(x * ratio + 0.5);
        *height = (int)(y * ratio + 0.5);
    }
}

std::pair<int, int> get_sar(unsigned int width, unsigned int height, unsigned int darWidth, unsigned int darHeight) {
    int x = darWidth  * height;
    int y = darHeight *  width;
    int a = x, b = y, c;
    while ((c = a % b) != 0)
        a = b, b = c;
    return std::make_pair<int, int>(x / b, y / b);
}

void set_auto_resolution(int& dst_w, int& dst_h, int dst_sar_w, int dst_sar_h, int src_w, int src_h, int src_sar_w, int src_sar_h, const int mod_w, const int mod_h, const RGYResizeResMode mode, const bool ignoreSAR, const sInputCrop& crop) {
    if (ignoreSAR) {
        dst_sar_w = 1;
        dst_sar_h = 1;
        src_sar_w = 1;
        src_sar_h = 1;
    }
    if (dst_w * dst_h < 0
        || mode == RGYResizeResMode::PreserveOrgAspectDec
        || mode == RGYResizeResMode::PreserveOrgAspectInc) {
        // cropの反映
        src_w -= (crop.e.left + crop.e.right);
        src_h -= (crop.e.bottom + crop.e.up);
        if (dst_sar_w * dst_sar_h <= 0) {
            dst_sar_w = dst_sar_h = 1;
        }
        // 最終的なアスペクト比の計算
        double dar = src_w / (double)src_h;
        if (dst_sar_w < 0 && dst_sar_h < 0) {
            dar = dst_sar_w / (double)dst_sar_h;
        } else {
            if (src_sar_w * src_sar_h > 0) {
                if (src_sar_w < 0) {
                    dar = src_sar_w / (double)src_sar_h;
                } else {
                    dar = (src_w * (double)src_sar_w) / (src_h * (double)src_sar_h);
                }
            }
            dar /= (dst_sar_w / (double)dst_sar_h);
        }
        // PreserveOrgAspectInc, PreserveOrgAspectDec の場合、
        // 解像度の変更が必要となる側のdst_xに負の値を設定し、
        // 次のステップで適用する解像度の計算を行う
        // このとき、この負の値はYUV420の条件等を考慮した mod_w, mod_h の負の数とする
        if (dst_w * dst_h > 0
            && (mode == RGYResizeResMode::PreserveOrgAspectDec
             || mode == RGYResizeResMode::PreserveOrgAspectInc)) {
            int w_new = (int)(dst_h * dar + 0.5);
            if (mode == RGYResizeResMode::PreserveOrgAspectDec) {
                // 大きくなってしまうほうを変更する
                if (w_new > dst_w) {
                    dst_h = -mod_h;
                } else {
                    dst_w = -mod_w;
                }
            } else { //mode == RGYResizeResMode::PreserveOrgAspectInc
                // 小さくなってしまうほうを変更する
                if (w_new < dst_w) {
                    dst_h = -mod_h;
                } else {
                    dst_w = -mod_w;
                }
            }
        }
        // 適用する解像度の計算
        if (dst_w < 0) {
            const int div = std::abs(dst_w);
            dst_w = (((int)(dst_h * dar) + (div >> 1)) / div) * div;
        } else { //dst_h < 0
            const int div = std::abs(dst_h);
            dst_h = (((int)(dst_w / dar) + (div >> 1)) / div) * div;
        }
    }
}
