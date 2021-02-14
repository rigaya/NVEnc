// -----------------------------------------------------------------------------------------
// QSVEnc/NVEnc by rigaya
// -----------------------------------------------------------------------------------------
// The MIT License
//
// Copyright (c) 2019 rigaya
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

#include <cstdint>
#include <cstring>
#include "rgy_codepage.h"

//BOM文字リスト
static const int MAX_UTF8_CHAR_LENGTH = 6;
static const uint8_t UTF8_BOM[]     = { 0xEF, 0xBB, 0xBF };
static const uint8_t UTF16_LE_BOM[] = { 0xFF, 0xFE };
static const uint8_t UTF16_BE_BOM[] = { 0xFE, 0xFF };

//ボム文字かどうか、コードページの判定
static uint32_t check_bom(const void* chr) {
    if (chr == nullptr) return CODE_PAGE_UNSET;
    if (memcmp(chr, UTF16_LE_BOM, sizeof(UTF16_LE_BOM)) == 0) return CODE_PAGE_UTF16_LE;
    if (memcmp(chr, UTF16_BE_BOM, sizeof(UTF16_BE_BOM)) == 0) return CODE_PAGE_UTF16_BE;
    if (memcmp(chr, UTF8_BOM,     sizeof(UTF8_BOM))     == 0) return CODE_PAGE_UTF8;
    return CODE_PAGE_UNSET;
}

static bool isJis(const void *str, uint32_t size_in_byte) {
    static const uint8_t ESCAPE[][7] = {
        //先頭に比較すべきバイト数
        { 3, 0x1B, 0x28, 0x42, 0x00, 0x00, 0x00 },
        { 3, 0x1B, 0x28, 0x4A, 0x00, 0x00, 0x00 },
        { 3, 0x1B, 0x28, 0x49, 0x00, 0x00, 0x00 },
        { 3, 0x1B, 0x24, 0x40, 0x00, 0x00, 0x00 },
        { 3, 0x1B, 0x24, 0x42, 0x00, 0x00, 0x00 },
        { 6, 0x1B, 0x26, 0x40, 0x1B, 0x24, 0x42 },
        { 4, 0x1B, 0x24, 0x28, 0x44, 0x00, 0x00 },
        { 0, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00 } //終了
    };
    const uint8_t * const str_fin = (const uint8_t *)str + size_in_byte;
    for (const uint8_t *chr = (const uint8_t *)str; chr < str_fin; chr++) {
        if (*chr > 0x7F)
            return false;
        for (int i = 0; ESCAPE[i][0]; i++) {
            if (str_fin - chr > ESCAPE[i][0] &&
                memcmp(chr, &ESCAPE[i][1], ESCAPE[i][0]) == 0)
                return true;
        }
    }
    return false;
}

static uint32_t isUTF16(const void *str, uint32_t size_in_byte) {
    const uint8_t * const str_fin = (const uint8_t *)str + size_in_byte;
    for (const uint8_t *chr = (const uint8_t *)str; chr < str_fin; chr++) {
        if (chr[0] == 0x00 && str_fin - chr > 1 && chr[1] <= 0x7F)
            return ((chr - (const uint8_t *)str) % 2 == 1) ? CODE_PAGE_UTF16_LE : CODE_PAGE_UTF16_BE;
    }
    return CODE_PAGE_UNSET;
}

static bool isASCII(const void *str, uint32_t size_in_byte) {
    const uint8_t * const str_fin = (const uint8_t *)str + size_in_byte;
    for (const uint8_t *chr = (const uint8_t *)str; chr < str_fin; chr++) {
        if (*chr == 0x1B || *chr >= 0x80)
            return false;
    }
    return true;
}

static uint32_t jpn_check(const void *str, uint32_t size_in_byte) {
    int score_sjis = 0;
    int score_euc = 0;
    int score_utf8 = 0;
    const uint8_t * const str_fin = (const uint8_t *)str + size_in_byte;
    for (const uint8_t *chr = (const uint8_t *)str; chr < str_fin - 1; chr++) {
        if ((0x81 <= chr[0] && chr[0] <= 0x9F) ||
            (0xE0 <= chr[0] && chr[0] <= 0xFC) ||
            (0x40 <= chr[1] && chr[1] <= 0x7E) ||
            (0x80 <= chr[1] && chr[1] <= 0xFC)) {
            score_sjis += 2; chr++;
        }
    }
    for (const uint8_t *chr = (const uint8_t *)str; chr < str_fin - 1; chr++) {
        if ((0xC0 <= chr[0] && chr[0] <= 0xDF) &&
            (0x80 <= chr[1] && chr[1] <= 0xBF)) {
            score_utf8 += 2; chr++;
        } else if (
            str_fin - chr > 2 &&
            (0xE0 <= chr[0] && chr[0] <= 0xEF) &&
            (0x80 <= chr[1] && chr[1] <= 0xBF) &&
            (0x80 <= chr[2] && chr[2] <= 0xBF)) {
            score_utf8 += 3; chr++;
        }
    }
    for (const uint8_t *chr = (const uint8_t *)str; chr < str_fin - 1; chr++) {
        if (((0xA1 <= chr[0] && chr[0] <= 0xFE) && (0xA1 <= chr[1] && chr[1] <= 0xFE)) ||
            (chr[0] == 0x8E                     && (0xA1 <= chr[1] && chr[1] <= 0xDF))) {
            score_euc += 2; chr++;
        } else if (
            str_fin - chr > 2 &&
            chr[0] == 0x8F &&
            (0xA1 <= chr[1] && chr[1] <= 0xFE) &&
            (0xA1 <= chr[2] && chr[2] <= 0xFE)) {
            score_euc += 3; chr += 2;
        }
    }
    if (score_sjis > score_euc && score_sjis > score_utf8)
        return CODE_PAGE_SJIS;
    if (score_utf8 > score_euc && score_utf8 > score_sjis)
        return CODE_PAGE_UTF8;
    if (score_euc > score_sjis && score_euc > score_utf8)
        return CODE_PAGE_EUC_JP;
    return CODE_PAGE_UNSET;
}

uint32_t get_code_page(const void *str, uint32_t size_in_byte) {
    uint32_t ret = CODE_PAGE_UNSET;
    if ((ret = check_bom(str)) != CODE_PAGE_UNSET)
        return ret;

    if (isJis(str, size_in_byte))
        return CODE_PAGE_JIS;

    if ((ret = isUTF16(str, size_in_byte)) != CODE_PAGE_UNSET)
        return ret;

    if (isASCII(str, size_in_byte))
        return CODE_PAGE_US_ASCII;

    return jpn_check(str, size_in_byte);
}

const char *codepage_str(uint32_t codepage) {
    switch (codepage) {
    case CODE_PAGE_SJIS:
        return "CP932";
    case CODE_PAGE_EUC_JP:
        return "EUC-JP";
    case CODE_PAGE_UTF16_LE:
        return "UTF16LE";
    case CODE_PAGE_UTF16_BE:
        return "UTF16BE";
    case CODE_PAGE_JIS:
        return "ISO2022JP";
    case CODE_PAGE_UTF8:
        return "UTF-8";
    default:
        return nullptr;
    }
}
