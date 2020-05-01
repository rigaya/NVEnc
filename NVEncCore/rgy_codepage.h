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
// -------------------------------------------------------------------------------------------

#pragma once
#ifndef __RGY_CODEPAGE_H__
#define __RGY_CODEPAGE_H__

//日本語環境の一般的なコードページ一覧
enum : uint32_t {
    CODE_PAGE_SJIS        = 932, //Shift-JIS
    CODE_PAGE_JIS         = 50220,
    CODE_PAGE_EUC_JP      = 51932,
    CODE_PAGE_UTF8        = 65001,
    CODE_PAGE_UTF16_LE    = 1200, //WindowsのUnicode WCHAR のコードページ
    CODE_PAGE_UTF16_BE    = 1201,
    CODE_PAGE_US_ASCII    = 20127,
    CODE_PAGE_WEST_EUROPE = 1252,  //厄介な西ヨーロッパ言語
    CODE_PAGE_UNSET       = 0xffffffff,
};

uint32_t get_code_page(const void *str, uint32_t size_in_byte);
const char *codepage_str(uint32_t codepage);

#endif //__RGY_CODEPAGE_H__
