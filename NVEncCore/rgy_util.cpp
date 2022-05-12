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

#include "rgy_util.h"
#include "rgy_codepage.h"
#if !(defined(_WIN32) || defined(_WIN64))
#include <iconv.h>
#endif

#pragma warning (push)
#pragma warning (disable: 4100)
#if defined(_WIN32) || defined(_WIN64)
unsigned int wstring_to_string(const wchar_t *wstr, std::string& str, uint32_t codepage) {
    if (wstr == nullptr) {
        str = "";
        return 0;
    }
    uint32_t flags = (codepage == CP_UTF8) ? 0 : WC_NO_BEST_FIT_CHARS;
    int multibyte_length = WideCharToMultiByte(codepage, flags, wstr, -1, nullptr, 0, nullptr, nullptr);
    std::vector<char> tmp(multibyte_length, 0);
    if (0 == WideCharToMultiByte(codepage, flags, wstr, -1, tmp.data(), (int)tmp.size(), nullptr, nullptr)) {
        str.clear();
        return 0;
    }
    str = tmp.data();
    return multibyte_length;
}
#else
unsigned int wstring_to_string(const wchar_t *wstr, std::string& str, uint32_t codepage) {
    if (wstr == nullptr) {
        str = "";
        return 0;
    }
    auto codepage_str_to = codepage_str(codepage);
    if (codepage_str_to == nullptr) codepage_str_to = "UTF-8";
    auto ic = iconv_open(codepage_str_to, "wchar_t"); //to, from
    auto input_len = (wcslen(wstr)+1) * 4;
    std::vector<char> buf(input_len, 0);
    memcpy(buf.data(), wstr, input_len);
    auto output_len = input_len * 8;
    std::vector<char> bufout(output_len, 0);
    char *outbuf = bufout.data();
    char *input = buf.data();
    iconv(ic, &input, &input_len, &outbuf, &output_len);
    iconv_close(ic);
    str = bufout.data();
    return output_len;
}
#endif //#if defined(_WIN32) || defined(_WIN64)

std::string wstring_to_string(const wchar_t *wstr, uint32_t codepage) {
    if (wstr == nullptr) {
        return "";
    }
    std::string str;
    wstring_to_string(wstr, str, codepage);
    return str;
}

std::string wstring_to_string(const std::wstring& wstr, uint32_t codepage) {
    std::string str;
    wstring_to_string(wstr.c_str(), str, codepage);
    return str;
}

unsigned int tchar_to_string(const TCHAR *tstr, std::string& str, uint32_t codepage) {
#if UNICODE
    return wstring_to_string(tstr, str, codepage);
#else
    str = (tstr) ? std::string(tstr) : "";
    return (unsigned int)str.length();
#endif
}

std::string tchar_to_string(const TCHAR *tstr, uint32_t codepage) {
    if (tstr == nullptr) {
        return "";
    }
    std::string str;
    tchar_to_string(tstr, str, codepage);
    return str;
}

std::wstring tchar_to_wstring(const tstring& tstr, uint32_t codepage) {
#if UNICODE
    return std::wstring(tstr);
#else
    return char_to_wstring(tstr, codepage);
#endif
}

std::wstring tchar_to_wstring(const TCHAR *tstr, uint32_t codepage) {
    if (tstr == nullptr) {
        return L"";
    }
    return tchar_to_wstring(tstring(tstr), codepage);
}

std::string tchar_to_string(const tstring& tstr, uint32_t codepage) {
    std::string str;
    tchar_to_string(tstr.c_str(), str, codepage);
    return str;
}

unsigned int wstring_to_tstring(const WCHAR *wstr, tstring& tstr, uint32_t codepage) {
    if (wstr == nullptr) {
        tstr = _T("");
        return 0;
    }
#if UNICODE
    tstr = std::wstring(wstr);
#else
    return wstring_to_string(wstr, tstr, codepage);
#endif
    return (unsigned int)tstr.length();
}

tstring wstring_to_tstring(const WCHAR *wstr, uint32_t codepage) {
    if (wstr == nullptr) {
        return _T("");
    }
    tstring tstr;
    wstring_to_tstring(wstr, tstr, codepage);
    return tstr;
}

tstring wstring_to_tstring(const std::wstring& wstr, uint32_t codepage) {
    tstring tstr;
    wstring_to_tstring(wstr.c_str(), tstr, codepage);
    return tstr;
}

#if defined(_WIN32) || defined(_WIN64)
unsigned int char_to_wstring(std::wstring& wstr, const char *str, uint32_t codepage) {
    if (str == nullptr) {
        wstr = L"";
        return 0;
    }
    int widechar_length = MultiByteToWideChar(codepage, 0, str, -1, nullptr, 0);
    std::vector<wchar_t> tmp(widechar_length, 0);
    if (0 == MultiByteToWideChar(codepage, 0, str, -1, tmp.data(), (int)tmp.size())) {
        wstr.clear();
        return 0;
    }
    wstr = tmp.data();
    return widechar_length;
}
unsigned int char_to_string(std::string& dst, uint32_t codepage_to, const char *src, uint32_t codepage_from) {
    if (src == nullptr) {
        dst = "";
        return 0;
    }
    if (codepage_to == codepage_from) {
        dst = src;
        return (unsigned int)dst.length();
    }
    std::wstring wstrtemp;
    char_to_wstring(wstrtemp, src, codepage_from);
    wstring_to_string(wstrtemp.c_str(), dst, codepage_to);
    return (unsigned int)dst.length();
}
#else
unsigned int char_to_wstring(std::wstring& wstr, const char *str, uint32_t codepage) {
    if (str == nullptr) {
        wstr = L"";
        return 0;
    }
    auto codepage_str_from = codepage_str(codepage);
    if (codepage_str_from == nullptr) codepage_str_from = "UTF-8";
    auto ic = iconv_open("wchar_t", codepage_str_from); //to, from
    if ((int64_t)ic == -1) {
        fprintf(stderr, "iconv_error\n");
    }
    auto input_len = strlen(str)+1;
    std::vector<char> buf(input_len);
    strcpy(buf.data(), str);
    auto output_len = (input_len + 1) * 8;
    std::vector<char> bufout(output_len, 0);
    char *inbuf = buf.data();
    char *outbuf = bufout.data();
    iconv(ic, &inbuf, &input_len, &outbuf, &output_len);
    iconv_close(ic);
    wstr = std::wstring((WCHAR *)bufout.data());
    return wstr.length();
}

unsigned int char_to_string(std::string& dst, uint32_t codepage_to, const char *src, uint32_t codepage_from) {
    if (src == nullptr) {
        dst = "";
        return 0;
    }
    auto codepage_str_from = codepage_str(codepage_from);
    if (codepage_str_from == nullptr) codepage_str_from = "UTF-8";
    auto codepage_str_to = codepage_str(codepage_to);
    if (codepage_str_to == nullptr) codepage_str_to = "UTF-8";
    if (codepage_to == codepage_from
        || strcmp(codepage_str_to, codepage_str_from) == 0) {
        dst = src;
        return dst.length();
    }
    auto ic = iconv_open(codepage_str_to, codepage_str_from); //to, from
    if ((int64_t)ic == -1) {
        fprintf(stderr, "iconv_error\n");
    }
    auto input_len = strlen(src)+1;
    std::vector<char> buf(input_len);
    strcpy(buf.data(), src);
    auto output_len = (input_len + 1) * 12;
    std::vector<char> bufout(output_len, 0);
    char *inbuf = buf.data();
    char *outbuf = bufout.data();
    iconv(ic, &inbuf, &input_len, &outbuf, &output_len);
    iconv_close(ic);
    dst = std::string(bufout.data());
    return dst.length();
}
#endif //#if defined(_WIN32) || defined(_WIN64)
std::wstring char_to_wstring(const char *str, uint32_t codepage) {
    if (str == nullptr) {
        return L"";
    }
    std::wstring wstr;
    char_to_wstring(wstr, str, codepage);
    return wstr;
}
std::wstring char_to_wstring(const std::string& str, uint32_t codepage) {
    std::wstring wstr;
    char_to_wstring(wstr, str.c_str(), codepage);
    return wstr;
}
std::string char_to_string(uint32_t codepage_to, const char *src, uint32_t codepage_from) {
    std::string dst;
    char_to_string(dst, codepage_to, src, codepage_from);
    return dst;
}

unsigned int char_to_tstring(tstring& tstr, const char *str, uint32_t codepage) {
#if UNICODE
    return char_to_wstring(tstr, str, codepage);
#else
    tstr = (str) ? std::string(str) : _T("");
    return (unsigned int)tstr.length();
#endif
}

tstring char_to_tstring(const char *str, uint32_t codepage) {
    if (str == nullptr) {
        return _T("");
    }
    tstring tstr;
    char_to_tstring(tstr, str, codepage);
    return tstr;
}
tstring char_to_tstring(const std::string& str, uint32_t codepage) {
    tstring tstr;
    char_to_tstring(tstr, str.c_str(), codepage);
    return tstr;
}
std::string strsprintf(const char* format, ...) {
    if (format == nullptr) {
        return "";
    }
    va_list args;
    va_start(args, format);
    const size_t len = _vscprintf(format, args) + 1;

    std::vector<char> buffer(len, 0);
    vsprintf(buffer.data(), format, args);
    va_end(args);
    std::string retStr = std::string(buffer.data());
    return retStr;
}
#if defined(_WIN32) || defined(_WIN64)
std::wstring strsprintf(const WCHAR* format, ...) {
    if (format == nullptr) {
        return L"";
    }
    va_list args;
    va_start(args, format);
    const size_t len = _vscwprintf(format, args) + 1;

    std::vector<WCHAR> buffer(len, 0);
    vswprintf(buffer.data(), buffer.size(), format, args);
    va_end(args);
    std::wstring retStr = std::wstring(buffer.data());
    return retStr;
}
#endif //#if defined(_WIN32) || defined(_WIN64)

std::string str_replace(std::string str, const std::string& from, const std::string& to) {
    std::string::size_type pos = 0;
    while(pos = str.find(from, pos), pos != std::string::npos) {
        str.replace(pos, from.length(), to);
        pos += to.length();
    }
    return str;
}

#if defined(_WIN32) || defined(_WIN64)
std::wstring str_replace(std::wstring str, const std::wstring& from, const std::wstring& to) {
    std::wstring::size_type pos = 0;
    while (pos = str.find(from, pos), pos != std::wstring::npos) {
        str.replace(pos, from.length(), to);
        pos += to.length();
    }
    return str;
}
#endif //#if defined(_WIN32) || defined(_WIN64)

#pragma warning (pop)
#if defined(_WIN32) || defined(_WIN64)
std::vector<std::wstring> split(const std::wstring &str, const std::wstring &delim, bool bTrim) {
    std::vector<std::wstring> res;
    size_t current = 0, found, delimlen = delim.size();
    while (std::wstring::npos != (found = str.find(delim, current))) {
        auto segment = std::wstring(str, current, found - current);
        if (bTrim) {
            segment = trim(segment);
        }
        if (!bTrim || segment.length()) {
            res.push_back(segment);
        }
        current = found + delimlen;
    }
    auto segment = std::wstring(str, current, str.size() - current);
    if (bTrim) {
        segment = trim(segment);
    }
    if (!bTrim || segment.length()) {
        res.push_back(std::wstring(segment.c_str()));
    }
    return res;
}
#endif //#if defined(_WIN32) || defined(_WIN64)

std::vector<std::string> split(const std::string &str, const std::string &delim, bool bTrim) {
    std::vector<std::string> res;
    size_t current = 0, found, delimlen = delim.size();
    while (std::string::npos != (found = str.find(delim, current))) {
        auto segment = std::string(str, current, found - current);
        if (bTrim) {
            segment = trim(segment);
        }
        if (!bTrim || segment.length()) {
            res.push_back(segment);
        }
        current = found + delimlen;
    }
    auto segment = std::string(str, current, str.size() - current);
    if (bTrim) {
        segment = trim(segment);
    }
    if (!bTrim || segment.length()) {
        res.push_back(std::string(segment.c_str()));
    }
    return res;
}

#if defined(_WIN32) || defined(_WIN64)
std::vector<std::wstring> sep_cmd(const std::wstring& cmd) {
    std::vector<std::wstring> args;
    int argc = 0;
    auto ptr = CommandLineToArgvW(cmd.c_str(), &argc);
    for (int i = 0; i < argc; i++) {
        args.push_back(ptr[i]);
    }
    args.push_back(L"");
    LocalFree(ptr);
    return std::move(args);
}

std::vector<std::string> sep_cmd(const std::string& cmd) {
    std::vector<std::string> args;
    std::wstring wcmd = char_to_wstring(cmd);
    for (const auto &warg : sep_cmd(wcmd)) {
        args.push_back(wstring_to_string(warg));
    }
    return std::move(args);
}
#endif //#if defined(_WIN32) || defined(_WIN64)

std::string lstrip(const std::string& string, const char* trim) {
    auto result = string;
    auto left = string.find_first_not_of(trim);
    if (left != std::string::npos) {
        result = string.substr(left, 0);
    }
    return result;
}

std::string rstrip(const std::string& string, const char* trim) {
    auto result = string;
    auto right = string.find_last_not_of(trim);
    if (right != std::string::npos) {
        result = string.substr(0, right);
    }
    return result;
}

std::string trim(const std::string& string, const char* trim) {
    auto result = string;
    auto left = string.find_first_not_of(trim);
    if (left != std::string::npos) {
        auto right = string.find_last_not_of(trim);
        result = string.substr(left, right - left + 1);
    }
    return result;
}

std::wstring lstrip(const std::wstring& string, const WCHAR* trim) {
    auto result = string;
    auto left = string.find_first_not_of(trim);
    if (left != std::string::npos) {
        result = string.substr(left, 0);
    }
    return result;
}

std::wstring rstrip(const std::wstring& string, const WCHAR* trim) {
    auto result = string;
    auto right = string.find_last_not_of(trim);
    if (right != std::string::npos) {
        result = string.substr(0, right);
    }
    return result;
}

std::wstring trim(const std::wstring& string, const WCHAR* trim) {
    auto result = string;
    auto left = string.find_first_not_of(trim);
    if (left != std::string::npos) {
        auto right = string.find_last_not_of(trim);
        result = string.substr(left, right - left + 1);
    }
    return result;
}

std::string add_indent(const std::string& str, const int indentLength) {
    const auto origLength = str.length();

    std::string indent(indentLength, ' ');

    std::string ret;
    ret.reserve(origLength + indentLength * origLength / 16);

    size_t current = 0, found;
    while (std::string::npos != (found = str.find("\n", current))) {
        auto segment = std::string(str, current, found - current);
        ret.append(indent);
        ret.append(segment);
        ret.append("\n");
        current = found + 1;
    }
    return ret;
}

std::wstring add_indent(const std::wstring& str, const int indentLength) {
    const auto origLength = str.length();

    std::wstring indent(indentLength, L' ');

    std::wstring ret;
    ret.reserve(origLength + indentLength * origLength / 16);

    size_t current = 0, found;
    while (std::wstring::npos != (found = str.find(L"\n", current))) {
        auto segment = std::wstring(str, current, found - current);
        ret.append(indent);
        ret.append(segment);
        ret.append(L"\n");
        current = found + 1;
    }
    return ret;
}

struct RGYSIPrefix {
    char prefix;
    bool inverse;
    int64_t pow2;
    int64_t pow10;

    RGYSIPrefix(char prefix_, bool inverse_, int64_t pow2_, int64_t pow10_) :
        prefix(prefix_), inverse(inverse_), pow2(pow2_), pow10(pow10_) {};
};

const auto RGY_SI_PREFIX_LIST = make_array<RGYSIPrefix>(
    RGYSIPrefix{ 'a', true,  rgy_pow_int<60,int64_t>(2), rgy_pow_int<18,int64_t>(10) },
    RGYSIPrefix{ 'f', true,  rgy_pow_int<50,int64_t>(2), rgy_pow_int<15,int64_t>(10) },
    RGYSIPrefix{ 'p', true,  rgy_pow_int<40,int64_t>(2), rgy_pow_int<12,int64_t>(10) },
    RGYSIPrefix{ 'n', true,  rgy_pow_int<30,int64_t>(2), rgy_pow_int< 9,int64_t>(10) },
    RGYSIPrefix{ 'u', true,  rgy_pow_int<20,int64_t>(2), rgy_pow_int< 6,int64_t>(10) },
    RGYSIPrefix{ 'm', true,  rgy_pow_int<10,int64_t>(2), rgy_pow_int< 3,int64_t>(10) },
    RGYSIPrefix{ 'k', false, rgy_pow_int<10,int64_t>(2), rgy_pow_int< 3,int64_t>(10) },
    RGYSIPrefix{ 'K', false, rgy_pow_int<10,int64_t>(2), rgy_pow_int< 3,int64_t>(10) },
    RGYSIPrefix{ 'M', false, rgy_pow_int<20,int64_t>(2), rgy_pow_int< 6,int64_t>(10) },
    RGYSIPrefix{ 'g', false, rgy_pow_int<30,int64_t>(2), rgy_pow_int< 9,int64_t>(10) },
    RGYSIPrefix{ 'G', false, rgy_pow_int<30,int64_t>(2), rgy_pow_int< 9,int64_t>(10) },
    RGYSIPrefix{ 't', false, rgy_pow_int<40,int64_t>(2), rgy_pow_int<12,int64_t>(10) },
    RGYSIPrefix{ 'T', false, rgy_pow_int<40,int64_t>(2), rgy_pow_int<12,int64_t>(10) },
    RGYSIPrefix{ 'P', false, rgy_pow_int<50,int64_t>(2), rgy_pow_int<15,int64_t>(10) },
    RGYSIPrefix{ 'E', false, rgy_pow_int<60,int64_t>(2), rgy_pow_int<18,int64_t>(10) }
    );

template<typename T>
static void rgy_apply_si_prefix(T& val, const TCHAR *endptr) {
    const auto prefix = tchar_to_string(endptr, CODE_PAGE_UTF8);
    if (prefix[0] != '\0') {
        auto siprefix = std::find_if(RGY_SI_PREFIX_LIST.begin(), RGY_SI_PREFIX_LIST.end(), [p = prefix[0]](const RGYSIPrefix& si) { return si.prefix == p; });
        if (siprefix != RGY_SI_PREFIX_LIST.end()) {
            const bool usepow2 = prefix[1] != 'i';
            if (siprefix->inverse) {
                val /= (usepow2) ? siprefix->pow2 : siprefix->pow10;
            } else {
                val *= (usepow2) ? siprefix->pow2 : siprefix->pow10;
            }
        }
    }
}

int rgy_parse_num(int& val, const tstring& str) {
    val = 0;
    try {
        size_t idx = 0;
        int64_t val64 = std::stoll(str, &idx, 0);
        auto endptr = str.c_str() + idx;
        rgy_apply_si_prefix(val64, endptr);
        if (val64 < std::numeric_limits<int>::min() || std::numeric_limits<int>::max() < val64) {
            val = 0;
            return 1;
        }
        val = (int)val64;
    } catch (...) {
        return 1;
    }
    return 0;
}

int rgy_parse_num(int64_t& val, const tstring& str) {
    val = 0;
    try {
        size_t idx = 0;
        val = std::stoll(str, &idx, 0);
        auto endptr = str.c_str() + idx;
        const auto prefix = tchar_to_string(endptr, CODE_PAGE_UTF8);
        rgy_apply_si_prefix(val, endptr);
    } catch (...) {
        return 1;
    }
    return 0;
}

int rgy_parse_num(float& val, const tstring& str) {
    val = 0;
    try {
        size_t idx = 0;
        double vald = std::stod(str, &idx);
        rgy_apply_si_prefix(vald, str.c_str() + idx);
        val = (float)vald;
    } catch (...) {
        return 1;
    }
    return 0;
}

int rgy_parse_num(double& val, const tstring& str) {
    val = 0;
    try {
        size_t idx = 0;
        val = std::stod(str, &idx);
        rgy_apply_si_prefix(val, str.c_str() + idx);
    } catch (...) {
        return 1;
    }
    return 0;
}

tstring rgy_print_num_with_siprefix(const int64_t value) {
    const RGYSIPrefix *usePrefix = nullptr;
    for (const auto& prefix : RGY_SI_PREFIX_LIST) {
        if (!prefix.inverse && value > prefix.pow10) {
            usePrefix = &prefix;
        }
    }
    if (usePrefix) {
        return strsprintf(_T("%.3f%c"), value / (double)usePrefix->pow10, usePrefix->prefix);
    } else {
        return strsprintf(_T("%lld"), value);
    }
}

tstring print_time(double time) {
    int sec = (int)time;
    time -= sec;
    int miniute = (int)(sec / 60);
    sec -= miniute * 60;
    int hour = miniute / 60;
    miniute -= hour * 60;
    tstring frac = strsprintf(_T("%.3f"), time);
    return strsprintf(_T("%d:%02d:%02d%s"), hour, miniute, sec, frac.substr(frac.find_first_of(_T("."))).c_str());
}

size_t malloc_degeneracy(void **ptr, size_t nSize, size_t nMinSize) {
    *ptr = nullptr;
    nMinSize = (std::max<size_t>)(nMinSize, 1);
    nSize = (std::max<size_t>)(nSize, nMinSize);
    //確保できなかったら、サイズを小さくして再度確保を試みる (最終的に1MBも確保できなかったら諦める)
    while (nSize >= nMinSize) {
        void *qtr = malloc(nSize);
        if (qtr != nullptr) {
            *ptr = qtr;
            return nSize;
        }
        size_t nNextSize = 0;
        for (size_t i = nMinSize; i < nSize; i<<=1) {
            nNextSize = i;
        }
        nSize = nNextSize;
    }
    return 0;
}

// convert float to half precision floating point
unsigned short float2half(float value) {
    // 1 : 8 : 23
    union {
        unsigned int u;
        float f;
    } tmp;

    tmp.f = value;

    // 1 : 8 : 23
    unsigned short sign = (tmp.u & 0x80000000) >> 31;
    unsigned short exponent = (tmp.u & 0x7F800000) >> 23;
    unsigned int significand = tmp.u & 0x7FFFFF;

    //     fprintf(stderr, "%d %d %d\n", sign, exponent, significand);

        // 1 : 5 : 10
    unsigned short fp16;
    if (exponent == 0) {
        // zero or denormal, always underflow
        fp16 = (sign << 15) | (0x00 << 10) | 0x00;
    } else if (exponent == 0xFF) {
        // infinity or NaN
        fp16 = (sign << 15) | (0x1F << 10) | (significand ? 0x200 : 0x00);
    } else {
        // normalized
        short newexp = exponent + (-127 + 15);
        if (newexp >= 31) {
            // overflow, return infinity
            fp16 = (sign << 15) | (0x1F << 10) | 0x00;
        } else if (newexp <= 0) {
            // underflow
            if (newexp >= -10) {
                // denormal half-precision
                unsigned short sig = (unsigned short)((significand | 0x800000) >> (14 - newexp));
                fp16 = (sign << 15) | (0x00 << 10) | sig;
            } else {
                // underflow
                fp16 = (sign << 15) | (0x00 << 10) | 0x00;
            }
        } else {
            fp16 = (unsigned short)((sign << 15) | (newexp << 10) | (significand >> 13));
        }
    }
    return fp16;
}
