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

#if !(defined(_WIN32) || defined(_WIN64))

#include <cstdio>
#include <regex>
#include <string>
#include "rgy_tchar.h"
#include "rgy_osdep.h"

uint32_t GetPrivateProfileString(const TCHAR *Section, const TCHAR *Key, const TCHAR *Default, TCHAR *buf, size_t nSize, const TCHAR *IniFile) {
    FILE *fp = fopen(IniFile, "r");
    if (fp != NULL) {
        TCHAR buffer[1024];

        auto tsection = std::basic_string<TCHAR>(_T("["));
        tsection += Section;
        tsection += _T("]");

        bool bTargetSection = false;
        while (_fgetts(buffer, _countof(buffer), fp) != NULL) {
            if (buffer[0] == _T('[')) {
                bTargetSection = (_tcscmp(buffer, tsection.c_str()) == 0);
            } else if (bTargetSection) {
                char *pDelim = _tcschr(buffer, _T('='));
                if (pDelim != NULL) {
                    *pDelim = _T('\0');
                    if (_tcscmp(buffer, Key) == 0) {
                        _tcscpy(buf, pDelim+1);
                        return _tcslen(buf);
                    }
                }
            }
        }
    }
    _tcscpy(buf, Default);
    return _tcslen(buf);
}

uint32_t GetPrivateProfileInt(const TCHAR *Section, const TCHAR *Key, const uint32_t defaultValue, const TCHAR *IniFile) {
    FILE *fp = fopen(IniFile, "r");
    if (fp != NULL) {
        TCHAR buffer[1024];

        auto tsection = std::basic_string<TCHAR>(_T("["));
        tsection += Section;
        tsection += _T("]");

        bool bTargetSection = false;
        while (_fgetts(buffer, _countof(buffer), fp) != NULL) {
            if (buffer[0] == _T('[')) {
                bTargetSection = (_tcscmp(buffer, tsection.c_str()) == 0);
            } else if (bTargetSection) {
                char *pDelim = _tcschr(buffer, _T('='));
                if (pDelim != NULL) {
                    *pDelim = _T('\0');
                    if (_tcscmp(buffer, Key) == 0) {
                        try {
                            uint32_t value = std::stoul(pDelim+1);
                            return value;
                        } catch (...) {
                            continue;
                        }
                    }
                }
            }
        }
    }
    return defaultValue;
}

#endif //#if !(defined(_WIN32) || defined(_WIN64))
