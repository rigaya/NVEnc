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
#include "rgy_util.h"
#include "rgy_codepage.h"

uint32_t GetPrivateProfileStringCP(const TCHAR *Section, const TCHAR *Key, const TCHAR *Default, TCHAR *buf, size_t nSize, const TCHAR *IniFile, uint32_t codepage) {
    FILE *fp = fopen(IniFile, "r");
    if (fp != NULL) {
        char buffer[4096];
        const auto len = fread(buffer, sizeof(char), sizeof(buffer)-1, fp);
        buffer[len] = '\0';
        if (codepage == CODE_PAGE_UNSET) {
            codepage = get_code_page(buffer, len);
        }
        _fseeki64(fp, 0, SEEK_SET);

        const auto section = std::string(_T("[")) + tchar_to_string(Section) + _T("]");
        const auto key = tchar_to_string(Key);

        bool bTargetSection = false;
        while (_fgetts(buffer, _countof(buffer), fp) != NULL) {
            auto line = trim(char_to_string(CP_THREAD_ACP, buffer, codepage));
            strcpy_s(buffer, line.c_str());
            if (buffer[0] == '[') {
                bTargetSection = strncmp(buffer, section.c_str(), section.length()) == 0;
            } else if (bTargetSection) {
                char *delim = strchr(buffer, '=');
                if (delim != NULL) {
                    *delim = '\0';
                    if (strcmp(buffer, key.c_str()) == 0) {
                        auto value = char_to_tstring(delim+1);
                        _tcscpy(buf, value.c_str());
                        return _tcslen(buf);
                    }
                }
            }
        }
    }
    _tcscpy(buf, Default);
    return _tcslen(buf);
}

uint32_t GetPrivateProfileIntCP(const TCHAR *Section, const TCHAR *Key, const uint32_t defaultValue, const TCHAR *IniFile, uint32_t codepage) {
    FILE *fp = fopen(IniFile, "r");
    if (fp != NULL) {
        char buffer[4096];
        const auto len = fread(buffer, sizeof(char), sizeof(buffer)-1, fp);
        buffer[len] = '\0';
        if (codepage == CODE_PAGE_UNSET) {
            codepage = get_code_page(buffer, len);
        }
        _fseeki64(fp, 0, SEEK_SET);

        const auto section = std::string(_T("[")) + tchar_to_string(Section) + _T("]");
        const auto key = tchar_to_string(Key);

        bool bTargetSection = false;
        while (_fgetts(buffer, _countof(buffer), fp) != NULL) {
            auto line = trim(char_to_string(CP_THREAD_ACP, buffer, codepage));
            strcpy_s(buffer, line.c_str());
            if (buffer[0] == '[') {
                bTargetSection = strncmp(buffer, section.c_str(), section.length()) == 0;
            } else if (bTargetSection) {
                char *delim = strchr(buffer, '=');
                if (delim != NULL) {
                    *delim = '\0';
                    if (strcmp(buffer, key.c_str()) == 0) {
                        try {
                            uint32_t value = std::stoul(delim+1);
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
