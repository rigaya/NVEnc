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

#pragma once
#ifndef __RGY_LOG_H__
#define __RGY_LOG_H__

#include <cstdint>
#include <string>
#include <memory>
#include "rgy_tchar.h"

//NVEnc.auo/QSVEnc.auoビルド時、/clrでは<thread>は使用できませんなどと出るので、
//前方宣言で回避する
namespace std {
    class mutex;
}

enum {
    RGY_LOG_TRACE = -3,
    RGY_LOG_DEBUG = -2,
    RGY_LOG_MORE  = -1,
    RGY_LOG_INFO  = 0,
    RGY_LOG_WARN  = 1,
    RGY_LOG_ERROR = 2,
    RGY_LOG_QUIET = 3,
};

class RGYLog {
protected:
    int m_nLogLevel = RGY_LOG_INFO;
    const TCHAR *m_pStrLog = nullptr;
    bool m_bHtml = false;
    std::unique_ptr<std::mutex> m_mtx;
    static const char *HTML_FOOTER;
public:
    RGYLog(const TCHAR *pLogFile, int log_level = RGY_LOG_INFO);
    virtual ~RGYLog();
    void init(const TCHAR *pLogFile, int log_level = RGY_LOG_INFO);
    void writeHtmlHeader();
    void writeFileHeader(const TCHAR *pDstFilename);
    void writeFileFooter();
    int getLogLevel() {
        return m_nLogLevel;
    }
    int setLogLevel(int newLogLevel) {
        int prevLogLevel = m_nLogLevel;
        m_nLogLevel = newLogLevel;
        return prevLogLevel;
    }
    bool logFileAvail() {
        return m_pStrLog != nullptr;
    }
    virtual void write_log(int log_level, const TCHAR *buffer, bool file_only = false);
    virtual void write(int log_level, const TCHAR *format, ...);
    virtual void write(int log_level, const wchar_t *format, va_list args);
    virtual void write(int log_level, const char *format, va_list args, uint32_t codepage);
    virtual void write_line(int log_level, const char *format, va_list args, uint32_t codepage);
};

#endif //__RGY_LOG_H__
