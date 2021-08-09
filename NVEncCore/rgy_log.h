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
#include <array>
#include "rgy_tchar.h"

//NVEnc.auo/QSVEnc.auoビルド時、/clrでは<thread>は使用できませんなどと出るので、
//前方宣言で回避する
namespace std {
    class mutex;
}

enum RGYLogLevel {
    RGY_LOG_TRACE = -3,
    RGY_LOG_DEBUG = -2,
    RGY_LOG_MORE  = -1,
    RGY_LOG_INFO  = 0,
    RGY_LOG_WARN  = 1,
    RGY_LOG_ERROR = 2,
    RGY_LOG_QUIET = 3,
};

static const std::array<std::pair<RGYLogLevel, const TCHAR *>, RGY_LOG_QUIET - RGY_LOG_TRACE + 1> RGY_LOG_LEVEL_STR = {
    std::pair<RGYLogLevel, const TCHAR *>{ RGY_LOG_TRACE,   _T("trace")},
    std::pair<RGYLogLevel, const TCHAR *>{ RGY_LOG_DEBUG,   _T("debug")},
    std::pair<RGYLogLevel, const TCHAR *>{ RGY_LOG_MORE,    _T("more")},
    std::pair<RGYLogLevel, const TCHAR *>{ RGY_LOG_INFO,    _T("info")},
    std::pair<RGYLogLevel, const TCHAR *>{ RGY_LOG_WARN,    _T("warn")},
    std::pair<RGYLogLevel, const TCHAR *>{ RGY_LOG_ERROR,   _T("error")},
    std::pair<RGYLogLevel, const TCHAR *>{ RGY_LOG_QUIET,   _T("quiet")}
};

enum RGYLogType {
    RGY_LOGT_ALL = -1,
    RGY_LOGT_APP,
    RGY_LOGT_CORE,
    RGY_LOGT_HDR10PLUS = RGY_LOGT_CORE,
    RGY_LOGT_DEV,
    RGY_LOGT_DEC,
    RGY_LOGT_IN,
    RGY_LOGT_OUT,
    RGY_LOGT_VPP,
    RGY_LOGT_VPP_BUILD = RGY_LOGT_VPP,
    RGY_LOGT_AMF,
    RGY_LOGT_OPENCL,
    RGY_LOGT_LIBAV,
    RGY_LOGT_LIBASS,
    RGY_LOGT_PERF_MONITOR,
    RGY_LOGT_CAPION2ASS,
};

static const std::array<std::pair<RGYLogType, const TCHAR *>, RGY_LOGT_CAPION2ASS - RGY_LOGT_ALL + 1> RGY_LOG_TYPE_STR = {
    std::pair<RGYLogType, const TCHAR *>{ RGY_LOGT_ALL,    _T("all")},
    std::pair<RGYLogType, const TCHAR *>{ RGY_LOGT_APP,    _T("app")},
    std::pair<RGYLogType, const TCHAR *>{ RGY_LOGT_DEV,    _T("device")},
    std::pair<RGYLogType, const TCHAR *>{ RGY_LOGT_CORE,   _T("core")},
    std::pair<RGYLogType, const TCHAR *>{ RGY_LOGT_DEC,    _T("decoder")},
    std::pair<RGYLogType, const TCHAR *>{ RGY_LOGT_IN,     _T("input")},
    std::pair<RGYLogType, const TCHAR *>{ RGY_LOGT_OUT,    _T("output")},
    std::pair<RGYLogType, const TCHAR *>{ RGY_LOGT_VPP,    _T("vpp")},
    std::pair<RGYLogType, const TCHAR *>{ RGY_LOGT_AMF,    _T("amf")},
    std::pair<RGYLogType, const TCHAR *>{ RGY_LOGT_OPENCL, _T("opencl")},
    std::pair<RGYLogType, const TCHAR *>{ RGY_LOGT_LIBAV,  _T("libav")},
    std::pair<RGYLogType, const TCHAR *>{ RGY_LOGT_LIBASS, _T("libass")},
    std::pair<RGYLogType, const TCHAR *>{ RGY_LOGT_PERF_MONITOR, _T("perfmonitor")},
    std::pair<RGYLogType, const TCHAR *>{ RGY_LOGT_CAPION2ASS,   _T("caption2ass")}
};

struct RGYParamLogLevel {
private:
    RGYLogLevel appcore_;
    RGYLogLevel appdevice_;
    RGYLogLevel appdecode_;
    RGYLogLevel appinput_;
    RGYLogLevel appoutput_;
    RGYLogLevel appvpp_;
    RGYLogLevel amf_;
    RGYLogLevel opencl_;
    RGYLogLevel libav_;
    RGYLogLevel libass_;
    RGYLogLevel perfmonitor_;
    RGYLogLevel caption2ass_;
public:
    RGYParamLogLevel();
    RGYParamLogLevel(const RGYLogLevel level);
    bool operator==(const RGYParamLogLevel &x) const;
    bool operator!=(const RGYParamLogLevel &x) const;
    RGYLogLevel set(const RGYLogLevel newLogLevel, const RGYLogType type);
    RGYLogLevel get(const RGYLogType type) const {
        switch (type) {
        case RGY_LOGT_DEC: return appdecode_;
        case RGY_LOGT_DEV: return appdevice_;
        case RGY_LOGT_IN: return appinput_;
        case RGY_LOGT_OUT: return appoutput_;
        case RGY_LOGT_VPP: return appvpp_;
        case RGY_LOGT_AMF: return amf_;
        case RGY_LOGT_OPENCL: return opencl_;
        case RGY_LOGT_LIBAV: return libav_;
        case RGY_LOGT_LIBASS: return libass_;
        case RGY_LOGT_PERF_MONITOR: return perfmonitor_;
        case RGY_LOGT_CAPION2ASS: return caption2ass_;
        case RGY_LOGT_APP:
        case RGY_LOGT_CORE:
        case RGY_LOGT_ALL:
        default:
            return appcore_;
        }
    };
    tstring to_string() const;
};

int rgy_print_stderr(int log_level, const TCHAR *mes, void *handle = NULL);

class RGYLog {
protected:
    RGYParamLogLevel m_nLogLevel;
    const TCHAR *m_pStrLog = nullptr;
    bool m_bHtml = false;
    std::unique_ptr<std::mutex> m_mtx;
    static const char *HTML_FOOTER;
public:
    RGYLog(const TCHAR *pLogFile, const RGYLogLevel log_level = RGY_LOG_INFO);
    RGYLog(const TCHAR *pLogFile, const RGYParamLogLevel& log_level);
    virtual ~RGYLog();
    void init(const TCHAR *pLogFile, const RGYParamLogLevel& log_level);
    void writeHtmlHeader();
    void writeFileHeader(const TCHAR *pDstFilename);
    void writeFileFooter();
    RGYParamLogLevel getLogLevelAll() const {
        return m_nLogLevel;
    }
    RGYParamLogLevel setLogLevelAll(const RGYParamLogLevel& newLogLevel) {
        auto prev = m_nLogLevel;
        m_nLogLevel = newLogLevel;
        return prev;
    }
    RGYLogLevel getLogLevel(const RGYLogType type) const {
        return m_nLogLevel.get(type);
    }
    RGYLogLevel setLogLevel(const RGYLogLevel newLogLevel, const RGYLogType type) {
        return m_nLogLevel.set(newLogLevel, type);
    }
    bool logFileAvail() {
        return m_pStrLog != nullptr;
    }
    virtual void write_log(RGYLogLevel log_level, const RGYLogType logtype, const TCHAR *buffer, bool file_only = false);
    virtual void write(RGYLogLevel log_level, const RGYLogType logtype, const TCHAR *format, ...);
    virtual void write(RGYLogLevel log_level, const RGYLogType logtype, const wchar_t *format, va_list args);
    virtual void write(RGYLogLevel log_level, const RGYLogType logtype, const char *format, va_list args, uint32_t codepage);
    virtual void write_line(RGYLogLevel log_level, const RGYLogType logtype, const char *format, va_list args, uint32_t codepage);
};

#endif //__RGY_LOG_H__
