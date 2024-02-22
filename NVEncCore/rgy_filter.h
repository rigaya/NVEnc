// -----------------------------------------------------------------------------------------
//     QSVEnc/VCEEnc/rkmppenc by rigaya
// -----------------------------------------------------------------------------------------
// The MIT License
//
// Copyright (c) 2019-2021 rigaya
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
// IABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.
//
// ------------------------------------------------------------------------------------------

#pragma once
#ifndef __RGY_FILTER_H__
#define __RGY_FILTER_H__

#include <cstdint>
#include "rgy_util.h"
#include "rgy_log.h"
#include "rgy_frame_info.h"
#include "convert_csp.h"
#include "rgy_prm.h"

struct AVPacket;

class RGYFilterParam {
public:
    RGYFrameInfo frameIn;
    RGYFrameInfo frameOut;
    rgy_rational<int> baseFps;
    bool bOutOverwrite;

    RGYFilterParam() : frameIn(), frameOut(), baseFps(), bOutOverwrite(false) {};
    virtual ~RGYFilterParam() {};
    virtual tstring print() const { return _T(""); };
};


enum FILTER_PATHTHROUGH_FRAMEINFO : uint32_t {
    FILTER_PATHTHROUGH_NONE      = 0x00u,
    FILTER_PATHTHROUGH_TIMESTAMP = 0x01u,
    FILTER_PATHTHROUGH_FLAGS     = 0x02u,
    FILTER_PATHTHROUGH_PICSTRUCT = 0x04u,
    FILTER_PATHTHROUGH_DATA      = 0x07u,

    FILTER_PATHTHROUGH_ALL       = 0x0fu,
};

static FILTER_PATHTHROUGH_FRAMEINFO operator|(FILTER_PATHTHROUGH_FRAMEINFO a, FILTER_PATHTHROUGH_FRAMEINFO b) {
    return (FILTER_PATHTHROUGH_FRAMEINFO)((uint32_t)a | (uint32_t)b);
}

static FILTER_PATHTHROUGH_FRAMEINFO operator|=(FILTER_PATHTHROUGH_FRAMEINFO &a, FILTER_PATHTHROUGH_FRAMEINFO b) {
    a = a | b;
    return a;
}

static FILTER_PATHTHROUGH_FRAMEINFO operator&(FILTER_PATHTHROUGH_FRAMEINFO a, FILTER_PATHTHROUGH_FRAMEINFO b) {
    return (FILTER_PATHTHROUGH_FRAMEINFO)((uint32_t)a & (uint32_t)b);
}

static FILTER_PATHTHROUGH_FRAMEINFO operator&=(FILTER_PATHTHROUGH_FRAMEINFO &a, FILTER_PATHTHROUGH_FRAMEINFO b) {
    a = a & b;
    return a;
}

static FILTER_PATHTHROUGH_FRAMEINFO operator~(FILTER_PATHTHROUGH_FRAMEINFO a) {
    return (FILTER_PATHTHROUGH_FRAMEINFO)(~((uint32_t)a));
}

class RGYFilterPerf {
public:
    RGYFilterPerf() : m_filterTimeMs(0.0), m_runCount(0) {};
    virtual ~RGYFilterPerf() { };

    double GetAvgTimeElapsed() const {
        return (m_runCount > 0) ? m_filterTimeMs / (double)m_runCount : 0.0;
    }
    virtual RGY_ERR checkPerformace(void *event_start, void *event_fin) = 0;
protected:
    void setTime(double time) {
        m_filterTimeMs += time;
        m_runCount++;
    }
    double m_filterTimeMs;
    int64_t m_runCount;
};

class RGYFilterBase {
public:
    RGYFilterBase();
    virtual ~RGYFilterBase();
    virtual RGY_ERR init(shared_ptr<RGYFilterParam> pParam, shared_ptr<RGYLog> pPrintMes) = 0;

    const tstring& name() const {
        return m_name;
    }
    const tstring& GetInputMessage() const {
        return m_infoStr;
    }
    const RGYFilterParam *GetFilterParam() const {
        return m_param.get();
    }
    virtual RGY_ERR addStreamPacket(AVPacket *pkt) { UNREFERENCED_PARAMETER(pkt); return RGY_ERR_UNSUPPORTED; };
    virtual int targetTrackIdx() { return 0; };
    virtual void setCheckPerformance(const bool check) = 0;
    double GetAvgTimeElapsed() { return (m_perfMonitor) ? m_perfMonitor->GetAvgTimeElapsed() : 0.0; }
protected:
    virtual RGY_ERR AllocFrameBuf(const RGYFrameInfo &frame, int frames) = 0;
    virtual void close() = 0;

    void AddMessage(RGYLogLevel log_level, const tstring &str) {
        if (m_pLog == nullptr || log_level < m_pLog->getLogLevel(RGY_LOGT_VPP)) {
            return;
        }
        auto lines = split(str, _T("\n"));
        for (const auto &line : lines) {
            if (line[0] != _T('\0')) {
                m_pLog->write(log_level, RGY_LOGT_VPP, (m_name + _T(": ") + line + _T("\n")).c_str());
            }
        }
    }
    void AddMessage(RGYLogLevel log_level, const TCHAR *format, ...) {
        if (m_pLog == nullptr || log_level < m_pLog->getLogLevel(RGY_LOGT_VPP)) {
            return;
        }

        va_list args;
        va_start(args, format);
        int len = _vsctprintf(format, args) + 1; // _vscprintf doesn't count terminating '\0'
        tstring buffer;
        buffer.resize(len, _T('\0'));
        _vstprintf_s(&buffer[0], len, format, args);
        va_end(args);
        AddMessage(log_level, buffer);
    }
    void setFilterInfo(const tstring &info) {
        m_infoStr = info;
        AddMessage(RGY_LOG_DEBUG, info);
    }

    tstring m_name;
    tstring m_infoStr;
    shared_ptr<RGYLog> m_pLog;  //ログ出力
    std::shared_ptr<RGYFilterParam> m_param;
    FILTER_PATHTHROUGH_FRAMEINFO m_pathThrough;
    std::unique_ptr<RGYFilterPerf> m_perfMonitor;
};

#endif //__RGY_FILTER_H__
