//  -----------------------------------------------------------------------------------------
//    NVEnc by rigaya
//  -----------------------------------------------------------------------------------------
//   ソースコードについて
//   ・無保証です。
//   ・本ソースコードを使用したことによるいかなる損害・トラブルについてrigayaは責任を負いません。
//   以上に了解して頂ける場合、本ソースコードの使用、複製、改変、再頒布を行って頂いて構いません。
//  -----------------------------------------------------------------------------------------

#pragma once

#include <stdio.h>
#include <tchar.h>
#include <string>
#include <mutex>
#include "NVEncUtil.h"
#include "NVEncVersion.h"
#include "ConvertCsp.h"
#include "NVEncParam.h"

class CNVEncLog {
protected:
    int m_nLogLevel = NV_LOG_INFO;
    const TCHAR *m_pStrLog = nullptr;
    bool m_bHtml = false;
    std::mutex m_mtx;
    static const char *HTML_FOOTER;
public:
    CNVEncLog(const TCHAR *pLogFile, int log_level = NV_LOG_INFO) {
        init(pLogFile, log_level);
    };
    virtual ~CNVEncLog() {
    };
    void init(const TCHAR *pLogFile, int log_level = NV_LOG_INFO);
    void writeHtmlHeader();
    void writeFileHeader(const TCHAR *pDstFilename);
    void writeFileFooter();
    int getLogLevel() {
        return m_nLogLevel;
    }
    virtual void operator()(int log_level, const TCHAR *format, ...);
};
