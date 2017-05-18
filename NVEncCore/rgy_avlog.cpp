// -----------------------------------------------------------------------------------------
// NVEnc by rigaya
// -----------------------------------------------------------------------------------------
//
// The MIT License
//
// Copyright (c) 2014-2016 rigaya
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

#include <memory>
#include "NVEncVersion.h"

#if ENABLE_AVCUVID_READER
#include <atomic>
#include "rgy_avlog.h"

static std::weak_ptr<RGYLog> g_pQSVLog;
static int print_prefix = 1;
static std::atomic<bool> g_bSetCustomLog(false);

static void av_qsv_log_callback(void *ptr, int level, const char *fmt, va_list vl) {
    if (auto pQSVLog = g_pQSVLog.lock()) {
        const int qsv_log_level = log_level_av2qsv(level);
        if (qsv_log_level >= pQSVLog->getLogLevel() && pQSVLog->logFileAvail()) {
            char mes[4096];
            av_log_format_line(ptr, level, fmt, vl, mes, sizeof(mes), &print_prefix);
            pQSVLog->write_log(qsv_log_level, char_to_tstring(mes, CP_UTF8).c_str(), true);
        }
    }
    av_log_default_callback(ptr, level, fmt, vl);
}

void av_qsv_log_set(std::shared_ptr<RGYLog>& pQSVLog) {
    g_pQSVLog = pQSVLog;
    g_bSetCustomLog = true;
    av_log_set_callback(av_qsv_log_callback);
}

void av_qsv_log_free() {
    if (g_bSetCustomLog) {
        g_bSetCustomLog = false;
        av_log_set_callback(av_log_default_callback);
        g_pQSVLog.reset();
    }
}

#endif //ENABLE_AVCUVID_READER
