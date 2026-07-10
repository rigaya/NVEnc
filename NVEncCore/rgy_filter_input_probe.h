// -----------------------------------------------------------------------------------------
// QSVEnc/NVEnc/VCEEnc by rigaya
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

#pragma once
#ifndef __RGY_FILTER_INPUT_PROBE_H__
#define __RGY_FILTER_INPUT_PROBE_H__

#include <cstring>
#include <string>
#include "rgy_avutil.h"

// Returns a short human-readable protocol name when the given input
// path cannot be re-opened safely for a private libav pre-scan, or
// nullptr when the path looks like a re-openable local file.
//
// "Re-openable" means: the filter can open a second AVFormatContext
// against the same source without consuming bytes from the primary
// reader. stdin and named-pipe protocols fail this property because
// the primary reader has already drained the stream by the time the
// filter's init() runs.
//
// Used by --vpp-ivtc expand= pre-scan, --vpp-descale kernel=auto
// probe, and --vpp-colorfix mode=auto / gray init-time analysis.
inline const char *unsupportedProbeProtocol(const std::string &filename) {
    if (filename == "-") {
        return "stdin";
    }
    if (filename.c_str() == std::strstr(filename.c_str(), R"(\\.\pipe\)")) {
        return "windows named pipe";
    }
    const char *protocol = avio_find_protocol_name(filename.c_str());
    if (protocol != nullptr && std::strcmp(protocol, "file") != 0) {
        return protocol;
    }
    return nullptr;
}

#endif // __RGY_FILTER_INPUT_PROBE_H__
