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
// --------------------------------------------------------------------------------------------

#include "rgy_timecode.h"

int64_t rational_rescale(int64_t v, rgy_rational<int> from, rgy_rational<int> to);

RGY_ERR RGYTimecode::init(const tstring &filename) {
    FILE *filep = nullptr;
    if (_tfopen_s(&filep, filename.c_str(), _T("w")) != 0 || filep == nullptr) {
        return RGY_ERR_FILE_OPEN;
    }
    fp.reset(filep);
    fprintf(fp.get(), "# timecode format v2\n");
    return RGY_ERR_NONE;
}

void RGYTimecode::write(int64_t timestamp, rgy_rational<int> timebase) {
    fprintf(fp.get(), "%.6lf\n", (double)timestamp * timebase.qdouble() * 1000.0);
    fflush(fp.get());
}

RGYTimecodeReader::RGYTimecodeReader() :
    m_fp(),
    m_filename(),
    m_timeBaseTimecode({ 1, 120000 }),
    m_prevPts(-1),
    m_prevDuration(0) {

}

RGYTimecodeReader::~RGYTimecodeReader() { m_fp.reset(); };

RGY_ERR RGYTimecodeReader::init(const tstring &filename, rgy_rational<int> timeBaseTimecode) {
    FILE *fp = nullptr;
    int error = 0;
    if (0 != (error = _tfopen_s(&fp, filename.c_str(), _T("rb"))) || fp == nullptr) {
        //AddMessage(RGY_LOG_ERROR, _T("Failed to open timecode file \"%s\": %s.\n"), prm->timecode.c_str(), _tcserror(error));
        return RGY_ERR_FILE_OPEN;
    }
    m_fp = std::unique_ptr<FILE, fp_deleter>(fp, fp_deleter());
    m_filename = filename;
    if (timeBaseTimecode.is_valid()) {
        m_timeBaseTimecode = timeBaseTimecode;
    }
    m_prevPts = -1;
    m_prevDuration = 0;
    return RGY_ERR_NONE;
}

std::tuple<RGY_ERR, double> RGYTimecodeReader::getValue() {
    double value = 0.0;
    char buffer[1024] = { 0 };
    while (fgets(buffer, _countof(buffer) - 1, m_fp.get()) != nullptr) {
        if (buffer[0] == '#') continue; // コメント

        if (sscanf_s(buffer, "%lf", &value) != 1) {
            return { RGY_ERR_INVALID_DATA_TYPE, 0.0 };
        }
        return { RGY_ERR_NONE, value };
    }
    return { RGY_ERR_MORE_DATA, 0.0 };
}

RGY_ERR RGYTimecodeReader::read(int64_t& timestamp, int64_t& duration) {
    auto [ err, value ] = getValue();
    if (err == RGY_ERR_INVALID_DATA_TYPE) {
        return RGY_ERR_INVALID_DATA_TYPE;
    } else if (err == RGY_ERR_MORE_DATA) { //EOF
        if (m_prevPts < 0) {
            return RGY_ERR_MORE_DATA; // EOF
        }
        timestamp = m_prevPts;
        duration = m_prevDuration;
        m_prevPts = -1;
        m_prevDuration = 0;
        return RGY_ERR_NONE;
    } else if (err != RGY_ERR_NONE) {
        return err;
    }
    const auto nextPts = rational_rescale((int64_t)(value * 1000.0 + 0.5), rgy_rational<int>(1, 1000000), m_timeBaseTimecode);
    if (m_prevPts < 0) {
        m_prevPts = nextPts;
        m_prevDuration = 0;
        return read(timestamp, duration);
    }
    timestamp = m_prevPts;
    duration = nextPts - m_prevPts;
    m_prevPts = nextPts;
    m_prevDuration = duration;
    return RGY_ERR_NONE;
}
