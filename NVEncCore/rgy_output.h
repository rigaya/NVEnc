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
#ifndef __RGY_OUTPUT_H__
#define __RGY_OUTPUT_H__

#include <memory>
#include <vector>
#include <unordered_map>
#include <mutex>
#include "rgy_osdep.h"
#include "rgy_tchar.h"
#include "rgy_log.h"
#include "rgy_status.h"
#include "rgy_avutil.h"
#include "rgy_bitstream.h"
#include "rgy_input.h"
#if ENCODER_NVENC
#include "NVEncUtil.h"
#endif //#if ENCODER_NVENC
#if ENCODER_QSV
#include "qsv_util.h"
#endif //#if ENCODER_QSV
#if ENCODER_VCEENC
#include "vce_util.h"
#endif //#if ENCODER_VCEENC

using std::unique_ptr;
using std::shared_ptr;

enum OutputType {
    OUT_TYPE_NONE = 0,
    OUT_TYPE_BITSTREAM,
    OUT_TYPE_SURFACE
};

struct RGYTimestampMapVal {
    int64_t timestamp, inputFrameId, duration;

    RGYTimestampMapVal() : timestamp(-1), inputFrameId(-1), duration(-1) {};
    RGYTimestampMapVal(int64_t timestamp_, int64_t inputFrameId_, int64_t duration_) : timestamp(timestamp_), inputFrameId(inputFrameId_), duration(duration_) {};
};

class RGYTimestamp {
private:
    std::unordered_map<int64_t, RGYTimestampMapVal> m_frame;
    std::mutex mtx;
    int64_t last_add_pts;
    int64_t last_check_pts;
    int64_t last_input_frame_id;
    int64_t offset;
    int64_t last_clean_id;
public:
    RGYTimestamp() : m_frame(), mtx(), last_add_pts(-1), last_check_pts(-1), offset(0), last_clean_id(-1) {};
    ~RGYTimestamp() {};
    void clear() {
        std::lock_guard<std::mutex> lock(mtx);
        m_frame.clear();
        last_check_pts = -1;
        offset = 0;
    }
    void add(int64_t pts, int64_t inputFrameId, int64_t duration) {
        std::lock_guard<std::mutex> lock(mtx);
        if (last_add_pts >= 0) { // 前のフレームのdurationの更新
            auto& last_add_pos = m_frame.find(last_add_pts)->second;
            last_add_pos.duration = pts - last_add_pos.timestamp;
            if (duration == 0) duration = last_add_pos.duration;
        }
        m_frame[pts] = RGYTimestampMapVal(pts, inputFrameId, duration);
        last_add_pts = pts;
    }
    RGYTimestampMapVal check(int64_t pts) {
        if (last_check_pts < 0 && pts > 0) {
            offset = -pts;
        }
        std::lock_guard<std::mutex> lock(mtx);
        pts += offset;
        auto pos = m_frame.find(pts);
        if (pos == m_frame.end()) {
            auto& last_check_pos = m_frame.find(last_check_pts)->second;
            pts = last_check_pos.timestamp + last_check_pos.duration / 2;
            auto next_pts = last_check_pos.timestamp + last_check_pos.duration;
            last_check_pos.duration = pts - last_check_pos.timestamp;
            m_frame[pts] = RGYTimestampMapVal(pts, last_input_frame_id, next_pts - pts);
            pos = m_frame.find(pts);
        }
        last_input_frame_id = pos->second.inputFrameId;
        last_check_pts = pos->second.timestamp;
        return pos->second;
    }
    RGYTimestampMapVal get(int64_t pts) {
        std::lock_guard<std::mutex> lock(mtx);
        auto pos = m_frame.find(pts);
        if (pos == m_frame.end()) {
            return RGYTimestampMapVal();
        }
        auto& ret = pos->second;
        if (ret.inputFrameId >= last_clean_id + 64) {
            for (auto it = m_frame.begin(); it != m_frame.end();) {
                if (it->second.inputFrameId < ret.inputFrameId - 32) {
                    it = m_frame.erase(it);
                } else {
                    it++;
                }
            }
            last_clean_id = ret.inputFrameId;
        }
        return ret;
    }
};

class RGYOutput {
public:
    RGYOutput();
    virtual ~RGYOutput();

    RGY_ERR Init(const TCHAR *strFileName, const VideoInfo *videoOutputInfo, const void *prm, shared_ptr<RGYLog> log, shared_ptr<EncodeStatus> encSatusInfo) {
        Close();
        m_printMes = log;
        m_encSatusInfo = encSatusInfo;
        if (videoOutputInfo) {
            m_VideoOutputInfo = *videoOutputInfo;
        }
        return Init(strFileName, videoOutputInfo, prm);
    }

    virtual RGY_ERR WriteNextFrame(RGYBitstream *pBitstream) = 0;
    virtual RGY_ERR WriteNextFrame(RGYFrame *pSurface) = 0;
    virtual void Close();

    virtual bool outputStdout() {
        return m_outputIsStdout;
    }

    virtual OutputType getOutType() {
        return m_OutType;
    }
    virtual void WaitFin() {
        return;
    }

    const TCHAR *GetOutputMessage() {
        const TCHAR *mes = m_strOutputInfo.c_str();
        return (mes) ? mes : _T("");
    }
    void AddMessage(RGYLogLevel log_level, const tstring& str) {
        if (m_printMes == nullptr || log_level < m_printMes->getLogLevel(RGY_LOGT_OUT)) {
            return;
        }
        auto lines = split(str, _T("\n"));
        for (const auto& line : lines) {
            if (line[0] != _T('\0')) {
                m_printMes->write(log_level, RGY_LOGT_OUT, (m_strWriterName + _T(": ") + line + _T("\n")).c_str());
            }
        }
    }
    void AddMessage(RGYLogLevel log_level, const TCHAR *format, ... ) {
        if (m_printMes == nullptr || log_level < m_printMes->getLogLevel(RGY_LOGT_OUT)) {
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
protected:
    virtual RGY_ERR Init(const TCHAR *strFileName, const VideoInfo *pOutputInfo, const void *prm) = 0;

    shared_ptr<EncodeStatus> m_encSatusInfo;
    unique_ptr<FILE, fp_deleter>  m_fDest;
    bool        m_outputIsStdout;
    bool        m_inited;
    bool        m_noOutput;
    OutputType  m_OutType;
    bool        m_sourceHWMem;
    bool        m_y4mHeaderWritten;
    tstring     m_strWriterName;
    tstring     m_strOutputInfo;
    VideoInfo   m_VideoOutputInfo;
    shared_ptr<RGYLog> m_printMes;  //ログ出力
    unique_ptr<char, malloc_deleter>            m_outputBuffer;
    unique_ptr<uint8_t, aligned_malloc_deleter> m_readBuffer;
    unique_ptr<uint8_t, aligned_malloc_deleter> m_UVBuffer;
};

struct RGYOutputRawPrm {
    bool benchmark;
    int bufSizeMB;
    RGY_CODEC codecId;
    const HEVCHDRSei *hedrsei;
    DOVIRpu *doviRpu;
    RGYTimestamp *vidTimestamp;
};

class RGYOutputRaw : public RGYOutput {
public:

    RGYOutputRaw();
    virtual ~RGYOutputRaw();

    virtual RGY_ERR WriteNextFrame(RGYBitstream *pBitstream) override;
    virtual RGY_ERR WriteNextFrame(RGYFrame *pSurface) override;
protected:
    virtual RGY_ERR Init(const TCHAR *strFileName, const VideoInfo *pOutputInfo, const void *prm) override;

    vector<uint8_t> m_outputBuf2;
    vector<uint8_t> m_seiNal;
    DOVIRpu *m_doviRpu;
    RGYTimestamp *m_timestamp;
    int64_t m_prevInputFrameId;
#if ENABLE_AVSW_READER
    unique_ptr<AVBSFContext, RGYAVDeleter<AVBSFContext>> m_pBsfc;
#endif //#if ENABLE_AVSW_READER
    decltype(parse_nal_unit_h264_c) *parse_nal_h264; // H.264用のnal unit分解関数へのポインタ
    decltype(parse_nal_unit_hevc_c) *parse_nal_hevc; // HEVC用のnal unit分解関数へのポインタ
};

std::unique_ptr<HEVCHDRSei> createHEVCHDRSei(const std::string &maxCll, const std::string &masterDisplay, CspTransfer atcSei, const RGYInput *reader);

RGY_ERR initWriters(
    shared_ptr<RGYOutput> &pFileWriter,
    vector<shared_ptr<RGYOutput>> &pFileWriterListAudio,
    shared_ptr<RGYInput> &pFileReader,
    vector<shared_ptr<RGYInput>> &audioReaders,
    RGYParamCommon *common,
    const VideoInfo *input,
    const RGYParamControl *ctrl,
    const VideoInfo outputVideoInfo,
    const sTrimParam &trimParam,
    const rgy_rational<int> outputTimebase,
#if ENABLE_AVSW_READER
    const vector<unique_ptr<AVChapter>> &chapters,
#endif //#if ENABLE_AVSW_READER
    const HEVCHDRSei *hedrsei,
    DOVIRpu *doviRpu,
    RGYTimestamp *vidTimestamp,
    const bool videoDtsUnavailable,
    const bool benchmark,
    shared_ptr<EncodeStatus> pStatus,
    shared_ptr<CPerfMonitor> pPerfMonitor,
    shared_ptr<RGYLog> log
);

#if ENCODER_QSV

struct YUVWriterParam {
    bool bY4m;
};

class RGYOutFrame : public RGYOutput {
public:

    RGYOutFrame();
    virtual ~RGYOutFrame();

    virtual RGY_ERR WriteNextFrame(RGYBitstream *pBitstream) override;
    virtual RGY_ERR WriteNextFrame(RGYFrame *pSurface) override;
protected:
    virtual RGY_ERR Init(const TCHAR *strFileName, const VideoInfo *pOutputInfo, const void *prm) override;

    bool m_bY4m;
};

#endif //#if ENCODER_QSV

#endif //__RGY_OUTPUT_H__
