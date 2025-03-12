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
#include "NVEncParam.h"
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
    int64_t timestamp, inputFrameId, encodeFrameId, duration;
    std::vector<std::shared_ptr<RGYFrameData>> dataList;

    RGYTimestampMapVal() : timestamp(-1), inputFrameId(-1), encodeFrameId(-1), duration(-1), dataList() {};
    RGYTimestampMapVal(int64_t timestamp_, int64_t inputFrameId_, int64_t encodeFrameId_, int64_t duration_, std::vector<std::shared_ptr<RGYFrameData>>& datalist)
        : timestamp(timestamp_), inputFrameId(inputFrameId_), encodeFrameId(encodeFrameId_), duration(duration_), dataList(datalist) {};
    void addMetadata(std::shared_ptr<RGYFrameData>& data) { dataList.push_back(data); }
    void addMetadata(std::vector<std::shared_ptr<RGYFrameData>>& list) { dataList.insert(dataList.end(), list.begin(), list.end()); }
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
    bool timestampPassThrough;
    bool noDurationFix; // durationの修正を行わない場合にtrue
public:
    RGYTimestamp(bool timestampPassThrough_, bool noDurationFix_) : m_frame(), mtx(), last_add_pts(-1), last_check_pts(-1), offset(0), last_clean_id(-1), timestampPassThrough(timestampPassThrough_), noDurationFix(noDurationFix_) {};
    ~RGYTimestamp() {};
    void clear() {
        std::lock_guard<std::mutex> lock(mtx);
        m_frame.clear();
        last_check_pts = -1;
        offset = 0;
    }
    void add(int64_t pts, int64_t inputFrameId, int64_t encodeFrameId, int64_t duration, std::vector<std::shared_ptr<RGYFrameData>> metadatalist) {
        std::lock_guard<std::mutex> lock(mtx);
        if (last_add_pts >= 0) { // 前のフレームのdurationの更新
            auto& last_add_pos = m_frame.find(last_add_pts)->second;
            if (!noDurationFix) last_add_pos.duration = pts - last_add_pos.timestamp;
            if (duration == 0) duration = last_add_pos.duration;
        }
        m_frame[pts] = RGYTimestampMapVal(pts, inputFrameId, encodeFrameId, duration, metadatalist);
        last_add_pts = pts;
    }
    RGYTimestampMapVal check(int64_t pts) {
        if (last_check_pts < 0 && pts > 0 && !timestampPassThrough) {
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
            m_frame[pts] = RGYTimestampMapVal(pts, last_input_frame_id, last_check_pos.encodeFrameId, next_pts - pts, last_check_pos.dataList);
            pos = m_frame.find(pts);
        }
        last_input_frame_id = pos->second.inputFrameId;
        last_check_pts = pos->second.timestamp;
        return pos->second;
    }
    void clean(const int64_t current_id) {
        if (current_id >= last_clean_id + 64) {
            for (auto it = m_frame.begin(); it != m_frame.end();) {
                if (it->second.inputFrameId < current_id - 64) {
                    it = m_frame.erase(it);
                } else {
                    it++;
                }
            }
            last_clean_id = current_id;
        }
    }
    RGYTimestampMapVal getByEncodeFrameID(const int64_t id) {
        std::lock_guard<std::mutex> lock(mtx);
        auto pos = m_frame.end();
        for (auto it = m_frame.begin(); it != m_frame.end(); it++) {
            if (it->second.encodeFrameId == id) {
                pos = it;
                break;
            }
        }
        if (pos == m_frame.end()) {
            return RGYTimestampMapVal();
        }
        auto ret = pos->second;
        clean(ret.inputFrameId);
        return ret;
    }
    RGYTimestampMapVal get(int64_t pts) {
        std::lock_guard<std::mutex> lock(mtx);
        auto pos = m_frame.find(pts);
        if (pos == m_frame.end()) {
            return RGYTimestampMapVal();
        }
        auto ret = pos->second;
        clean(ret.inputFrameId);
        return ret;
    }
};

class RGYDurationCheck {
private:
    std::array<int64_t, 64> m_ts;
    int m_idx;
    std::unordered_map<int, int64_t> m_duration;
public:
    RGYDurationCheck() : m_ts(), m_idx(0), m_duration() {};
    void add(int64_t ts) {
        m_ts[m_idx] = ts;
        m_idx++;
        if (m_idx >= m_ts.size()) {
            std::sort(m_ts.begin(), m_ts.end());
            // [ 0 - 1 ] ... [ 30 - 31 ] までのフレーム長さをチェック
            for (size_t i = 1; i < m_ts.size() / 2; i++) {
                const int duration = (int)(m_ts[i] - m_ts[i - 1]);
                if (m_duration.count(duration) > 0) {
                    m_duration[duration]++;
                } else {
                    m_duration[duration] = 1;
                }
            }
            // [ 31 - 63 ] までを先頭に移動
            memmove(m_ts.data(), m_ts.data() + (m_ts.size() / 2 - 1), (m_ts.size() / 2 + 1) * sizeof(int64_t));
            m_idx = (int)(m_ts.size() / 2) + 1;
        }
    }
    std::unordered_map<int, int64_t> getDuration() {
        std::sort(m_ts.begin(), m_ts.begin() + m_idx);
        for (int i = 1; i < m_idx; i++) {
            const int duration = (int)(m_ts[i] - m_ts[i - 1]);
            if (m_duration.count(duration) > 0) {
                m_duration[duration]++;
            } else {
                m_duration[duration] = 1;
            }
        }
        m_idx = 0;
        return m_duration;
    }
};

class RGYOutputBSF {
public:
    RGYOutputBSF(AVBSFContext *bsf, RGY_CODEC codec, tstring strWriterName, shared_ptr<RGYLog> log);
    virtual ~RGYOutputBSF();
    RGY_ERR applyBitstreamFilter(RGYBitstream *bitstream);
protected:
    void AddMessage(RGYLogLevel log_level, const tstring& str) {
        if (m_log == nullptr || log_level < m_log->getLogLevel(RGY_LOGT_OUT)) {
            return;
        }
        auto lines = split(str, _T("\n"));
        for (const auto& line : lines) {
            if (line[0] != _T('\0')) {
                m_log->write(log_level, RGY_LOGT_OUT, (m_strWriterName + _T(": ") + line + _T("\n")).c_str());
            }
        }
    }
    void AddMessage(RGYLogLevel log_level, const TCHAR *format, ...) {
        if (m_log == nullptr || log_level < m_log->getLogLevel(RGY_LOGT_OUT)) {
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
    tstring m_strWriterName;
    shared_ptr<RGYLog> m_log;  //ログ出力
    RGY_CODEC m_codec;
    std::unique_ptr<AVBSFContext, RGYAVDeleter<AVBSFContext>> m_bsfc;
    std::unique_ptr<AVPacket, RGYAVDeleter<AVPacket>> m_pkt;
    std::vector<uint8_t> m_bsfBuffer;
    decltype(parse_nal_unit_h264_c) *m_parse_nal_h264; // H.264用のnal unit分解関数へのポインタ
    decltype(parse_nal_unit_hevc_c) *m_parse_nal_hevc; // HEVC用のnal unit分解関数へのポインタ
};

enum class RGYOutputInsertMetadataPosition {
    Prefix,
    FrontOfLastFrame,
    Appendix
};


struct RGYOutputInsertMetadata {
    std::vector<uint8_t> mdata;
    bool onSequenceHeader;
    RGYOutputInsertMetadataPosition pos;
    bool written;

    static RGYOutputInsertMetadataPosition dhdr10plus_pos(const RGY_CODEC codec) {
        return codec == RGY_CODEC_HEVC ? RGYOutputInsertMetadataPosition::Prefix : RGYOutputInsertMetadataPosition::FrontOfLastFrame;
    };
    static RGYOutputInsertMetadataPosition dovirpu_pos(const RGY_CODEC codec) {
        return codec == RGY_CODEC_HEVC ? RGYOutputInsertMetadataPosition::Appendix : RGYOutputInsertMetadataPosition::FrontOfLastFrame;
    };
    RGYOutputInsertMetadata(std::vector<uint8_t>& data, bool onSeqHeader, RGYOutputInsertMetadataPosition pos_) : mdata(data), onSequenceHeader(onSeqHeader), pos(pos_), written(false) {};
};

#pragma pack(push, 1)
struct RGYOutputRawPEExtHeader {
    size_t allocSize; // ヘッダを含んだメモリの確保サイズ
    int64_t pts;
    int64_t dts;
    int64_t duration;
    int64_t inputFrameIdx;
    int64_t encodeFrameIdx;
    RGY_PICSTRUCT picstruct;
    RGY_FRAMETYPE frameType;
    uint32_t flags;
    size_t size; // この後のデータのサイズ
};
#pragma pack(pop)

static_assert(std::is_trivially_copyable<RGYOutputRawPEExtHeader>::value);

static const size_t RGY_PE_EXT_HEADER_DATA_NORMAL_BUF_SIZE = 32 * 1024;
 
class RGYOutput {
public:
    RGYOutput();
    virtual ~RGYOutput();

    RGY_ERR Init(const TCHAR *strFileName, const VideoInfo *videoOutputInfo, const void *prm, shared_ptr<RGYLog> log, shared_ptr<EncodeStatus> encSatusInfo) {
        Close();
        m_printMes = log;
        m_encSatusInfo = encSatusInfo;
        m_outFilename = strFileName;
        if (videoOutputInfo) {
            m_VideoOutputInfo = *videoOutputInfo;
        }
        return Init(strFileName, videoOutputInfo, prm);
    }

    RGY_ERR InitVideoBsf(const VideoInfo *videoOutputInfo);

    RGY_ERR InsertMetadata(RGYBitstream *bitstream, std::vector<std::unique_ptr<RGYOutputInsertMetadata>>& metadataList);

    RGY_ERR OverwriteHEVCAlphaChannelInfoSEI(RGYBitstream *bitstream);

    template<typename T>
    std::pair<RGY_ERR, std::vector<uint8_t>> getMetadata(const RGYFrameDataType metadataType, const RGYTimestampMapVal& bs_framedata, const RGYFrameDataMetadataConvertParam *convPrm);

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
    static const char* OUT_DEBUG_FILE_HEADER;

    virtual RGY_ERR Init(const TCHAR *strFileName, const VideoInfo *pOutputInfo, const void *prm) = 0;

    RGY_ERR writeRawDebug(RGYBitstream *pBitstream);
    RGY_ERR readRawDebug(RGYBitstream *pBitstream);

    tstring     m_outFilename;
    std::shared_ptr<EncodeStatus> m_encSatusInfo;
    std::unique_ptr<FILE, fp_deleter> m_fDest;
    std::unique_ptr<FILE, fp_deleter> m_fpDebug;
    std::unique_ptr<FILE, fp_deleter> m_fpOutReplay;
    bool        m_outputIsStdout;
    bool        m_inited;
    bool        m_noOutput;
    OutputType  m_OutType;
    bool        m_sourceHWMem;
    bool        m_y4mHeaderWritten;
    bool        m_enableHEVCAlphaChannelInfoSEIOverwrite;
    int         m_HEVCAlphaChannelMode;
    tstring     m_strWriterName;
    tstring     m_strOutputInfo;
    VideoInfo   m_VideoOutputInfo;
    std::shared_ptr<RGYLog> m_printMes;  //ログ出力
    std::unique_ptr<char, malloc_deleter>            m_outputBuffer;
    std::unique_ptr<uint8_t, aligned_malloc_deleter> m_readBuffer;
    std::unique_ptr<uint8_t, aligned_malloc_deleter> m_UVBuffer;
    std::unique_ptr<RGYOutputBSF> m_bsf;
    decltype(parse_nal_unit_hevc_c) *m_parse_nal_hevc; // HEVC用のnal unit分解関数へのポインタ
};

struct RGYOutputRawPEExtHeader;

struct RGYOutputRawPrm {
    bool benchmark;
    bool debugDirectAV1Out;
    bool debugRawOut;
    bool extPERaw;
    bool HEVCAlphaChannel;
    int  HEVCAlphaChannelMode;
    tstring outReplayFile;
    RGY_CODEC outReplayCodec;
    int bufSizeMB;
    RGY_CODEC codecId;
    const RGYHDRMetadata *hdrMetadataIn;
    RGYHDR10Plus *hdr10plus;
    bool hdr10plusMetadataCopy;   //hdr10plusのmetadataをコピー
    RGYDOVIProfile doviProfile;
    DOVIRpu *doviRpu;
    bool doviRpuMetadataCopy;     //doviのmetadataのコピー
    RGYDOVIRpuConvertParam doviRpuConvertParam;
    RGYTimestamp *vidTimestamp;
    RGYQueueMPMP<RGYOutputRawPEExtHeader*> *qFirstProcessData;
    RGYQueueMPMP<RGYOutputRawPEExtHeader*> *qFirstProcessDataFree;
    RGYQueueMPMP<RGYOutputRawPEExtHeader*> *qFirstProcessDataFreeLarge;
};

class RGYOutputRaw : public RGYOutput {
public:

    RGYOutputRaw();
    virtual ~RGYOutputRaw();

    virtual RGY_ERR WriteNextFrame(RGYBitstream *pBitstream) override;
    virtual RGY_ERR WriteNextFrame(RGYFrame *pSurface) override;
protected:
    virtual RGY_ERR Init(const TCHAR *strFileName, const VideoInfo *pOutputInfo, const void *prm) override;
    virtual RGY_ERR WriteNextOneFrame(RGYBitstream *pBitstream);

    vector<uint8_t> m_outputBuf2;
    vector<uint8_t> m_hdrBitstream;
    RGYHDR10Plus *m_hdr10plus;
    bool m_hdr10plusMetadataCopy;
    RGYDOVIProfile m_doviProfileDst;
    DOVIRpu *m_doviRpu;
    bool m_doviRpuMetadataCopy;
    RGYDOVIRpuConvertParam m_doviRpuConvertParam;
    RGYTimestamp *m_timestamp;
    int64_t m_prevInputFrameId;
    int64_t m_prevEncodeFrameId;
    bool m_debugDirectAV1Out;
    bool m_extPERaw;
    RGYQueueMPMP<RGYOutputRawPEExtHeader*> *m_qFirstProcessData;
    RGYQueueMPMP<RGYOutputRawPEExtHeader*> *m_qFirstProcessDataFree;
    RGYQueueMPMP<RGYOutputRawPEExtHeader*> *m_qFirstProcessDataFreeLarge;
};

std::unique_ptr<RGYHDRMetadata> createHEVCHDRSei(const std::string &maxCll, const std::string &masterDisplay, CspTransfer atcSei, const RGYInput *reader);

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
    const RGYHDRMetadata *hdrMetadataIn,
    RGYHDR10Plus *hdr10plus,
    DOVIRpu *doviRpu,
    RGYTimestamp *vidTimestamp,
    const bool videoDtsUnavailable,
    const bool benchmark,
    const bool HEVCAlphaChannel,
    const int HEVCAlphaChannelMode,
    RGYPoolAVPacket *poolPkt,
    RGYPoolAVFrame *poolFrame,
    shared_ptr<EncodeStatus> pStatus,
    shared_ptr<CPerfMonitor> pPerfMonitor,
    shared_ptr<RGYLog> log
);

#if ENCODER_QSV || ENCODER_NVENC

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

#endif //#if ENCODER_QSV || ENCODER_NVENC

#endif //__RGY_OUTPUT_H__
