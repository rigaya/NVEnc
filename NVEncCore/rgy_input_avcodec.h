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
#ifndef __RGY_INPUT_AVCODEC_H__
#define __RGY_INPUT_AVCODEC_H__

#include "rgy_input.h"
#include "rgy_version.h"

#if ENABLE_AVSW_READER
#include "rgy_avutil.h"
#include "rgy_queue.h"
#include "rgy_perf_monitor.h"
#include "rgy_bitstream.h"
#include "convert_csp.h"
#include <deque>
#include <atomic>
#include <thread>
#include <cassert>

#if (defined(_WIN32) || defined(_WIN64))
#define USE_CUSTOM_INPUT 1
#include "rgy_caption.h"
#else
#define USE_CUSTOM_INPUT 0
#endif

using std::vector;
using std::pair;
using std::deque;

static const uint32_t AVCODEC_READER_INPUT_BUF_SIZE = 16 * 1024 * 1024;
static const uint32_t AV_FRAME_MAX_REORDER = 16;
static const int FRAMEPOS_POC_INVALID = -1;

static const char* HDR10PLUS_METADATA_KEY = "rgy_hdr10plus_metadata";

enum RGYPtsStatus : uint32_t {
    RGY_PTS_UNKNOWN           = 0x00,
    RGY_PTS_NORMAL            = 0x01,
    RGY_PTS_SOMETIMES_INVALID = 0x02, //時折、無効なptsを得る
    RGY_PTS_HALF_INVALID      = 0x04, //PAFFなため、半分のフレームのptsやdtsが無効
    RGY_PTS_ALL_INVALID       = 0x08, //すべてのフレームのptsやdtsが無効
    RGY_PTS_NONKEY_INVALID    = 0x10, //キーフレーム以外のフレームのptsやdtsが無効
    RGY_PTS_DUPLICATE         = 0x20, //重複するpts/dtsが存在する
    RGY_DTS_SOMETIMES_INVALID = 0x40, //時折、無効なdtsを得る
};

static RGYPtsStatus operator|(RGYPtsStatus a, RGYPtsStatus b) {
    return (RGYPtsStatus)((uint32_t)a | (uint32_t)b);
}

static RGYPtsStatus operator|=(RGYPtsStatus& a, RGYPtsStatus b) {
    a = a | b;
    return a;
}

static RGYPtsStatus operator&(RGYPtsStatus a, RGYPtsStatus b) {
    return (RGYPtsStatus)((uint32_t)a & (uint32_t)b);
}

static RGYPtsStatus operator&=(RGYPtsStatus& a, RGYPtsStatus b) {
    a = (RGYPtsStatus)((uint32_t)a & (uint32_t)b);
    return a;
}

//フレームの位置情報と長さを格納する
typedef struct FramePos {
    int64_t pts;  //pts
    int64_t dts;  //dts
    int duration;  //該当フレーム/フィールドの表示時間
    int duration2; //ペアフィールドの表示時間
    int poc; //出力時のフレーム番号
    uint8_t flags;    //flags (キーフレームならAV_PKT_FLAG_KEY)
    uint8_t pic_struct; //RGY_PICSTRUCT_xxx
    uint8_t repeat_pict; //通常は1, RFFなら2+
    uint8_t pict_type; //I,P,Bフレーム
} FramePos;

#if _DEBUG
#define DEBUG_FRAME_COPY(x) { if (m_fpDebugCopyFrameData) { (x); } }
#else
#define DEBUG_FRAME_COPY(x)
#endif

static FramePos framePosInit() {
    FramePos pos;
    pos.pts = 0;
    pos.dts = 0;
    pos.duration = 0;
    pos.duration2 = 0;
    pos.poc = FRAMEPOS_POC_INVALID;
    pos.flags = 0;
    pos.pic_struct = RGY_PICSTRUCT_FRAME;
    pos.repeat_pict = 0;
    pos.pict_type = 0;
    return pos;
}

static FramePos framePos(int64_t pts, int64_t dts,
    int duration, int duration2 = 0,
    int poc = FRAMEPOS_POC_INVALID,
    uint8_t flags = 0, uint8_t pic_struct = RGY_PICSTRUCT_FRAME, uint8_t repeat_pict = 0, uint8_t pict_type = 0) {
    FramePos pos;
    pos.pts = pts;
    pos.dts = dts;
    pos.duration = duration;
    pos.duration2 = duration2;
    pos.poc = poc;
    pos.flags = flags;
    pos.pic_struct = pic_struct;
    pos.repeat_pict = repeat_pict;
    pos.pict_type = pict_type;
    return pos;
}

class CompareFramePos {
public:
    uint32_t threshold;
    CompareFramePos() : threshold(0xFFFFFFFF) {
    }
    bool operator() (const FramePos& posA, const FramePos& posB) const {
        return ((uint32_t)std::abs(posA.pts - posB.pts) < threshold) ? posA.pts < posB.pts : posB.pts < posA.pts;
    }
};

class FramePosList {
public:
    FramePosList() :
        m_frameDuration(0.0),
        m_list(),
        m_nextFixNumIndex(0),
        m_inputFin(false),
        m_duration(0),
        m_durationNum(0),
        m_streamPtsStatus(RGY_PTS_UNKNOWN),
        m_lastPoc(0),
        m_firstKeyframePts(AV_NOPTS_VALUE),
        m_PAFFRewind(0),
        m_ptsWrapArroundThreshold(0xFFFFFFFF),
        m_fpDebugCopyFrameData() {
        m_list.init();
        static_assert(sizeof(m_list.get()[0]) == sizeof(m_list.get()->data), "FramePos must not have padding.");
    };
    virtual ~FramePosList() {
        clear();
    }
#pragma warning(push)
#pragma warning(disable:4100)
    int setLogCopyFrameData(const TCHAR *pLogFileName) {
        if (pLogFileName == nullptr) return 0;
#if _DEBUG
        FILE *fp = NULL;
        if (_tfopen_s(&fp, pLogFileName, _T("w"))) {
            return 1;
        }
        m_fpDebugCopyFrameData.reset(fp);
        return 0;
#else
        return 1;
#endif
    }
#pragma warning(pop)
    //filenameに情報をcsv形式で出力する
    int printList(const TCHAR *filename) {
        const int nList = (int)m_list.size();
        if (nList == 0) {
            return 0;
        }
        if (filename == nullptr) {
            return 1;
        }
        FILE *fp = NULL;
        if (0 != _tfopen_s(&fp, filename, _T("wb"))) {
            return 1;
        }
        fprintf(fp, "pts,dts,duration,duration2,poc,flags,pic_struct,repeat_pict,pict_type\r\n");
        for (int i = 0; i < nList; i++) {
            fprintf(fp, "%lld,%lld,%d,%d,%d,%d,%d,%d,%d\r\n",
                (lls)m_list[i].data.pts, (lls)m_list[i].data.dts,
                m_list[i].data.duration, m_list[i].data.duration2,
                m_list[i].data.poc,
                (int)m_list[i].data.flags, (int)m_list[i].data.pic_struct, (int)m_list[i].data.repeat_pict, (int)m_list[i].data.pict_type);
        }
        fclose(fp);
        return 0;
    }
    //indexの位置への参照を返す
    // !! push側のスレッドからのみ有効 !!
    FramePos& list(uint32_t index) {
        return m_list[index].data;
    }
    //初期化
    void clear() {
        m_list.close();
        m_frameDuration = 0.0;
        m_nextFixNumIndex = 0;
        m_inputFin = false;
        m_duration = 0;
        m_durationNum = 0;
        m_streamPtsStatus = RGY_PTS_UNKNOWN;
        m_lastPoc = 0;
        m_firstKeyframePts = AV_NOPTS_VALUE;
        m_PAFFRewind = 0;
        m_ptsWrapArroundThreshold = 0xFFFFFFFF;
        m_fpDebugCopyFrameData.reset();
        m_list.init();
    }
    //ここまで計算したdurationを返す
    int64_t duration() const {
        return m_duration;
    }
    //登録された(ptsの確定していないものを含む)フレーム数を返す
    int frameNum() const {
        return (int)m_list.size();
    }
    //ptsが確定したフレーム数を返す
    int fixedNum() const {
        return m_nextFixNumIndex;
    }
    void clearPtsStatus() {
        if (m_streamPtsStatus & RGY_PTS_DUPLICATE) {
            const int nListSize = (int)m_list.size();
            for (int i = 0; i < nListSize; i++) {
                if (m_list[i].data.duration == 0
                    && m_list[i].data.pts != AV_NOPTS_VALUE
                    && m_list[i].data.dts != AV_NOPTS_VALUE
                    && m_list[i+1].data.pts - m_list[i].data.pts <= (std::min)(m_list[i+1].data.duration / 10, 1)
                    && m_list[i+1].data.dts - m_list[i].data.dts <= (std::min)(m_list[i+1].data.duration / 10, 1)) {
                    m_list[i].data.duration = m_list[i+1].data.duration;
                }
            }
        }
        m_lastPoc = 0;
        m_nextFixNumIndex = 0;
        m_streamPtsStatus = RGY_PTS_UNKNOWN;
        m_PAFFRewind = 0;
        m_ptsWrapArroundThreshold = 0xFFFFFFFF;
    }
    RGYPtsStatus getStreamPtsStatus() const {
        return m_streamPtsStatus;
    }
    FramePos findpts(int64_t pts, uint32_t *lastIndex) {
        FramePos pos_last = framePosInit();
        for (uint32_t index = *lastIndex + 1; ; index++) {
            FramePos pos = framePosInit();
            if (!m_list.copy(&pos, index)) {
                break;
            }
            if (pts == pos.pts) {
                *lastIndex = index;
                return pos;
            }
            pos_last = pos;
        }
        //最初から探索
        for (uint32_t index = 0; ; index++) {
            FramePos pos = framePosInit();
            if (!m_list.copy(&pos, index)) {
                break;
            }
            if (pts == pos.pts) {
                *lastIndex = index;
                return pos;
            }
            //pts < demux.videoFramePts[i]であるなら、その前のフレームを返す
            if (pts < pos.pts) {
                *lastIndex = index-1;
                return pos_last;
            }
            pos_last = pos;
        }
        //エラー
        FramePos poserr = framePosInit();
        return poserr;
    }
    //FramePosを追加し、内部状態を変更する
    void add(const FramePos& pos) {
        m_list.push(pos);
        const int nListSize = (int)m_list.size();
        //自分のフレームのインデックス
        const int nIndex = nListSize-1;
        //ptsの補正
        adjustFrameInfo(nIndex);
        //最初のキーフレームの位置を記憶しておく
        if (m_firstKeyframePts == AV_NOPTS_VALUE && (pos.flags & AV_PKT_FLAG_KEY) && nIndex == 0) {
            m_firstKeyframePts = m_list[nIndex].data.pts;
        }
        //m_streamPtsStatusがRGY_PTS_UNKNOWNの場合には、ソートなどは行わない
        if (m_inputFin || (m_streamPtsStatus && nListSize - m_nextFixNumIndex > (int)AV_FRAME_MAX_REORDER)) {
            //ptsでソート
            sortPts(m_nextFixNumIndex, nListSize - m_nextFixNumIndex);
            setPocAndFix(nListSize);
        }
        calcDuration();
    };
    //pocの一致するフレームの情報のコピーを返す
    FramePos copy(int poc, uint32_t *lastIndex) {
        assert(lastIndex != nullptr);
        for (uint32_t index = *lastIndex + 1; ; index++) {
            FramePos pos = framePosInit();
            if (!m_list.copy(&pos, index)) {
                break;
            }
            if (pos.poc == poc) {
                *lastIndex = index;
                DEBUG_FRAME_COPY(_ftprintf(m_fpDebugCopyFrameData.get(), _T("request poc: %8d, hit index: %8d, pts: %lld\n"), poc, index, (lls)pos.pts));
                return pos;
            }
            if (m_inputFin && pos.poc == -1) {
                //もう読み込みは終了しているが、さらなるフレーム情報の要求が来ている
                //予想より出力が過剰になっているということで、tsなどで最初がopengopの場合に起こりうる
                //なにかおかしなことが起こっており、異常なのだが、最後の最後でエラーとしてしまうのもあほらしい
                //とりあえず、ptsを推定して返してしまう
                pos.poc = poc;
                FramePos pos_tmp = framePosInit();
                m_list.copy(&pos_tmp, index-1);
                int nLastPoc = pos_tmp.poc;
                int64_t nLastPts = pos_tmp.pts;
                m_list.copy(&pos_tmp, 0);
                int64_t pts0 = pos_tmp.pts;
                m_list.copy(&pos_tmp, 1);
                if (pos_tmp.poc == -1) {
                    m_list.copy(&pos_tmp, 2);
                }
                int64_t pts1 = pos_tmp.pts;
                int nFrameDuration = (int)(pts1 - pts0);
                pos.pts = nLastPts + (poc - nLastPoc) * nFrameDuration;
                DEBUG_FRAME_COPY(_ftprintf(m_fpDebugCopyFrameData.get(), _T("request poc: %8d, hit index: %8d [invalid], estimated pts: %lld\n"), poc, index, (lls)pos.pts));
                return pos;
            }
        }
        //エラー
        FramePos pos = framePosInit();
        DEBUG_FRAME_COPY(_ftprintf(m_fpDebugCopyFrameData.get(), _T("request: %8d, invalid, list size: %d\n"), poc, (int)m_list.size()));
        return pos;
    }
    //入力が終了した際に使用し、内部状態を変更する
    void fin(const FramePos& pos, int64_t total_duration) {
        m_inputFin = true;
        if (m_streamPtsStatus == RGY_PTS_UNKNOWN) {
            checkPtsStatus();
        }
        const int nFrame = (int)m_list.size();
        sortPts(m_nextFixNumIndex, nFrame - m_nextFixNumIndex);
        m_nextFixNumIndex += m_PAFFRewind;
        for (int i = m_nextFixNumIndex; i < nFrame; i++) {
            adjustDurationAfterSort(m_nextFixNumIndex);
            setPoc(i);
        }
        m_nextFixNumIndex = nFrame;
        add(pos);
        m_nextFixNumIndex += m_PAFFRewind;
        m_PAFFRewind = 0;
        m_duration = total_duration;
        m_durationNum = m_nextFixNumIndex;
    }
    bool isEof() const {
        return m_inputFin;
    }
    //現在の情報から、ptsの状態を確認する
    //さらにptsの補正、ptsのソート、pocの確定を行う
    void checkPtsStatus(double durationHintifPtsAllInvalid = 0.0) {
        const int nInputPacketCount = (int)m_list.size();
        int nInputFrames = 0;
        int nInputFields = 0;
        int nInputKeys = 0;
        int nDuplicateFrameInfo = 0;
        int nInvalidPtsCount = 0;
        int nInvalidDtsCount = 0;
        int nInvalidPtsCountField = 0;
        int nInvalidPtsCountKeyFrame = 0;
        int nInvalidPtsCountNonKeyFrame = 0;
        int nInvalidDuration = 0;
        bool bFractionExists = std::abs(durationHintifPtsAllInvalid - (int)(durationHintifPtsAllInvalid + 0.5)) > 1e-6;
        vector<std::pair<int, int>> durationHistgram;
        for (int i = 0; i < nInputPacketCount; i++) {
            nInputFrames += (m_list[i].data.pic_struct & RGY_PICSTRUCT_FRAME) != 0;
            nInputFields += (m_list[i].data.pic_struct & RGY_PICSTRUCT_FIELD) != 0;
            nInputKeys   += (m_list[i].data.flags & AV_PKT_FLAG_KEY) != 0;
            nInvalidDuration += m_list[i].data.duration <= 0;
            if (m_list[i].data.pts == AV_NOPTS_VALUE) {
                nInvalidPtsCount++;
                nInvalidPtsCountField += (m_list[i].data.pic_struct & RGY_PICSTRUCT_FIELD) != 0;
                nInvalidPtsCountKeyFrame += (m_list[i].data.flags & AV_PKT_FLAG_KEY) != 0;
                nInvalidPtsCountNonKeyFrame += (m_list[i].data.flags & AV_PKT_FLAG_KEY) == 0;
            }
            if (m_list[i].data.dts == AV_NOPTS_VALUE) {
                nInvalidDtsCount++;
            }
            if (i > 0) {
                //VP8/VP9では重複するpts/dts/durationを持つフレームが存在することがあるが、これを無視する
                if (bFractionExists
                    && m_list[i].data.duration > 0
                    && m_list[i].data.pts != AV_NOPTS_VALUE
                    && m_list[i].data.dts != AV_NOPTS_VALUE
                    && m_list[i].data.pts - m_list[i-1].data.pts <= (std::min)(m_list[i].data.duration / 10, 1)
                    && m_list[i].data.dts - m_list[i-1].data.dts <= (std::min)(m_list[i].data.duration / 10, 1)
                    && m_list[i].data.duration == m_list[i-1].data.duration) {
                    nDuplicateFrameInfo++;
                }
            }
            int nDuration = m_list[i].data.duration;
            auto target = std::find_if(durationHistgram.begin(), durationHistgram.end(), [nDuration](const std::pair<int, int>& pair) { return pair.first == nDuration; });
            if (target != durationHistgram.end()) {
                target->second++;
            } else {
                durationHistgram.push_back(std::make_pair(nDuration, 1));
            }
        }
        //多い順にソートする
        std::sort(durationHistgram.begin(), durationHistgram.end(), [](const std::pair<int, int>& pairA, const std::pair<int, int>& pairB) { return pairA.second > pairB.second; });
        m_streamPtsStatus = RGY_PTS_UNKNOWN;
        if (nDuplicateFrameInfo > 0) {
            //VP8/VP9では重複するpts/dts/durationを持つフレームが存在することがあるが、これを無視する
            m_streamPtsStatus |= RGY_PTS_DUPLICATE;
        }
        if (nInvalidPtsCount == 0) {
            m_streamPtsStatus |= RGY_PTS_NORMAL;
        } else {
            m_frameDuration = durationHintifPtsAllInvalid;
            if (nInvalidPtsCount >= nInputPacketCount - 1) {
                if (m_list[0].data.duration || durationHintifPtsAllInvalid > 0.0) {
                    //durationが得られていれば、durationに基づいて、cfrでptsを発行する
                    //主にH.264/HEVCのESなど
                    m_streamPtsStatus |= RGY_PTS_ALL_INVALID;
                } else {
                    //durationがなければ、dtsを見てptsを発行する
                    //主にVC-1ストリームなど
                    m_streamPtsStatus |= RGY_PTS_SOMETIMES_INVALID;
                }
            } else if (nInputFields > 0 && nInvalidPtsCountField <= nInputFields / 2) {
                //主にH.264のPAFFストリームなど
                m_streamPtsStatus |= RGY_PTS_HALF_INVALID;
            } else if (nInvalidPtsCountKeyFrame == 0 && nInvalidPtsCountNonKeyFrame > (nInputPacketCount - nInputKeys) * 3 / 4) {
                m_streamPtsStatus |= RGY_PTS_NONKEY_INVALID;
                if (nInvalidPtsCount == nInvalidDtsCount) {
                    //ワンセグなど、ptsもdtsもキーフレーム以外は得られない場合
                    m_streamPtsStatus |= RGY_DTS_SOMETIMES_INVALID;
                }
                if (nInvalidDuration == 0) {
                    //ptsがだいぶいかれてるので、安定してdurationが得られていれば、durationベースで作っていったほうが早い
                    m_streamPtsStatus |= RGY_PTS_SOMETIMES_INVALID;
                }
            }
            if (!(m_streamPtsStatus & (RGY_PTS_ALL_INVALID | RGY_PTS_HALF_INVALID | RGY_PTS_NONKEY_INVALID | RGY_PTS_SOMETIMES_INVALID))
                && nInvalidPtsCount > nInputPacketCount / 16) {
                m_streamPtsStatus |= RGY_PTS_SOMETIMES_INVALID;
            }
        }
        if ((m_streamPtsStatus & RGY_PTS_ALL_INVALID)) {
            auto& mostPopularDuration = durationHistgram[durationHistgram.size() > 1 && durationHistgram[0].first == 0];
            if ((m_frameDuration > 0.0 && m_list[0].data.duration == 0) || mostPopularDuration.first == 0) {
                //主にH.264/HEVCのESなど向けの対策
                m_list[0].data.duration = (int)(m_frameDuration * ((m_list[0].data.pic_struct & RGY_PICSTRUCT_FIELD) ? 0.5 : 1.0) + 0.5);
            } else {
                //durationのヒストグラムを作成
                m_frameDuration = durationHistgram[durationHistgram.size() > 1 && durationHistgram[0].first == 0].first;
            }
        }
        for (int i = m_nextFixNumIndex; i < nInputPacketCount; i++) {
            adjustFrameInfo(i);
        }
        sortPts(m_nextFixNumIndex, nInputPacketCount - m_nextFixNumIndex);
        setPocAndFix(nInputPacketCount);
        if (m_nextFixNumIndex > 1) {
            int64_t pts0 = m_list[0].data.pts;
            int64_t pts1 = m_list[1 + (m_list[0].data.poc == -1)].data.pts;
            m_ptsWrapArroundThreshold = (uint32_t)clamp((int64_t)(std::max)((uint32_t)(pts1 - pts0), (uint32_t)(m_frameDuration + 0.5)) * 360, 360, (int64_t)0xFFFFFFFF);
        }
    }
    RGY_PICSTRUCT getVideoPicStruct() {
        const int nListSize = (int)m_list.size();
        for (int i = 0; i < nListSize; i++) {
            auto pic_struct = m_list[i].data.pic_struct;
            if (pic_struct & RGY_PICSTRUCT_INTERLACED) {
                return (RGY_PICSTRUCT)(pic_struct & RGY_PICSTRUCT_INTERLACED);
            }
        }
        return RGY_PICSTRUCT_FRAME;
    }
protected:
    //ptsでソート
    void sortPts(uint32_t index, uint32_t len) {
#if (!defined(_MSC_VER) && __cplusplus <= 201103) || defined(__NVCC__)
        FramePos *pStart = (FramePos *)m_list.get(index);
        FramePos *pEnd = (FramePos *)m_list.get(index + len);
        std::sort(pStart, pEnd, CompareFramePos());
#else
        const auto nPtsWrapArroundThreshold = m_ptsWrapArroundThreshold;
        std::sort(m_list.get(index), m_list.get(index + len), [nPtsWrapArroundThreshold](const auto& posA, const auto& posB) {
            return ((uint32_t)(std::abs(posA.data.pts - posB.data.pts)) < nPtsWrapArroundThreshold) ? posA.data.pts < posB.data.pts : posB.data.pts < posA.data.pts; });
#endif
    }
    //ptsの補正
    void adjustFrameInfo(uint32_t nIndex) {
        if (m_streamPtsStatus & RGY_PTS_SOMETIMES_INVALID) {
            if (m_streamPtsStatus & RGY_DTS_SOMETIMES_INVALID) {
                //ptsもdtsはあてにならないので、durationから再構築する (ワンセグなど)
                if (nIndex == 0) {
                    if (m_list[nIndex].data.pts == AV_NOPTS_VALUE) {
                        m_list[nIndex].data.pts = 0;
                    }
                } else if (m_list[nIndex].data.pts == AV_NOPTS_VALUE) {
                    m_list[nIndex].data.pts = m_list[nIndex-1].data.pts + m_list[nIndex-1].data.duration;
                }
            } else {
                //ptsはあてにならないので、dtsから再構築する (VC-1など)
                int64_t firstFramePtsDtsDiff = m_list[0].data.pts - m_list[0].data.dts;
                if (nIndex > 0 && m_list[nIndex].data.dts == AV_NOPTS_VALUE) {
                    m_list[nIndex].data.dts = m_list[nIndex-1].data.dts + m_list[0].data.duration;
                }
                m_list[nIndex].data.pts = m_list[nIndex].data.dts + firstFramePtsDtsDiff;
            }
        } else if (m_list[nIndex].data.pts == AV_NOPTS_VALUE) {
            if (nIndex == 0) {
                m_list[nIndex].data.pts = 0;
                m_list[nIndex].data.dts = 0;
            } else if (m_streamPtsStatus & (RGY_PTS_ALL_INVALID | RGY_PTS_NONKEY_INVALID)) {
                //AVPacketのもたらすptsが無効であれば、CFRを仮定して適当にptsとdurationを突っ込んでいく
                double frameDuration = m_frameDuration * ((m_list[0].data.pic_struct & RGY_PICSTRUCT_FIELD) ? 2.0 : 1.0);
                m_list[nIndex].data.pts = (int64_t)(nIndex * frameDuration * ((m_list[nIndex].data.pic_struct & RGY_PICSTRUCT_FIELD) ? 0.5 : 1.0) + 0.5);
                m_list[nIndex].data.dts = m_list[nIndex].data.pts;
            } else if (m_streamPtsStatus & RGY_PTS_NONKEY_INVALID) {
                //キーフレーム以外のptsとdtsが無効な場合は、適当に推定する
                double frameDuration = m_frameDuration * ((m_list[0].data.pic_struct & RGY_PICSTRUCT_FIELD) ? 2.0 : 1.0);
                m_list[nIndex].data.pts = m_list[nIndex-1].data.pts + (int)(frameDuration * ((m_list[nIndex].data.pic_struct & RGY_PICSTRUCT_FIELD) ? 0.5 : 1.0) + 0.5);
                m_list[nIndex].data.dts = m_list[nIndex-1].data.dts + (int)(frameDuration * ((m_list[nIndex].data.pic_struct & RGY_PICSTRUCT_FIELD) ? 0.5 : 1.0) + 0.5);
            } else if (m_streamPtsStatus & RGY_PTS_HALF_INVALID) {
                //ptsがないのは音声抽出で、正常に抽出されない問題が生じる
                //半分PTSがないPAFFのような動画については、前のフレームからの補完を行う
                if (m_list[nIndex].data.dts == AV_NOPTS_VALUE) {
                    m_list[nIndex].data.dts = m_list[nIndex-1].data.dts + m_list[nIndex-1].data.duration;
                }
                m_list[nIndex].data.pts = m_list[nIndex-1].data.pts + m_list[nIndex-1].data.duration;
            } else if (m_streamPtsStatus & RGY_PTS_NORMAL) {
                if (m_list[nIndex].data.pts == AV_NOPTS_VALUE) {
                    m_list[nIndex].data.pts = m_list[nIndex-1].data.pts + m_list[nIndex-1].data.duration;
                }
            }
        }
    }
    //ソートにより確定したptsに対して、pocを設定する
    void setPoc(int index) {
        if ((m_streamPtsStatus & RGY_PTS_DUPLICATE)
            && m_list[index].data.duration == 0
            && m_list[index+1].data.pts - m_list[index].data.pts <= (std::min)(m_list[index+1].data.duration / 10, 1)
            && m_list[index+1].data.dts - m_list[index].data.dts <= (std::min)(m_list[index+1].data.duration / 10, 1)) {
            //VP8/VP9では重複するpts/dts/durationを持つフレームが存在することがあるが、これを無視する
            m_list[index].data.poc = FRAMEPOS_POC_INVALID;
        } else if (m_list[index].data.pic_struct & RGY_PICSTRUCT_FIELD) {
            if (index > 0 && (m_list[index-1].data.poc != FRAMEPOS_POC_INVALID && (m_list[index-1].data.pic_struct & RGY_PICSTRUCT_FIELD))) {
                m_list[index].data.poc = FRAMEPOS_POC_INVALID;
                m_list[index-1].data.duration2 = m_list[index].data.duration;
            } else {
                m_list[index].data.poc = m_lastPoc++;
            }
        } else {
            m_list[index].data.poc = m_lastPoc++;
        }
    }
    //ソート後にindexのdurationを再計算する
    //ソートはindex+1まで確定している必要がある
    //ソート後のこの段階では、AV_NOPTS_VALUEはないものとする
    void adjustDurationAfterSort(int index) {
        int diff = (int)(m_list[index+1].data.pts - m_list[index].data.pts);
        if ((m_streamPtsStatus & RGY_PTS_DUPLICATE)
            && diff <= 1
            && m_list[index].data.duration > 0
            && m_list[index].data.pts != AV_NOPTS_VALUE
            && m_list[index].data.dts != AV_NOPTS_VALUE
            && m_list[index+1].data.duration == m_list[index].data.duration
            && m_list[index+1].data.pts - m_list[index].data.pts <= (std::min)(m_list[index].data.duration / 10, 1)
            && m_list[index+1].data.dts - m_list[index].data.dts <= (std::min)(m_list[index].data.duration / 10, 1)) {
            //VP8/VP9では重複するpts/dts/durationを持つフレームが存在することがあるが、これを無視する
            m_list[index].data.duration = 0;
        } else if (diff > 0) {
            m_list[index].data.duration = diff;
        }
    }
    //進捗表示用のdurationの計算を行う
    //これは16フレームに1回行う
    void calcDuration() {
        int nNonDurationCalculatedFrames = m_nextFixNumIndex - m_durationNum;
        if (nNonDurationCalculatedFrames >= 16) {
            const auto *pos_fixed = m_list.get(m_durationNum);
            int64_t duration = pos_fixed[nNonDurationCalculatedFrames-1].data.pts - pos_fixed[0].data.pts;
            if (duration < 0 || duration > m_ptsWrapArroundThreshold) {
                duration = 0;
                for (int i = 1; i < nNonDurationCalculatedFrames; i++) {
                    int64_t diff = (std::max<int64_t>)(0, pos_fixed[i].data.pts - pos_fixed[i-1].data.pts);
                    int64_t last_frame_dur = (std::max<int64_t>)(0, pos_fixed[i-1].data.duration);
                    duration += (diff > m_ptsWrapArroundThreshold) ? last_frame_dur : diff;
                }
            }
            m_duration += duration;
            m_durationNum += nNonDurationCalculatedFrames;
        }
    }
    //pocを確定させる
    void setPocAndFix(int nSortedSize) {
        //ソートによりptsが確定している範囲
        //本来はnSortedSize - (int)AV_FRAME_MAX_REORDERでよいが、durationを確定させるためにはさらにもう一枚必要になる
        int nSortFixedSize = nSortedSize - (int)AV_FRAME_MAX_REORDER - 1;
        m_nextFixNumIndex += m_PAFFRewind;
        for (; m_nextFixNumIndex < nSortFixedSize; m_nextFixNumIndex++) {
            if (m_list[m_nextFixNumIndex].data.pts < m_firstKeyframePts //ソートの先頭のptsが塚下キーフレームの先頭のptsよりも小さいことがある(opengop)
                && m_nextFixNumIndex <= 16) { //wrap arroundの場合は除く
                //これはフレームリストから取り除く
                m_list.pop();
                m_nextFixNumIndex--;
                nSortFixedSize--;
            } else {
                adjustDurationAfterSort(m_nextFixNumIndex);
                //ソートにより確定したptsに対して、pocとdurationを設定する
                setPoc(m_nextFixNumIndex);
            }
        }
        m_PAFFRewind = 0;
        //もし、現在のインデックスがフィールドデータの片割れなら、次のフィールドがくるまでdurationは確定しない
        //setPocでduration2が埋まるのを待つ必要がある
        if (m_nextFixNumIndex > 0
            && (m_list[m_nextFixNumIndex-1].data.pic_struct & RGY_PICSTRUCT_FIELD)
            && m_list[m_nextFixNumIndex-1].data.poc != FRAMEPOS_POC_INVALID) {
            m_nextFixNumIndex--;
            m_PAFFRewind = 1;
        }
    }
protected:
    double m_frameDuration; //CFRを仮定する際のフレーム長 (RGY_PTS_ALL_INVALID, RGY_PTS_NONKEY_INVALID, RGY_PTS_NONKEY_INVALID時有効)
    RGYQueueSPSP<FramePos, 1> m_list; //内部データサイズとFramePosのデータサイズを一致させるため、alignを1に設定
    int m_nextFixNumIndex; //次にptsを確定させるフレームのインデックス
    bool m_inputFin; //入力が終了したことを示すフラグ
    int64_t m_duration; //m_durationNumのフレーム数分のdurationの総和
    int m_durationNum; //durationを計算したフレーム数
    RGYPtsStatus m_streamPtsStatus; //入力から提供されるptsの状態 (RGY_PTS_xxx)
    uint32_t m_lastPoc; //ptsが確定したフレームのうち、直近のpoc
    int64_t m_firstKeyframePts; //最初のキーフレームのpts
    int m_PAFFRewind; //PAFFのdurationを確定させるため、戻した枚数
    uint32_t m_ptsWrapArroundThreshold; //wrap arroundを判定する閾値
    unique_ptr<FILE, fp_deleter> m_fpDebugCopyFrameData; //copyのデバッグ用
};


//動画フレームのデータ
typedef struct VideoFrameData {
    FramePos *frame;      //各フレームの情報への配列 (デコードが開始されるフレームから取得する)
    int fixed_num;        //frame配列でフレーム順序が確定したフレーム数
    int num;              //frame配列で現在格納されているデータ数
    int capacity;         //frame配列を確保した数
    int64_t duration;     //合計の動画の長さ
} VideoFrameData;

typedef struct AVDemuxFormat {
    AVFormatContext          *formatCtx;             //動画ファイルのformatContext
    double                    analyzeSec;            //動画ファイルを先頭から分析する時間
    bool                      isPipe;                //入力がパイプ
    uint32_t                  preReadBufferIdx;      //先読みバッファの読み込み履歴
    int                       audioTracks;           //存在する音声のトラック数
    int                       subtitleTracks;        //存在する字幕のトラック数
    int                       dataTracks;            //存在するデータのトラック数
    int                       attachmentTracks;      //存在するAttachmentのトラック数
    RGYAVSync                 AVSyncMode;            //音声・映像同期モード
    AVDictionary             *formatOptions;         //avformat_open_inputに渡すオプション

    FILE                     *fpInput;               //入力ファイルポインタ
    char                     *inputBuffer;           //入力バッファ
    int                       inputBufferSize;       //入力バッファサイズ
    uint64_t                  inputFilesize;         //入力ファイルサイズ
} AVDemuxFormat;

typedef struct AVDemuxVideo {
                                                     //動画は音声のみ抽出する場合でも同期のため参照することがあり、
                                                     //pCodecCtxのチェックだけでは読み込むかどうか判定できないので、
                                                     //実際に使用するかどうかはこのフラグをチェックする
    bool                      readVideo;
    const AVStream           *stream;                //動画のStream, 動画を読み込むかどうかの判定には使用しないこと (readVideoを使用)
    const AVCodec            *codecDecode;           //動画のデコーダ (使用しない場合はnullptr)
    AVCodecContext           *codecCtxDecode;        //動画のデコーダ (使用しない場合はnullptr)
    AVFrame                  *frame;                 //動画デコード用のフレーム
    int                       index;                 //動画のストリームID
    int64_t                   streamFirstKeyPts;     //動画ファイルの最初のpts
    uint32_t                  streamPtsInvalid;      //動画ファイルのptsが無効 (H.264/ES, 等)
    int                       RFFEstimate;           //動画がRFFの可能性がある
    bool                      gotFirstKeyframe;      //動画の最初のキーフレームを取得済み
    AVBSFContext             *bsfcCtx;               //必要なら使用するbitstreamfilter
    uint8_t                  *extradata;             //動画のヘッダ情報
    int                       extradataSize;         //動画のヘッダサイズ
    AVRational                nAvgFramerate;         //動画のフレームレート

    int                       nSampleGetCount;       //sampleをGetNextBitstreamで取得した数

    AVCodecParserContext     *pParserCtx;            //動画ストリームのParser
    AVCodecContext           *pCodecCtxParser;       //動画ストリームのParser用

    int                       HWDecodeDeviceId;      //HWデコードする場合に選択したデバイス

    bool                      bUseHEVCmp42AnnexB;
    bool                      hdr10plusMetadataCopy; //HDR10plusのメタ情報を取得する
    bool                      doviRpuCopy;           //dovi rpuのメタ情報を取得する

    AVMasteringDisplayMetadata *masteringDisplay;    //入力ファイルから抽出したHDRメタ情報
    AVContentLightMetadata   *contentLight;          //入力ファイルから抽出したHDRメタ情報

    RGYListRef<RGYFrameDataQP> *qpTableListRef;      //qp tableを格納するときのベース構造体
    decltype(parse_nal_unit_h264_c) *parse_nal_h264; // H.264用のnal unit分解関数へのポインタ
    decltype(parse_nal_unit_hevc_c) *parse_nal_hevc; // HEVC用のnal unit分解関数へのポインタ
} AVDemuxVideo;

typedef struct AVDemuxThread {
    int                          threadInput;        //入力スレッドを使用する
    std::atomic<bool>            bAbortInput;        //読み込みスレッドに停止を通知する
    std::thread                  thInput;            //読み込みスレッド
    PerfQueueInfo               *queueInfo;          //キューの情報を格納する構造体
} AVDemuxThread;

typedef struct AVDemuxer {
    AVDemuxFormat            format;
    AVDemuxVideo             video;
    FramePosList             frames;
    vector<AVDemuxStream>    stream;
    vector<const AVChapter*> chapter;
    AVDemuxThread            thread;
    RGYQueueSPSP<AVPacket>   qVideoPkt;
    deque<AVPacket>          qStreamPktL1;
    RGYQueueSPSP<AVPacket>   qStreamPktL2;
} AVDemuxer;

enum AVCAPTION_STATE {
    //エラー
    AVCAPTION_ERROR = -3,
    AVCAPTION_DISABLED = -2,
    AVCAPTION_NOT_TS = -1,
    //初期状態
    AVCAPTION_UNKNOWN = 0,
    //ts処理中
    AVCAPTION_IS_TS = 1,
};


#if ENABLE_CAPTION2ASS
class AVCaption2Ass {
public:
    AVCaption2Ass() : m_cap2ass(), m_pLog(), m_subList(), m_buffer(),
        m_index(-1), m_trackId(0),
        m_state(AVCAPTION_UNKNOWN), m_resolutionDetermined(false) {
    };
    ~AVCaption2Ass() { close(); };
    bool enabled() const {
        return m_cap2ass.enabled() && m_state >= AVCAPTION_UNKNOWN;
    }
    void close() {
        m_state = AVCAPTION_UNKNOWN;
        m_cap2ass.close();
        m_pLog.reset();
    }
    RGY_ERR init(std::shared_ptr<RGYLog> pLog, C2AFormat format) {
        m_pLog = pLog;
        return m_cap2ass.init(pLog, format);
    }
    C2AFormat format() const {
        return m_cap2ass.format();
    }
    AVCAPTION_STATE state() const {
        return m_state;
    }
    void disable() {
        m_state = AVCAPTION_DISABLED;
    }
    void reset() {
        //m_resolutionDeterminedはリセットしない
        m_state = AVCAPTION_UNKNOWN;
        m_cap2ass.reset();
        m_buffer.clear();
    }
    void setIndex(int streamIndex, int trackId) {
        m_index = streamIndex;
        m_trackId = trackId;
    }
    void setOutputResolution(int w, int h, int sar_x, int sar_y) {
        m_cap2ass.setOutputResolution(w, h, sar_x, sar_y);
        m_resolutionDetermined = true;
    }
    void printParam(RGYLogLevel log_level) {
        m_cap2ass.printParam(log_level);
    }
    void setVidFirstKeyPts(int64_t pts) {
        m_cap2ass.setVidFirstKeyPts(pts);
    }
    AVDemuxStream stream() const {
        std::string header = m_cap2ass.assHeader();
        AVDemuxStream stream;
        memset(&stream, 0, sizeof(AVDemuxStream));
        stream.index = m_index;
        stream.trackId = m_trackId;
        stream.subtitleHeader = av_strdup(header.c_str());
        stream.subtitleHeaderSize = (int)header.length();
        stream.timebase = av_make_q(1, 90000);
        stream.caption2ass = m_cap2ass.format();
        return stream;
    }
    RGY_ERR proc(uint8_t *buf, size_t buf_size, decltype(AVDemuxer::qStreamPktL1)& qStreamPkt) {
        auto ret = RGY_ERR_NONE;
        if (buf_size == 0) {
            return ret;
        }
        if (m_state == AVCAPTION_UNKNOWN) {
            m_state = m_cap2ass.isTS(buf, buf_size) ? AVCAPTION_IS_TS : AVCAPTION_NOT_TS;
        }
        if (m_state == AVCAPTION_IS_TS) {
            if (!m_resolutionDetermined) {
                //出力解像度が決まるまでデータを蓄積
                vector_cat(m_buffer, buf, buf_size);
            } else {
                if (m_buffer.size() > 0) {
                    ret = m_cap2ass.proc(m_buffer.data(), m_buffer.size(), m_subList);
                    m_buffer.clear();
                    m_buffer.shrink_to_fit();
                }
                if (ret == RGY_ERR_NONE) {
                    ret = m_cap2ass.proc(buf, buf_size, m_subList);
                }
                if (ret != RGY_ERR_NONE) {
                    m_state = AVCAPTION_ERROR;
                } else if (m_index >= 0) { //インデックスが決まるまでは、クラス内にためておく
                    for (auto it = m_subList.begin(); it != m_subList.end(); it++) {
                        it->stream_index = m_index;
                        qStreamPkt.push_back(*it);
                    }
                    m_subList.clear();
                }
            }
        }
        return ret;
    }
protected:
    Caption2Ass m_cap2ass; //Caption2Ass処理
    std::shared_ptr<RGYLog> m_pLog;
    std::vector<AVPacket> m_subList;

    //解像度が決まるまでデータを取っておくバッファ
    std::vector<uint8_t> m_buffer;

    int m_index;
    int m_trackId;

    //現在の処理状態
    AVCAPTION_STATE m_state;

    //出力解像度が決まったら処理を開始するので、
    //出力解像度が決まったかどうかを示すフラグ
    bool m_resolutionDetermined;
};
#else
#pragma warning(push)
#pragma warning(disable: 4100)
class AVCaption2Ass {
public:
    AVCaption2Ass() {};
    ~AVCaption2Ass() {  };
    bool enabled() const {
        return false;
    }
    void close() {
    }
    RGY_ERR init(std::shared_ptr<RGYLog> pLog, C2AFormat format) {
        return RGY_ERR_NONE;
    }
    C2AFormat format() const {
        return FORMAT_INVALID;
    }
    AVCAPTION_STATE state() const {
        return AVCAPTION_DISABLED;
    }
    void disable() {
    }
    void reset() {
    }
    void setIndex(int streamIndex, int trackId) {
    }
    void setOutputResolution(int w, int h, int sar_x, int sar_y) {
    }
    void printParam(int log_level) {
    }
    void setVidFirstKeyPts(int64_t pts) {
    }
    AVDemuxStream stream() const {
        AVDemuxStream stream;
        memset(&stream, 0, sizeof(AVDemuxStream));
        return stream;
    }
    RGY_ERR proc(uint8_t *buf, int buf_size, decltype(AVDemuxer::qStreamPktL1)& qStreamPkt) {
        return RGY_ERR_NONE;
    }
protected:
};
#pragma warning(pop)
#endif //#if ENABLE_CAPTION2ASS

class RGYInputAvcodecPrm : public RGYInputPrm {
public:
    int            inputRetry;              //ファイルオープンを再試行する回数
    uint8_t        memType;                 //使用するメモリの種類
    const TCHAR   *pInputFormat;            //入力フォーマット
    bool           readVideo;               //映像の読み込みを行うかどうか
    int            videoTrack;              //動画トラックの選択
    int            videoStreamId;           //動画StreamIdの選択
    uint32_t       readAudio;               //音声の読み込みを行うかどうか (AVQSV_AUDIO_xxx)
    bool           readSubtitle;            //字幕の読み込みを行うかどうか
    bool           readData;                //データの読み込みを行うかどうか
    bool           readAttachment;          //Attachmentの読み込みを行うかどうか
    bool           readChapter;             //チャプターの読み込みを行うかどうか
    rgy_rational<int> videoAvgFramerate;       //動画のフレームレート
    double         analyzeSec;              //入力ファイルを分析する秒数
    int64_t        probesize;               //probeするデータサイズ
    int            nTrimCount;              //Trimする動画フレームの領域の数
    sTrim         *pTrimList;               //Trimする動画フレームの領域のリスト
    int            fileIndex;               //audio-source, sub-source等のファイルインデックス、動画と同じファイルなら-1
    int            trackStartAudio;         //音声のトラック番号の開始点
    int            trackStartSubtitle;      //字幕のトラック番号の開始点
    int            trackStartData;          //データのトラック番号の開始点
    int            nAudioSelectCount;       //muxする音声のトラック数
    AudioSelect **ppAudioSelect;            //muxする音声のトラック番号のリスト 1,2,...(1から連番で指定)
    int            nSubtitleSelectCount;    //muxする字幕のトラック数
    SubtitleSelect **ppSubtitleSelect;      //muxする字幕のトラック番号のリスト 1,2,...(1から連番で指定)
    int            nDataSelectCount;        //muxするデータのトラック数
    DataSelect   **ppDataSelect;            //muxするデータのトラック番号のリスト 1,2,...(1から連番で指定)
    int            nAttachmentSelectCount;  //muxするAttachmentのトラック数
    DataSelect   **ppAttachmentSelect;      //muxするAttachmentのトラック番号のリスト 1,2,...(1から連番で指定)
    RGYAVSync      AVSyncMode;              //音声・映像同期モード
    int            procSpeedLimit;          //プリデコードする場合の処理速度制限 (0で制限なし)
    float          seekSec;                 //指定された秒数分先頭を飛ばす
    tstring        logFramePosList;         //FramePosListの内容を入力終了時に出力する (デバッグ用)
    tstring        logCopyFrameData;        //frame情報copy関数のログ出力先 (デバッグ用)
    tstring        logPackets;              //読み込んだパケットの情報を出力する
    int            threadInput;             //入力スレッドを有効にする
    RGYParamThread threadParamInput;        //入力スレッドのスレッドアフィニティ
    PerfQueueInfo *queueInfo;               //キューの情報を格納する構造体
    DeviceCodecCsp *HWDecCodecCsp;          //HWデコーダのサポートするコーデックと色空間
    bool           videoDetectPulldown;     //pulldownの検出を試みるかどうか
    C2AFormat      caption2ass;             //caption2assの処理の有効化
    bool           parseHDRmetadata;        //HDR関連のmeta情報を取得する
    bool           hdr10plusMetadataCopy;  //HDR10plus関連のmeta情報を取得する
    bool           interlaceAutoFrame;      //フレームごとにインタレの検出を行う
    RGYListRef<RGYFrameDataQP> *qpTableListRef; //qp tableを格納するときのベース構造体
    bool           lowLatency;
    RGYOptList     inputOpt;                //入力オプション

    RGYInputAvcodecPrm(RGYInputPrm base);
    virtual ~RGYInputAvcodecPrm() {};
};

class RGYInputAvcodec : public RGYInput
{
public:
    RGYInputAvcodec();
    virtual ~RGYInputAvcodec();

    virtual void Close() override;

    //動画ストリームの1フレーム分のデータをm_sPacketに格納する
    //m_sPacketからの取得はGetNextBitstreamで行う
    virtual RGY_ERR LoadNextFrame(RGYFrame *pSurface) override;

    //動画ストリームの1フレーム分のデータをbitstreamに追加する (リーダー側のデータは消す)
    virtual RGY_ERR GetNextBitstream(RGYBitstream *pBitstream) override;

    //動画ストリームの1フレーム分のデータをbitstreamに追加する (リーダー側のデータは残す)
    virtual RGY_ERR GetNextBitstreamNoDelete(RGYBitstream *pBitstream) override;

    //ストリームのヘッダ部分を取得する
    virtual RGY_ERR GetHeader(RGYBitstream *pBitstream) override;

    //入力ファイルのグローバルメタデータを取得する
    const AVDictionary *GetInputFormatMetadata();

    //動画の入力情報を取得する
    const AVStream *GetInputVideoStream();

    //動画の長さを取得する
    double GetInputVideoDuration();

    //音声・字幕パケットの配列を取得する
    virtual vector<AVPacket> GetStreamDataPackets(int inputFrame) override;

    //音声・字幕のコーデックコンテキストを取得する
    virtual vector<AVDemuxStream> GetInputStreamInfo() override;

    //AVMEDIA_TYPE_ATTACHMENTのstreamを取得する
    vector<const AVStream *> GetInputAttachmentStreams();

    //チャプターリストを取得する
    vector<const AVChapter *> GetChapterList();

    //フレーム情報構造へのポインタを返す
    FramePosList *GetFramePosList();

    virtual rgy_rational<int> getInputTimebase() override;

    //入力ファイルに存在する音声のトラック数を返す
    int GetAudioTrackCount() override;

    //入力ファイルに存在する字幕のトラック数を返す
    int GetSubtitleTrackCount() override;

    //入力ファイルに存在するデータのトラック数を返す
    int GetDataTrackCount() override;

    //動画の最初のフレームのptsを取得する
    int64_t GetVideoFirstKeyPts();

    //入力に使用する予定のdeviceIDを取得する
    int GetHWDecDeviceID();

    //入力スレッドのハンドルを取得する
    HANDLE getThreadHandleInput();

    //出力する動画の情報をセット
    void setOutputVideoInfo(int w, int h, int sar_x, int sar_y, bool mux);

    //HDRのmetadataへのポインタを返す
    const AVMasteringDisplayMetadata *getMasteringDisplay() const;
    const AVContentLightMetadata *getContentLight() const;

    RGYFrameDataHDR10plus *getHDR10plusMetaData(const AVFrame* frame);
    RGYFrameDataHDR10plus *getHDR10plusMetaData(const AVPacket* pkt);
    RGYFrameDataDOVIRpu *getDoviRpu(const AVFrame *frame);

#if USE_CUSTOM_INPUT
    int readPacket(uint8_t *buf, int buf_size);
    int writePacket(uint8_t *buf, int buf_size);
    int64_t seek(int64_t offset, int whence);
#endif //USE_CUSTOM_INPUT
protected:
    virtual RGY_ERR Init(const TCHAR *strFileName, VideoInfo *inputInfo, const RGYInputPrm *prm) override;

    RGY_ERR parseHDRData();

    RGY_ERR parseHDR10plus(AVPacket *pkt);

    RGY_ERR initFormatCtx(const TCHAR *strFileName, const RGYInputAvcodecPrm *input_prm, const int iretry);
    RGY_ERR initVideoBsfs();
    RGY_ERR initVideoParser();
    RGY_ERR parseVideoExtraData(const AVPacket *pkt);

    //streamで指定されたtrackの言語属性がlangの言語と等しいか
    bool isSelectedLangTrack(const std::string &lang, const AVStream *stream);

    //streamで指定されたtrackのコーデックがselectCodecのコーデックと等しいか
    bool isSelectedCodecTrack(const std::string &selectCodec, const AVStream *stream);

    void SetExtraData(AVCodecParameters *codecParam, const uint8_t *data, uint32_t size);

    //avcodecのコーデックIDからHWデコード可能ならRGY_CODECを返す
    RGY_CODEC checkHWDecoderAvailable(AVCodecID id, AVPixelFormat pixfmt, const CodecCsp *HWDecCodecCsp);

    //情報を取得できた動画ストリームがあるかを確認
    bool hasVideoWithStreamInfo() const;

    //avcodecのストリームIDを取得 (typeはAVMEDIA_TYPE_xxxxx)
    vector<int> getStreamIndex(AVMediaType type);

    //VC-1のスタートコードの確認
    bool vc1StartCodeExists(uint8_t *ptr);

    //対象ストリームのパケットを取得
    int getSample(AVPacket *pkt, bool bTreatFirstPacketAsKeyframe = false);

    //対象・字幕の音声パケットを追加するかどうか
    bool checkStreamPacketToAdd(AVPacket *pkt, AVDemuxStream *stream);

    //対象のパケットの必要な対象のストリーム情報へのポインタ
    AVDemuxStream *getPacketStreamData(const AVPacket *pkt);

    //qStreamPktL1をチェックし、framePosListから必要な音声パケットかどうかを判定し、
    //必要ならqStreamPktL2に移し、不要ならパケットを開放する
    void CheckAndMoveStreamPacketList();

    //音声パケットの配列を取得する (映像を読み込んでいないときに使用)
    void GetAudioDataPacketsWhenNoVideoRead(int inputFrame);

    //対象音声ストリームのキューの中の最初のパケットを探す
    const AVPacket *findFirstAudioStreamPackets(const AVDemuxStream& streamInfo);

    //QSVでデコードした際の最初のフレームのptsを取得する
    //さらに、平均フレームレートを推定する
    //fpsDecoderはdecoderの推定したfps
    RGY_ERR getFirstFramePosAndFrameRate(const sTrim *pTrimList, int nTrimCount, bool bDetectpulldown, bool lowLatency, rgy_rational<int> fpsOverride);

    //読み込みスレッド関数
    RGY_ERR ThreadFuncRead(RGYParamThread threadParam);

    //指定したptsとtimebaseから、該当する動画フレームを取得する
    int getVideoFrameIdx(int64_t pts, AVRational timebase, int iStart);

    //ptsを動画のtimebaseから音声のtimebaseに変換する
    int64_t convertTimebaseVidToStream(int64_t pts, const AVDemuxStream *stream);

    void hevcMp42Annexb(AVPacket *pkt);

    //VC-1のヘッダの修正を行う
    void vc1FixHeader(int nLengthFix = -1);

    //VC-1のフレームヘッダを追加
    void vc1AddFrameHeader(AVPacket *pkt);

    void CloseStream(AVDemuxStream *audio);
    void CloseVideo(AVDemuxVideo *video);
    void CloseFormat(AVDemuxFormat *format);
    void CloseThread();

    AVDemuxer        m_Demux;                      //デコード用情報
    tstring          m_logFramePosList;           //FramePosListの内容を入力終了時に出力する (デバッグ用)
    std::unique_ptr<FILE, fp_deleter> m_fpPacketList; // 読み取ったパケット情報を出力するファイル
    vector<uint8_t>  m_hevcMp42AnnexbBuffer;       //HEVCのmp4->AnnexB簡易変換用バッファ
    AVCaption2Ass    m_cap2ass;
};

#endif //ENABLE_AVSW_READER

#endif //__RGY_INPUT_AVCODEC_H__
