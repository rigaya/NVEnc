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

#pragma once

#include "NVEncFilter.h"
#include "NVEncParam.h"

static const int STREAM_OPT = 1;

#define AFS_SOURCE_CACHE_NUM 16
#define AFS_SCAN_CACHE_NUM   16
#define AFS_STRIPE_CACHE_NUM 16

#define AFS_FLAG_SHIFT0      0x01
#define AFS_FLAG_SHIFT1      0x02
#define AFS_FLAG_SHIFT2      0x04
#define AFS_FLAG_SHIFT3      0x08
#define AFS_FLAG_FRAME_DROP  0x10
#define AFS_FLAG_SMOOTHING   0x20
#define AFS_FLAG_FORCE24     0x40
//#define AFS_FLAG_ERROR       0x80
#define AFS_FLAG_PROGRESSIVE 0x80
#define AFS_FLAG_RFF         0x10
#define AFS_MASK_SHIFT0      0xfe
#define AFS_MASK_SHIFT1      0xfd
#define AFS_MASK_SHIFT2      0xfb
#define AFS_MASK_SHIFT3      0xf7
#define AFS_MASK_FRAME_DROP  0xef
#define AFS_MASK_SMOOTHING   0xdf
#define AFS_MASK_FORCE24     0xbf
#define AFS_MASK_ERROR       0x7f

#define AFS_STATUS_DEFAULT   0

#define AFS_SHARE_SIZE       0x0018
#define AFS_OFFSET_SHARE_N   0x0000
#define AFS_OFFSET_SHARE_ERR 0x0004
#define AFS_OFFSET_FRAME_N   0x0008
#define AFS_OFFSET_STARTFRM  0x000C
#define AFS_OFFSET_STATUSPTR 0x0010

#define ISRFF(x) (((x) & (AFS_FLAG_PROGRESSIVE | AFS_FLAG_RFF)) == (AFS_FLAG_PROGRESSIVE | AFS_FLAG_RFF))

class NVEncFilterParamAfs : public NVEncFilterParam {
public:
    VppAfs afs;
    rgy_rational<int> inFps;
    rgy_rational<int> inTimebase;
    rgy_rational<int> outTimebase;
    tstring outFilename;
    CUctx_flags cudaSchedule;

    NVEncFilterParamAfs() : afs(), inFps(), inTimebase(), outTimebase(), outFilename(), cudaSchedule(CU_CTX_SCHED_BLOCKING_SYNC) {

    };
    virtual ~NVEncFilterParamAfs() {};
    virtual tstring print() const override;
};

class afsSourceCache {
public:
    afsSourceCache();
    ~afsSourceCache();

    RGY_ERR alloc(const RGYFrameInfo& frameInfo);

    RGY_ERR add(const RGYFrameInfo *pInputFrame, cudaStream_t stream);

    RGY_ERR sep_field_uv(RGYFrameInfo *pDstFrame, const RGYFrameInfo *pSrcFrame, cudaStream_t stream);

    CUFrameBuf *get(int iframe) {
        iframe = clamp(iframe, 0, m_nFramesInput-1);
        return &m_sourceArray[iframe & (AFS_SOURCE_CACHE_NUM-1)];
    }
    int inframe() const { return m_nFramesInput; }
    void clear();
protected:
    CUFrameBuf m_sourceArray[AFS_SOURCE_CACHE_NUM];
    int m_nFramesInput;
};

struct AFS_SCAN_DATA {
    CUFrameBuf map;
    int status, frame, mode, tb_order, thre_shift, thre_deint, thre_Ymotion, thre_Cmotion;
    AFS_SCAN_CLIP clip;
    int ff_motion, lf_motion;
    unique_ptr<cudaEvent_t, cudaevent_deleter> cuevent;
};

class afsScanCache {
public:
    afsScanCache();
    ~afsScanCache();

    void clearcache(int iframe);
    void initcache(int iframe);

    RGY_ERR alloc(const RGYFrameInfo& frameInfo);

    AFS_SCAN_DATA *get(int iframe) {
        return &m_scanArray[iframe & (AFS_SCAN_CACHE_NUM-1)];
    }

    void clear();
protected:
    AFS_SCAN_DATA m_scanArray[AFS_SCAN_CACHE_NUM];
};

struct AFS_STRIPE_DATA {
    CUFrameBuf map;
    int status, frame, count0, count1;
    unique_ptr<cudaEvent_t, cudaevent_deleter> cuevent;
    CUMemBufPair buf_count_stripe;
};

class afsStripeCache {
public:
    afsStripeCache();
    ~afsStripeCache();

    void clearcache(int iframe);
    void initcache(int iframe);
    void expire(int iframe);

    RGY_ERR alloc(const RGYFrameInfo& frameInfo);

    AFS_STRIPE_DATA *get(int iframe) {
        return &m_stripeArray[iframe & (AFS_STRIPE_CACHE_NUM-1)];
    }
    AFS_STRIPE_DATA *filter(int iframe, int analyze, cudaStream_t stream, RGY_ERR *pErr);

    void clear();
protected:
    RGY_ERR map_filter(AFS_STRIPE_DATA *dst, AFS_STRIPE_DATA *sp, cudaStream_t stream);

    AFS_STRIPE_DATA *getFiltered() {
        return &m_stripeArray[AFS_STRIPE_CACHE_NUM];
    }
    AFS_STRIPE_DATA m_stripeArray[AFS_STRIPE_CACHE_NUM + 1];
};

class afsStatus {
public:
    afsStatus() : m_ptr(nullptr), m_buf_size(0) { };
    ~afsStatus() { clear(); };

    uint8_t& operator[](int iframe) {
        iframe = std::max<int>(0, iframe);
        if (iframe >= m_buf_size) {
            const int new_bufsize = std::max(128, m_buf_size * 2);
            m_ptr = (uint8_t *)realloc(m_ptr, new_bufsize * sizeof(uint8_t));
            memset(m_ptr + m_buf_size, 0, (new_bufsize - m_buf_size) * sizeof(uint8_t));
            m_buf_size = new_bufsize;
        }
        return m_ptr[iframe];
    }
    void clear() {
        if (m_ptr) {
            free(m_ptr);
            m_ptr = nullptr;
        }
        m_buf_size = 0;
    }
protected:
    uint8_t *m_ptr;
    int m_buf_size;
};

struct afsFrameTs {
    int64_t pos;
    int64_t orig_pts;
    RGY_PICSTRUCT picstruct;
    int iframe;
};

class afsStreamStatus {
public:
    static const int64_t AFS_SSTS_DROP  = -1;
    static const int64_t AFS_SSTS_ERROR = -2;
    afsStreamStatus();
    ~afsStreamStatus();

    int open_log(const tstring& log_filename);
    void init(uint8_t status, int drop24);
    int set_status(int iframe, uint8_t status, int drop24, int64_t orig_pts);
    int64_t get_duration(int64_t iframe);
private:
    void write_log(const afsFrameTs *const frameTs);

    bool m_initialized;
    int m_quarter_jitter;
    int m_additional_jitter;
    int m_phase24;
    int m_position24;
    int m_prev_jitter;
    int m_prev_rff_smooth;
    uint8_t m_prev_status;
    int64_t m_set_frame;
    afsFrameTs m_pos[16];
    unique_ptr<FILE, fp_deleter> m_fpLog;
};

class NVEncFilterAfs : public NVEncFilter {
public:
    NVEncFilterAfs();
    virtual ~NVEncFilterAfs();
    virtual RGY_ERR init(shared_ptr<NVEncFilterParam> pParam, shared_ptr<RGYLog> pPrintMes) override;
protected:
    virtual RGY_ERR run_filter(const RGYFrameInfo *pInputFrame, RGYFrameInfo **ppOutputFrames, int *pOutputFrameNum, cudaStream_t stream) override;
    virtual void close() override;
    RGY_ERR check_param(shared_ptr<NVEncFilterParamAfs> pAfsParam);

    RGY_ERR analyze_stripe(CUFrameBuf *p0, CUFrameBuf *p1, AFS_SCAN_DATA *sp, CUMemBufPair *count_motion, const NVEncFilterParamAfs *pAfsPrm, cudaStream_t stream);
    bool scan_frame_result_cached(int iframe, const VppAfs *pAfsPrm);
    RGY_ERR scan_frame(int iframe, int force, const NVEncFilterParamAfs *pAfsPrm, cudaStream_t stream);
    RGY_ERR count_motion(AFS_SCAN_DATA *sp, const AFS_SCAN_CLIP *clip);

    RGY_ERR merge_scan(AFS_STRIPE_DATA *sp, AFS_SCAN_DATA *sp0, AFS_SCAN_DATA *sp1, CUMemBufPair *count_stripe, const NVEncFilterParamAfs *pAfsPrm, cudaStream_t stream);
    RGY_ERR count_stripe(AFS_STRIPE_DATA *sp, const AFS_SCAN_CLIP *clip, int tb_order);

    RGY_ERR get_stripe_info(int frame, int mode, const NVEncFilterParamAfs *pAfsPrm);
    int detect_telecine_cross(int iframe, int coeff_shift);
    RGY_ERR analyze_frame(int iframe, const NVEncFilterParamAfs *pAfsPrm, int reverse[4], int assume_shift[4], int result_stat[4]);

    RGY_ERR synthesize(int iframe, CUFrameBuf *pOut, CUFrameBuf *p0, CUFrameBuf *p1, AFS_STRIPE_DATA *sip, const NVEncFilterParamAfs *pAfsPrm, cudaStream_t stream);

    int open_timecode(tstring tc_filename);
    void write_timecode(int64_t pts, const rgy_rational<int>& timebase);

    int open_log(tstring log_filename);
    void write_log(int64_t pts, const rgy_rational<int>& timebase);

    unique_ptr<cudaStream_t, cudastream_deleter> m_streamAnalyze;
    unique_ptr<cudaStream_t, cudastream_deleter> m_streamCopy;
    unique_ptr<cudaEvent_t, cudaevent_deleter> m_eventSrcAdd;
    unique_ptr<cudaEvent_t, cudaevent_deleter> m_eventScanFrame;
    unique_ptr<cudaEvent_t, cudaevent_deleter> m_eventMergeScan;
    int m_nFrame;
    int64_t m_nPts;

    afsSourceCache  m_source;
    afsScanCache    m_scan;
    afsStripeCache  m_stripe;
    afsStatus       m_status;
    afsStreamStatus m_streamsts;
    CUMemBufPair    m_count_motion;
    unique_ptr<FILE, fp_deleter> m_fpTimecode;
};
