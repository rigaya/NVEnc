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

class NVEncFilterParamDecimate : public NVEncFilterParam {
public:
    VppDecimate decimate;
    tstring outfilename;

    NVEncFilterParamDecimate() : decimate(), outfilename() {};
    virtual ~NVEncFilterParamDecimate() {};
    virtual tstring print() const override;
};

using funcCalcDiff = std::function<cudaError_t(const RGYFrameInfo *, const RGYFrameInfo *, CUMemBufPair&,
    const int, const int, const bool, cudaStream_t, cudaEvent_t, cudaStream_t)>;

enum DecimateSelectResult : uint32_t {
    NONE         = 0x00000,
    ORDER        = 0x0ffff,
    DROP         = 0x10000,
    DUPLICATE    = 0x20000,
    SCENE_CHANGE = 0x40000,
};

static DecimateSelectResult operator|(DecimateSelectResult a, DecimateSelectResult b) {
    return (DecimateSelectResult)((uint32_t)a | (uint32_t)b);
}

static DecimateSelectResult operator|=(DecimateSelectResult &a, DecimateSelectResult b) {
    a = a | b;
    return a;
}

static DecimateSelectResult operator|=(DecimateSelectResult &a, uint32_t b) {
    a = a | (DecimateSelectResult)b;
    return a;
}

static DecimateSelectResult operator&(DecimateSelectResult a, DecimateSelectResult b) {
    return (DecimateSelectResult)((uint32_t)a & (uint32_t)b);
}

static DecimateSelectResult operator&=(DecimateSelectResult &a, DecimateSelectResult b) {
    a = (DecimateSelectResult)((uint32_t)a & (uint32_t)b);
    return a;
}

class NVEncFilterDecimateFrameData {
public:
    NVEncFilterDecimateFrameData();
    ~NVEncFilterDecimateFrameData();

    CUFrameBuf *get() { return &m_buf; }
    const CUFrameBuf *get() const { return &m_buf; }
    cudaError_t set(const RGYFrameInfo *pInputFrame, int inputFrameId, int blockSizeX, int blockSizeY, cudaStream_t stream);
    int id() const { return m_inFrameId; }
    cudaError_t calcDiff(funcCalcDiff func, const NVEncFilterDecimateFrameData *target, const bool chroma,
        cudaStream_t streamDiff, cudaEvent_t eventTransfer, cudaStream_t streamTransfer);
    void calcDiffFromTmp();

    int64_t diffMaxBlock() const { return m_diffMaxBlock; }
    int64_t diffTotal() const { return m_diffTotal; }
private:
    int m_inFrameId;
    int m_blockX;
    int m_blockY;
    CUFrameBuf m_buf;
    CUMemBufPair m_tmp;
    int64_t m_diffMaxBlock;
    int64_t m_diffTotal;
};


class NVEncFilterDecimateCache {
public:
    NVEncFilterDecimateCache();
    ~NVEncFilterDecimateCache();
    void init(int bufCount, int blockX, int blockY);
    cudaError_t add(const RGYFrameInfo *pInputFrame, cudaStream_t stream = 0);
    NVEncFilterDecimateFrameData *frame(int iframe) {
        iframe = clamp(iframe, 0, m_inputFrames - 1);
        return m_frames[iframe % m_frames.size()].get();
    }
    CUFrameBuf *get(int iframe) {
        return frame(iframe)->get();
    }
    int inframe() const { return m_inputFrames; }
private:
    int m_blockX;
    int m_blockY;
    int m_inputFrames;
    std::vector<std::unique_ptr<NVEncFilterDecimateFrameData>> m_frames;
};

class NVEncFilterDecimate : public NVEncFilter {
public:
    NVEncFilterDecimate();
    virtual ~NVEncFilterDecimate();
    virtual RGY_ERR init(shared_ptr<NVEncFilterParam> pParam, shared_ptr<RGYLog> pPrintMes) override;
protected:
    virtual RGY_ERR run_filter(const RGYFrameInfo *pInputFrame, RGYFrameInfo **ppOutputFrames, int *pOutputFrameNum, cudaStream_t stream) override;
    virtual void close() override;
    virtual RGY_ERR checkParam(const std::shared_ptr<NVEncFilterParamDecimate> pParam);
    RGY_ERR setOutputFrame(int64_t nextTimestamp, RGYFrameInfo **ppOutputFrames, int *pOutputFrameNum);

    std::vector<DecimateSelectResult> selectDropFrame(const int iframeStart);
    RGY_ERR calcDiffWithPrevFrameAndSetDiffToCurr(const int curr, const int prev, cudaStream_t stream);

    bool m_flushed;
    int m_frameLastDropped;
    int64_t m_frameLastInputDuration;
    int64_t m_threSceneChange;
    int64_t m_threDuplicate;
    NVEncFilterDecimateCache m_cache;
    std::unique_ptr<cudaEvent_t, cudaevent_deleter> m_eventDiff;
    std::unique_ptr<cudaEvent_t, cudaevent_deleter> m_eventTransfer;
    std::unique_ptr<cudaStream_t, cudastream_deleter> m_streamDiff;
    std::unique_ptr<cudaStream_t, cudastream_deleter> m_streamTransfer;
    unique_ptr<FILE, fp_deleter> m_fpLog;
};
