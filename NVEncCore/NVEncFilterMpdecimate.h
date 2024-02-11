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

class NVEncFilterParamMpdecimate : public NVEncFilterParam {
public:
    VppMpdecimate mpdecimate;
    tstring outfilename;

    NVEncFilterParamMpdecimate() : mpdecimate(), outfilename() {};
    virtual ~NVEncFilterParamMpdecimate() {};
    virtual tstring print() const override;
};

using funcCalcDiff = std::function<RGY_ERR(const RGYFrameInfo *, const RGYFrameInfo *, CUMemBufPair *,
    const int, const int, const bool, cudaStream_t, cudaEvent_t, cudaStream_t)>;

class NVEncFilterMpdecimateFrameData {
public:
    NVEncFilterMpdecimateFrameData(std::shared_ptr<RGYLog> log);
    ~NVEncFilterMpdecimateFrameData();

    CUFrameBuf *get() { return &m_buf; }
    const CUFrameBuf *get() const { return &m_buf; }
    RGY_ERR set(const RGYFrameInfo *pInputFrame, int inputFrameId, cudaStream_t stream);
    int id() const { return m_inFrameId; }
    void reset() { m_inFrameId = -1; }
    RGY_ERR calcDiff(const NVEncFilterMpdecimateFrameData *ref,
        cudaStream_t streamDiff, cudaEvent_t eventTransfer, cudaStream_t streamTransfer);
    bool checkIfFrameCanbeDropped(const int hi, const int lo, const float factor);
private:
    std::shared_ptr<RGYLog> m_log;
    int m_inFrameId;
    CUFrameBuf m_buf;
    CUFrameBufPair m_tmp;
};

class NVEncFilterMpdecimateCache {
public:
    NVEncFilterMpdecimateCache();
    ~NVEncFilterMpdecimateCache();
    void init(int bufCount, std::shared_ptr<RGYLog> log);
    RGY_ERR add(const RGYFrameInfo *pInputFrame, cudaStream_t stream = 0);
    void removeFromCache(int iframe) {
        for (auto &f : m_frames) {
            if (f->id() == iframe) {
                f->reset();
                return;
            }
        }
    }
    NVEncFilterMpdecimateFrameData *frame(int iframe) {
        for (auto &f : m_frames) {
            if (f->id() == iframe) {
                return f.get();
            }
        }
        return nullptr;
    }
    NVEncFilterMpdecimateFrameData *getEmpty() {
        for (auto &f : m_frames) {
            if (f->id() < 0) {
                return f.get();
            }
        }
        return nullptr;
    }
    CUFrameBuf *get(int iframe) {
        return frame(iframe)->get();
    }
    int inframe() const { return m_inputFrames; }
private:
    std::shared_ptr<RGYLog> m_log;
    int m_inputFrames;
    std::vector<std::unique_ptr<NVEncFilterMpdecimateFrameData>> m_frames;
};

class NVEncFilterMpdecimate : public NVEncFilter {
public:
    NVEncFilterMpdecimate();
    virtual ~NVEncFilterMpdecimate();
    virtual RGY_ERR init(shared_ptr<NVEncFilterParam> pParam, shared_ptr<RGYLog> pPrintMes) override;
protected:
    virtual RGY_ERR run_filter(const RGYFrameInfo *pInputFrame, RGYFrameInfo **ppOutputFrames, int *pOutputFrameNum, cudaStream_t stream) override;
    virtual void close() override;
    virtual RGY_ERR checkParam(const std::shared_ptr<NVEncFilterParamMpdecimate> pParam);
    bool dropFrame(NVEncFilterMpdecimateFrameData *targetFrame);

    int m_dropCount;
    int m_ref;
    int m_target;
    NVEncFilterMpdecimateCache m_cache;
    std::unique_ptr<cudaEvent_t, cudaevent_deleter> m_eventDiff;
    std::unique_ptr<cudaEvent_t, cudaevent_deleter> m_eventTransfer;
    std::unique_ptr<cudaStream_t, cudastream_deleter> m_streamDiff;
    std::unique_ptr<cudaStream_t, cudastream_deleter> m_streamTransfer;
    unique_ptr<FILE, fp_deleter> m_fpLog;
};
