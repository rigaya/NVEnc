// -----------------------------------------------------------------------------------------
// QSVEnc/NVEnc by rigaya
// -----------------------------------------------------------------------------------------
// The MIT License
//
// Copyright (c) 2020 rigaya
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

#pragma once
#ifndef __RGY_FRAME_H__
#define __RGY_FRAME_H__

#include <memory>
#include <array>
#include "rgy_version.h"
#include "rgy_err.h"
#include "convert_csp.h"
#if !FOR_AUO && ENCODER_NVENC
#include "rgy_cuda_util.h"
#endif //#if !FOR_AUO && ENCODER_NVENC

enum RGYFrameDataType {
    RGY_FRAME_DATA_NONE,
    RGY_FRAME_DATA_QP,
    RGY_FRAME_DATA_METADATA,
    RGY_FRAME_DATA_HDR10PLUS,
    RGY_FRAME_DATA_DOVIRPU,

    RGY_FRAME_DATA_MAX,
};

class RGYFrameData {
public:
    RGYFrameData() : m_dataType(RGY_FRAME_DATA_NONE) {};
    virtual ~RGYFrameData() {};
    RGYFrameDataType dataType() const { return m_dataType; }
protected:
    RGYFrameDataType m_dataType;
};

struct CUFrameBuf;

class RGYFrameDataQP : public RGYFrameData {
public:
    RGYFrameDataQP();
    virtual ~RGYFrameDataQP();
    RGY_ERR setQPTable(const int8_t *qpTable, int qpw, int qph, int qppitch, int scaleType, int frameType, int64_t timestamp);
#if !FOR_AUO && ENCODER_NVENC
    RGY_ERR transferToGPU(cudaStream_t stream);
#endif //#if !FOR_AUO && ENCODER_NVENC
    int frameType() const { return m_frameType; }
    int qpScaleType() const { return m_qpScaleType; }
#if !FOR_AUO && ENCODER_NVENC
    cudaEvent_t event() { return *m_event.get(); }
    CUFrameBuf *qpDev() { return m_qpDev.get(); }
#endif //#if !FOR_AUO && ENCODER_NVENC
protected:
    int m_frameType;
    int m_qpScaleType;
#if !FOR_AUO && ENCODER_NVENC
    std::unique_ptr<CUFrameBuf> m_qpDev;
    std::unique_ptr<cudaEvent_t, cudaevent_deleter> m_event;
#endif //#if !FOR_AUO && ENCODER_NVENC
    RGYFrameInfo m_qpHost;
};

class RGYFrameDataMetadata : public RGYFrameData {
public:
    RGYFrameDataMetadata();
    RGYFrameDataMetadata(const uint8_t* data, size_t size, int64_t timestamp);
    virtual ~RGYFrameDataMetadata();

    virtual std::vector<uint8_t> gen_nal() const = 0;
    virtual std::vector<uint8_t> gen_obu() const = 0;
    const std::vector<uint8_t>& getData() const { return m_data; }
    int64_t timestamp() const { return m_timestamp; }
protected:
    int64_t m_timestamp;
    std::vector<uint8_t> m_data;
};

class RGYFrameDataHDR10plus : public RGYFrameDataMetadata {
public:
    RGYFrameDataHDR10plus();
    RGYFrameDataHDR10plus(const uint8_t* data, size_t size, int64_t timestamp);
    virtual ~RGYFrameDataHDR10plus();
    virtual std::vector<uint8_t> gen_nal() const override;
    virtual std::vector<uint8_t> gen_obu() const override;
};

class RGYFrameDataDOVIRpu : public RGYFrameDataMetadata {
public:
    RGYFrameDataDOVIRpu();
    RGYFrameDataDOVIRpu(const uint8_t* data, size_t size, int64_t timestamp);
    virtual ~RGYFrameDataDOVIRpu();
    virtual std::vector<uint8_t> gen_nal() const override;
    virtual std::vector<uint8_t> gen_obu() const override;
};
#if !ENCODER_NVENC
struct RGYFrame {
public:
    RGYFrame() {};
    virtual ~RGYFrame() { }
    virtual bool isempty() const = 0;
    std::array<void*, _countof(RGYFrameInfo::ptr)> ptr() const {
        auto frame = getInfo();
        std::array<void*, _countof(RGYFrameInfo::ptr)> ptrarray;
        for (size_t i = 0; i < ptrarray.size(); i++) {
            ptrarray[i] = (void *)frame.ptr[i];
        }
        return ptrarray;
    }
    void ptrArray(void *array[3], bool bRGB) {
        auto frame = getInfo();
        UNREFERENCED_PARAMETER(bRGB);
        array[0] = (void *)frame.ptr[0];
        array[1] = (void *)frame.ptr[1];
        array[2] = (void *)frame.ptr[2];
    }
    uint8_t *ptrY() const {
        return getInfo().ptr[0];
    }
    uint8_t *ptrUV() const {
        return getInfo().ptr[1];
    }
    uint8_t *ptrU() const {
        return getInfo().ptr[1];
    }
    uint8_t *ptrV() const {
        return getInfo().ptr[2];
    }
    uint8_t *ptrRGB() const {
        return getInfo().ptr[0];
    }
    RGY_CSP csp() const {
        return getInfo().csp;
    }
    RGY_MEM_TYPE mem_type() const {
        return getInfo().mem_type;
    }
    int width() const {
        return getInfo().width;
    }
    int height() const {
        return getInfo().height;
    }
    uint32_t pitch(int index = 0) const {
        return getInfo().pitch[index];
    }
    uint64_t timestamp() const {
        return getInfo().timestamp;
    }
    virtual void setTimestamp(uint64_t timestamp) = 0;
    int64_t duration() const {
        return getInfo().duration;
    }
    virtual void setDuration(uint64_t duration) = 0;
    RGY_PICSTRUCT picstruct() const {
        return getInfo().picstruct;
    }
    virtual void setPicstruct(RGY_PICSTRUCT picstruct) = 0;
    int inputFrameId() const {
        return getInfo().inputFrameId;
    }
    virtual void setInputFrameId(int id) = 0;
    RGY_FRAME_FLAGS flags() const {
        return getInfo().flags;
    }
    virtual void setFlags(RGY_FRAME_FLAGS flags) = 0;
    virtual void clearDataList() = 0;
    virtual const std::vector<std::shared_ptr<RGYFrameData>>& dataList() const = 0;
    virtual std::vector<std::shared_ptr<RGYFrameData>>& dataList() = 0;
    virtual void setDataList(const std::vector<std::shared_ptr<RGYFrameData>>& dataList) = 0;

    void setPropertyFrom(const RGYFrame *frame) {
        setDuration(frame->duration());
        setTimestamp(frame->timestamp());
        setInputFrameId(frame->inputFrameId());
        setPicstruct(frame->picstruct());
        setFlags(frame->flags());
        setDataList(frame->dataList());
    }
protected:
    virtual RGYFrameInfo getInfo() const = 0;
};
#endif

#if 0
struct RGYSysFrame : public RGYFrame {
public:
    RGYSysFrame();
    RGYSysFrame(const RGYFrameInfo& frame_);
    virtual ~RGYSysFrame();
    virtual RGY_ERR allocate(const int width, const int height, const RGY_CSP csp, const int bitdepth);
    virtual RGY_ERR allocate(const RGYFrameInfo &frame);
    virtual void deallocate();
    const RGYFrameInfo& frameInfo() { return frame; }
    virtual bool isempty() const { return !frame.ptr[0]; }
    virtual void setTimestamp(uint64_t timestamp) override { frame.timestamp = timestamp; }
    virtual void setDuration(uint64_t duration) override { frame.duration = duration; }
    virtual void setPicstruct(RGY_PICSTRUCT picstruct) override { frame.picstruct = picstruct; }
    virtual void setInputFrameId(int id) override { frame.inputFrameId = id; }
    virtual void setFlags(RGY_FRAME_FLAGS frameflags) override { frame.flags = frameflags; }
    virtual void clearDataList() override { frame.dataList.clear(); }
    virtual const std::vector<std::shared_ptr<RGYFrameData>>& dataList() const override { return frame.dataList; }
    virtual std::vector<std::shared_ptr<RGYFrameData>>& dataList() override { return frame.dataList; }
    virtual void setDataList(std::vector<std::shared_ptr<RGYFrameData>>& dataList) override { frame.dataList = dataList; }
protected:
    RGYSysFrame(const RGYSysFrame &) = delete;
    void operator =(const RGYSysFrame &) = delete;
    virtual RGYFrameInfo getInfo() const override {
        return frame;
    }
    RGYFrameInfo frame;
    bool allocatedFirstPlaneOnly; // 最初のplaneでフレーム全体を確保している
};
#endif

#endif //__RGY_FRAME_H__
