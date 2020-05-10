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
#include "rgy_version.h"
#include "rgy_err.h"
#include "convert_csp.h"
#if !FOR_AUO && ENCODER_NVENC
#include "rgy_cuda_util.h"
#endif //#if !FOR_AUO && ENCODER_NVENC

enum RGYFrameDataType {
    RGY_FRAME_DATA_NONE,
    RGY_FRAME_DATA_QP,
    RGY_FRAME_DATA_HDR10PLUS,

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
    std::unique_ptr<cudaStream_t, cudastream_deleter> m_stream;
#endif //#if !FOR_AUO && ENCODER_NVENC
    FrameInfo m_qpHost;
};

class RGYFrameDataHDR10plus : public RGYFrameData {
public:
    RGYFrameDataHDR10plus();
    RGYFrameDataHDR10plus(const uint8_t* data, size_t size, int64_t timestamp);
    virtual ~RGYFrameDataHDR10plus();

    const std::vector<uint8_t>& getData() { return m_data; }
    int64_t timestamp() const { return m_timestamp; }
protected:
    int64_t m_timestamp;
    std::vector<uint8_t> m_data;
};

#endif //__RGY_FRAME_H__
