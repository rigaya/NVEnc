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

#include "rgy_util.h"
#include "rgy_frame.h"

RGYFrameDataQP::RGYFrameDataQP() :
    m_frameType(0),
    m_qpScaleType(0),
#if ENCODER_NVENC
    m_qpDev(),
    m_event(std::unique_ptr<cudaEvent_t, cudaevent_deleter>(nullptr, cudaevent_deleter())),
    m_stream(std::unique_ptr<cudaStream_t, cudastream_deleter>(nullptr, cudastream_deleter())),
#endif //#if ENCODER_NVENC
    m_qpHost() {
    m_dataType = RGY_FRAME_DATA_QP;
};

RGYFrameDataQP::~RGYFrameDataQP() {
    m_qpDev.reset();
    if (m_qpHost.ptr) {
        cudaFree(m_qpHost.ptr);
        m_qpHost.ptr = nullptr;
    }
    m_event.reset();
};

RGY_ERR RGYFrameDataQP::setQPTable(const int8_t *qpTable, int qpw, int qph, int qppitch, int scaleType, int frameType, int64_t timestamp) {
    m_qpScaleType = scaleType;
    m_frameType = frameType;
    if (m_qpHost.ptr == nullptr
        || m_qpHost.width != qpw
        || m_qpHost.height != qph) {
        m_qpHost.csp = RGY_CSP_Y8;
        m_qpHost.width = qpw;
        m_qpHost.height = qph;
        m_qpHost.flags = RGY_FRAME_FLAG_NONE;
        m_qpHost.pitch = ALIGN(m_qpHost.width, 128);
        m_qpHost.deivce_mem = false;
        m_qpHost.duration = 0;
        m_qpHost.timestamp = timestamp;
        m_qpHost.picstruct = RGY_PICSTRUCT_FRAME;
        m_qpHost.dataList.clear();
#if ENCODER_NVENC
        if (m_qpHost.ptr) {
            cudaFree(m_qpHost.ptr);
            m_qpHost.ptr = nullptr;
        }
        auto cudaerr = cudaMallocHost(&m_qpHost.ptr, m_qpHost.pitch * m_qpHost.height);
        if (cudaerr != cudaSuccess) {
            return RGY_ERR_MEMORY_ALLOC;
        }
#else
        if (m_qpHost.ptr) {
            _aligned_free(m_qpHost.ptr);
            m_qpHost.ptr = nullptr;
        }
        m_qpHost.ptr = (uint8_t *)_aligned_malloc(m_qpHost.pitch * m_qpHost.height, 64);
        if (m_qpHost.ptr == nullptr) {
            return RGY_ERR_MEMORY_ALLOC;
        }
#endif //#if ENCODER_NVENC
    }
    for (int y = 0; y < m_qpHost.height; y++) {
        memcpy(m_qpHost.ptr + y * m_qpHost.pitch, qpTable + y * qppitch, m_qpHost.width);
    }
    return RGY_ERR_NONE;
}

#if ENCODER_NVENC
RGY_ERR RGYFrameDataQP::transferToGPU(cudaStream_t stream) {
    if (!m_qpDev) {
        m_qpDev = std::make_unique<CUFrameBuf>(m_qpHost.width, m_qpHost.height, m_qpHost.csp);
    }
    if (!m_stream) {
        m_stream = std::unique_ptr<cudaStream_t, cudastream_deleter>(new cudaStream_t(), cudastream_deleter());
        auto cudaerr = cudaStreamCreateWithFlags(m_stream.get(), cudaStreamNonBlocking);
        if (cudaerr != cudaSuccess) {
            return RGY_ERR_CUDA;
        }
    }
    if (!m_event) {
        m_event = std::unique_ptr<cudaEvent_t, cudaevent_deleter>(new cudaEvent_t(), cudaevent_deleter());
        auto cudaerr = cudaEventCreate(m_event.get());
        if (cudaerr != cudaSuccess) {
            return RGY_ERR_CUDA;
        }
    }
    auto cudaerr = copyFrameDataAsync(&m_qpDev->frame, &m_qpHost, *m_stream.get());
    if (cudaerr != cudaSuccess) {
        return RGY_ERR_MEMORY_ALLOC;
    }
    cudaerr = cudaEventRecord(*m_event.get(), *m_stream.get());
    if (cudaerr != cudaSuccess) {
        return RGY_ERR_CUDA;
    }
    return RGY_ERR_NONE;
}
#endif //#if ENCODER_NVENC
