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

#include <map>
#include <array>
#include <iostream>
#include <fstream>
#include <algorithm>
#include <numeric>
#include <type_traits>
#define _USE_MATH_DEFINES
#include <cmath>
#include "convert_csp.h"
#include "NVEncFilterMpdecimate.h"
#include "NVEncParam.h"
#pragma warning (push)
#pragma warning (disable: 4819)
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#pragma warning (pop)
#include "rgy_cuda_util_kernel.h"

#define MPDECIMATE_BLOCK_X (32)
#define MPDECIMATE_BLOCK_Y (8)

__device__ __inline__
int func_diff_pix(int a, int b) {
    return abs(a - b);
}

template<typename Type4>
__global__ void kernel_block_diff(
    const uint8_t *__restrict__ p0, const int p0_pitch,
    const uint8_t *__restrict__ p1, const int p1_pitch,
    const int width, const int height,
    uint8_t *__restrict__ pDst, const int dst_pitch) {
    const int lx = threadIdx.x; //スレッド数=MPDECIMATE_BLOCK_X
    const int ly = threadIdx.y; //スレッド数=MPDECIMATE_BLOCK_Y
    const int blockoffset_x = blockIdx.x * blockDim.x;
    const int blockoffset_y = blockIdx.y * blockDim.y;
    const int imgx = (blockoffset_x + lx) * 8;
    const int imgy = (blockoffset_y + ly);

    int diff = 0;
    if (imgx < width && imgy < height) {
        p0 += imgy * p0_pitch + imgx * sizeof(Type4::x);
        p1 += imgy * p1_pitch + imgx * sizeof(Type4::x);
        Type4 *ptrp0 = (Type4 *)p0;
        Type4 *ptrp1 = (Type4 *)p1;
        {
            Type4 pix0 = ptrp0[0];
            Type4 pix1 = ptrp1[0];
            diff += func_diff_pix(pix0.x, pix1.x);
            if (imgx + 1 < width) diff += func_diff_pix(pix0.y, pix1.y);
            if (imgx + 2 < width) diff += func_diff_pix(pix0.z, pix1.z);
            if (imgx + 3 < width) diff += func_diff_pix(pix0.w, pix1.w);
        }
        if (imgx + 4 < width) {
            Type4 pix0 = ptrp0[1];
            Type4 pix1 = ptrp1[1];
            diff += func_diff_pix(pix0.x, pix1.x);
            if (imgx + 5 < width) diff += func_diff_pix(pix0.y, pix1.y);
            if (imgx + 6 < width) diff += func_diff_pix(pix0.z, pix1.z);
            if (imgx + 7 < width) diff += func_diff_pix(pix0.w, pix1.w);
        }
    }

    __shared__ int tmp[MPDECIMATE_BLOCK_Y][MPDECIMATE_BLOCK_X +1];
    tmp[ly][lx] = diff;
    __syncthreads();
    if (ly == 0) {
        #pragma unroll
        for (int i = 1; i < 8; i++) {
            diff += tmp[i][lx];
        }
        const int block8x8X = blockoffset_x + lx;
        const int block8x8Y = blockIdx.y;
        pDst += block8x8Y * dst_pitch + block8x8X * sizeof(diff);
        *(int *)pDst = diff;
    }
}

template<typename Type4>
RGY_ERR calc_block_diff_plane(const RGYFrameInfo *p0, const RGYFrameInfo *p1, RGYFrameInfo *tmp, cudaStream_t streamDiff) {
    const int width = p0->width;
    const int height = p0->height;
    dim3 blockSize(MPDECIMATE_BLOCK_X, MPDECIMATE_BLOCK_Y);
    dim3 gridSize(divCeil(width, blockSize.x * 8), divCeil(height, blockSize.y));
    kernel_block_diff<Type4><<< gridSize, blockSize, 0, streamDiff >>> (
        (const uint8_t *)p0->ptr, p0->pitch,
        (const uint8_t *)p1->ptr, p1->pitch,
        width, height,
        (uint8_t *)tmp->ptr, tmp->pitch);
    return err_to_rgy(cudaGetLastError());
}

template<typename Type4>
RGY_ERR calc_block_diff_frame(const RGYFrameInfo *p0, const RGYFrameInfo *p1, RGYFrameInfo *tmp, cudaStream_t streamDiff) {
    for (int i = 0; i < RGY_CSP_PLANES[p0->csp]; i++) {
        const auto plane0 = getPlane(p0, (RGY_PLANE)i);
        const auto plane1 = getPlane(p1, (RGY_PLANE)i);
        auto planeTmp = getPlane(tmp, (RGY_PLANE)i);
        auto sts = calc_block_diff_plane<Type4>( &plane0, &plane1, &planeTmp, streamDiff);
        if (sts != RGY_ERR_NONE) {
            return sts;
        }
    }
    return RGY_ERR_NONE;
}

NVEncFilterMpdecimateFrameData::NVEncFilterMpdecimateFrameData(std::shared_ptr<RGYLog> log) :
    m_log(log),
    m_inFrameId(-1),
    m_buf(),
    m_tmp() {

}

NVEncFilterMpdecimateFrameData::~NVEncFilterMpdecimateFrameData() {
    m_buf.clear();
}

RGY_ERR NVEncFilterMpdecimateFrameData::set(const RGYFrameInfo *pInputFrame, int inputFrameId, cudaStream_t stream) {
    m_inFrameId = inputFrameId;
    if (m_buf.frame.ptr == nullptr) {
        m_buf.alloc(pInputFrame->width, pInputFrame->height, pInputFrame->csp);
    }
    if (m_tmp.frameDev.ptr == nullptr) {
        m_tmp.alloc(divCeil(pInputFrame->width, 8), divCeil(pInputFrame->height, 8), RGY_CSP_YUV444_32);
    }

    auto sts = m_buf.copyFrameAsync(pInputFrame, stream);
    if (sts != RGY_ERR_NONE) {
        m_log->write(RGY_LOG_ERROR, RGY_LOGT_VPP, _T("failed to set frame to data cache: %s.\n"), get_err_mes(sts));
        return sts;
    }
    copyFrameProp(&m_buf.frame, pInputFrame);
    return RGY_ERR_NONE;
}

RGY_ERR NVEncFilterMpdecimateFrameData::calcDiff(const NVEncFilterMpdecimateFrameData *ref,
    cudaStream_t streamDiff, cudaEvent_t eventTransfer, cudaStream_t streamTransfer) {
    static const std::map<RGY_CSP, decltype(calc_block_diff_frame<uchar4>)*> func_list = {
        { RGY_CSP_YV12,      calc_block_diff_frame<uchar4>  },
        { RGY_CSP_YV12_16,   calc_block_diff_frame<ushort4> },
        { RGY_CSP_YUV444,    calc_block_diff_frame<uchar4>  },
        { RGY_CSP_YUV444_16, calc_block_diff_frame<ushort4> }
    };
    if (func_list.count(ref->m_buf.frame.csp) == 0) {
        m_log->write(RGY_LOG_ERROR, RGY_LOGT_VPP, _T("unsupported csp %s.\n"), RGY_CSP_NAMES[ref->m_buf.frame.csp]);
        return RGY_ERR_UNSUPPORTED;
    }
    auto sts = func_list.at(ref->m_buf.frame.csp)(&m_buf.frame, &ref->get()->frame, &m_tmp.frameDev, streamDiff);
    if (sts != RGY_ERR_NONE) {
        m_log->write(RGY_LOG_ERROR, RGY_LOGT_VPP, _T("failed to run calcDiff: %s.\n"), get_err_mes(sts));
        return RGY_ERR_CUDA;
    }

    if ((sts = err_to_rgy(cudaEventRecord(eventTransfer, streamDiff))) != RGY_ERR_NONE) {
        m_log->write(RGY_LOG_ERROR, RGY_LOGT_VPP, _T("failed to cudaEventRecord in calcDiff: %s.\n"), get_err_mes(sts));
        return sts;
    }
    if ((sts = err_to_rgy(cudaStreamWaitEvent(streamTransfer, eventTransfer, 0))) != RGY_ERR_NONE) {
        m_log->write(RGY_LOG_ERROR, RGY_LOGT_VPP, _T("failed to cudaStreamWaitEvent in calcDiff: %s.\n"), get_err_mes(sts));
        return sts;
    }
    if ((sts = m_tmp.copyDtoHAsync(streamTransfer)) != RGY_ERR_NONE) {
        m_log->write(RGY_LOG_ERROR, RGY_LOGT_VPP, _T("failed to copyDtoHAsync in calcDiff: %s.\n"), get_err_mes(sts));
        return sts;
    }
    return RGY_ERR_NONE;
}

bool NVEncFilterMpdecimateFrameData::checkIfFrameCanbeDropped(const int hi, const int lo, const float factor) {
    const int threshold = (int)((float)m_tmp.frameHost.width * m_tmp.frameHost.height * factor + 0.5f);
    int loCount = 0;
    for (int iplane = 0; iplane < RGY_CSP_PLANES[m_buf.frame.csp]; iplane++) {
        const auto plane = getPlane(&m_buf.frame, (RGY_PLANE)iplane);
        const int blockw = divCeil(plane.width, 8);
        const int blockh = divCeil(plane.height, 8);
        for (int j = 0; j < blockh; j++) {
            const int *ptrResult = (const int *)(m_tmp.frameHost.ptr + j * m_tmp.frameHost.pitch);
            for (int i = 0; i < blockw; i++) {
                const int result = ptrResult[i];
                if (result > hi) {
                    return false;
                }
                if (result > lo) {
                    loCount++;
                    if (loCount > threshold) {
                        return false;
                    }
                }
            }
        }
    }
    return true;
}

NVEncFilterMpdecimateCache::NVEncFilterMpdecimateCache() : m_inputFrames(0), m_frames() {

}

NVEncFilterMpdecimateCache::~NVEncFilterMpdecimateCache() {
    m_frames.clear();
}

void NVEncFilterMpdecimateCache::init(int bufCount, std::shared_ptr<RGYLog> log) {
    m_log = log;
    m_frames.clear();
    for (int i = 0; i < bufCount; i++) {
        m_frames.push_back(std::make_unique<NVEncFilterMpdecimateFrameData>(log));
    }
}

RGY_ERR NVEncFilterMpdecimateCache::add(const RGYFrameInfo *pInputFrame, cudaStream_t stream) {
    const int id = m_inputFrames++;
    return getEmpty()->set(pInputFrame, id, stream);
}

NVEncFilterMpdecimate::NVEncFilterMpdecimate() : m_dropCount(0), m_ref(-1), m_target(-1), m_cache(), m_eventDiff(), m_streamDiff(), m_streamTransfer() {
    m_sFilterName = _T("mpdecimate");
}

NVEncFilterMpdecimate::~NVEncFilterMpdecimate() {
    close();
}

RGY_ERR NVEncFilterMpdecimate::checkParam(const std::shared_ptr<NVEncFilterParamMpdecimate> prm) {
    if (prm->frameOut.height <= 0 || prm->frameOut.width <= 0) {
        AddMessage(RGY_LOG_ERROR, _T("Invalid frame size.\n"));
        return RGY_ERR_INVALID_PARAM;
    }
    if (prm->mpdecimate.lo <= 0) {
        AddMessage(RGY_LOG_ERROR, _T("\"lo\" must a positive value.\n"));
        return RGY_ERR_INVALID_PARAM;
    }
    if (prm->mpdecimate.hi <= 0) {
        AddMessage(RGY_LOG_ERROR, _T("\"hi\" must a positive value.\n"));
        return RGY_ERR_INVALID_PARAM;
    }
    if (prm->mpdecimate.frac < 0.0) {
        AddMessage(RGY_LOG_ERROR, _T("\"frac\" must a positive value.\n"));
        return RGY_ERR_INVALID_PARAM;
    }
    return RGY_ERR_NONE;
}

RGY_ERR NVEncFilterMpdecimate::init(shared_ptr<NVEncFilterParam> pParam, shared_ptr<RGYLog> pPrintMes) {
    RGY_ERR sts = RGY_ERR_NONE;
    m_pPrintMes = pPrintMes;
    auto prm = std::dynamic_pointer_cast<NVEncFilterParamMpdecimate>(pParam);
    if (!prm) {
        AddMessage(RGY_LOG_ERROR, _T("Invalid parameter type.\n"));
        return RGY_ERR_INVALID_PARAM;
    }
    //パラメータチェック
    if ((sts = checkParam(prm)) != RGY_ERR_NONE) {
        return sts;
    }

    if (!m_pParam || std::dynamic_pointer_cast<NVEncFilterParamMpdecimate>(m_pParam)->mpdecimate != prm->mpdecimate) {

        m_cache.init(2, m_pPrintMes);

        m_eventDiff = std::unique_ptr<cudaEvent_t, cudaevent_deleter>(new cudaEvent_t(), cudaevent_deleter());
        if (RGY_ERR_NONE != (sts = err_to_rgy(cudaEventCreateWithFlags(m_eventDiff.get(), cudaEventDisableTiming)))) {
            AddMessage(RGY_LOG_ERROR, _T("failed to cudaEventCreateWithFlags: %s.\n"), get_err_mes(sts));
            return sts;
        }
        AddMessage(RGY_LOG_DEBUG, _T("cudaEventCreateWithFlags for m_eventDiff: Success.\n"));

        m_eventTransfer = std::unique_ptr<cudaEvent_t, cudaevent_deleter>(new cudaEvent_t(), cudaevent_deleter());
        if (RGY_ERR_NONE != (sts = err_to_rgy(cudaEventCreateWithFlags(m_eventTransfer.get(), cudaEventDisableTiming)))) {
            AddMessage(RGY_LOG_ERROR, _T("failed to cudaEventCreateWithFlags: %s.\n"), get_err_mes(sts));
            return sts;
        }
        AddMessage(RGY_LOG_DEBUG, _T("cudaEventCreateWithFlags for m_eventTransfer: Success.\n"));

        m_streamDiff = std::unique_ptr<cudaStream_t, cudastream_deleter>(new cudaStream_t(), cudastream_deleter());
        if (RGY_ERR_NONE != (sts = err_to_rgy(cudaStreamCreateWithFlags(m_streamDiff.get(), 0/*cudaStreamNonBlocking*/)))) {
            AddMessage(RGY_LOG_ERROR, _T("failed to cudaStreamCreateWithFlags: %s.\n"), get_err_mes(sts));
            return sts;
        }
        AddMessage(RGY_LOG_DEBUG, _T("cudaStreamCreateWithFlags for m_streamDiff: Success.\n"));

        m_streamTransfer = std::unique_ptr<cudaStream_t, cudastream_deleter>(new cudaStream_t(), cudastream_deleter());
        if (RGY_ERR_NONE != (sts = err_to_rgy(cudaStreamCreateWithFlags(m_streamTransfer.get(), 0/*cudaStreamNonBlocking*/)))) {
            AddMessage(RGY_LOG_ERROR, _T("failed to cudaStreamCreateWithFlags: %s.\n"), get_err_mes(sts));
            return sts;
        }
        AddMessage(RGY_LOG_DEBUG, _T("cudaStreamCreateWithFlags for m_streamTransfer: Success.\n"));

        prm->frameOut.pitch = prm->frameIn.pitch;

        m_fpLog.reset();
        if (prm->mpdecimate.log) {
            const tstring logfilename = prm->outfilename + _T(".mpdecimate.log.txt");
            m_fpLog = std::unique_ptr<FILE, fp_deleter>(_tfopen(logfilename.c_str(), _T("w")), fp_deleter());
            AddMessage(RGY_LOG_DEBUG, _T("Opened log file: %s.\n"), logfilename.c_str());
        }

        const int max_value = (1 << RGY_CSP_BIT_DEPTH[prm->frameIn.csp]) - 1;
        m_nPathThrough &= (~(FILTER_PATHTHROUGH_TIMESTAMP));
        m_dropCount = 0;
        m_ref = -1;
        m_target = -1;

        setFilterInfo(pParam->print());
    }
    m_pParam = pParam;
    return sts;
}

tstring NVEncFilterParamMpdecimate::print() const {
    return mpdecimate.print();
}

bool NVEncFilterMpdecimate::dropFrame(NVEncFilterMpdecimateFrameData *targetFrame) {
    auto prm = std::dynamic_pointer_cast<NVEncFilterParamMpdecimate>(m_pParam);
    if (!prm) {
        AddMessage(RGY_LOG_ERROR, _T("Invalid parameter type.\n"));
        return false;
    }
    if (prm->mpdecimate.max > 0 &&
        m_dropCount >= prm->mpdecimate.max) {
        return false;
    }
    if (prm->mpdecimate.max < 0 &&
        (m_dropCount - 1) > prm->mpdecimate.max) {
        return false;
    }
    const int bit_depth = RGY_CSP_BIT_DEPTH[targetFrame->get()->frame.csp];
    return targetFrame->checkIfFrameCanbeDropped(prm->mpdecimate.hi << (bit_depth - 8), prm->mpdecimate.lo << (bit_depth - 8), prm->mpdecimate.frac);
}

RGY_ERR NVEncFilterMpdecimate::run_filter(const RGYFrameInfo *pInputFrame, RGYFrameInfo **ppOutputFrames, int *pOutputFrameNum, cudaStream_t stream) {
    RGY_ERR sts = RGY_ERR_NONE;
    auto prm = std::dynamic_pointer_cast<NVEncFilterParamMpdecimate>(m_pParam);
    if (!prm) {
        AddMessage(RGY_LOG_ERROR, _T("Invalid parameter type.\n"));
        return RGY_ERR_INVALID_PARAM;
    }
    if (pInputFrame->ptr == nullptr && m_ref < 0) {
        //終了
        *pOutputFrameNum = 0;
        ppOutputFrames[0] = nullptr;
        return sts;
    }
    if (m_ref < 0) {
        m_ref = m_cache.inframe();
        auto err = m_cache.add(pInputFrame, stream);
        if (err != RGY_ERR_NONE) {
            AddMessage(RGY_LOG_ERROR, _T("failed to add frame to cache: %s.\n"), get_err_mes(err));
            return err;
        }
        *pOutputFrameNum = 1;
        ppOutputFrames[0] = &m_cache.get(m_ref)->frame;
        if (m_fpLog) {
            fprintf(m_fpLog.get(), "  %8d: %10lld\n", m_ref, (long long)ppOutputFrames[0]->timestamp);
        }
        return sts;
    }
    if (m_target >= 0) {
        auto targetFrame = m_cache.frame(m_target);
        //GPU->CPUの転送終了を待機
        cudaStreamSynchronize(*m_streamTransfer.get());

        const bool drop = dropFrame(targetFrame) && pInputFrame->ptr != nullptr; //最終フレームは必ず出力する
        if (m_fpLog) {
            fprintf(m_fpLog.get(), "%s %8d: %10lld\n", (drop) ? "d" : " ", m_target, (long long)targetFrame->get()->frame.timestamp);
        }
        if (drop) {
            targetFrame->reset();
            m_target = -1;
            m_dropCount = std::max(1, m_dropCount + 1);
            *pOutputFrameNum = 0;
            ppOutputFrames[0] = nullptr;
        } else {
            m_dropCount = std::min(-1, m_dropCount - 1);
            m_cache.frame(m_ref)->reset();
            m_ref = m_target;
            m_target = -1;
            *pOutputFrameNum = 1;
            ppOutputFrames[0] = &targetFrame->get()->frame;
        }
    }
    if (pInputFrame->ptr != nullptr) {
        m_target = m_cache.inframe();
        auto err = m_cache.add(pInputFrame, stream);
        if (err != RGY_ERR_NONE) {
            AddMessage(RGY_LOG_ERROR, _T("failed to add frame to cache: %s.\n"), get_err_mes(err));
            return err;
        }
        cudaEventRecord(*m_eventDiff.get(), stream);
        cudaStreamWaitEvent(*m_streamDiff.get(), *m_eventDiff.get(), 0);
        err = m_cache.frame(m_target)->calcDiff(m_cache.frame(m_ref), *m_streamDiff.get(), *m_eventTransfer.get(), *m_streamTransfer.get());
        if (err != RGY_ERR_NONE) {
            AddMessage(RGY_LOG_ERROR, _T("failed to run calcDiff: %s.\n"), get_err_mes(err));
            return err;
        }
    }
    return RGY_ERR_NONE;
}

void NVEncFilterMpdecimate::close() {
    m_pFrameBuf.clear();
    m_eventDiff.reset();
    m_streamDiff.reset();
    m_streamTransfer.reset();
    m_fpLog.reset();
}
