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

#include <array>
#include <numeric>
#include "convert_csp.h"
#include "NVEncFilter.h"
#include "NVEncFilterParam.h"
#include "NVEncFilterNVOFFRUC.h"

NVEncNVOFFRUCFuncs::NVEncNVOFFRUCFuncs() :
    hModule(),
    fcreate(nullptr),
    fload(nullptr),
    fdelete(nullptr),
    fcreateHandle(nullptr),
    fregisterResource(nullptr),
    fcloseHandle(nullptr),
    fproc(nullptr) {
}
NVEncNVOFFRUCFuncs::~NVEncNVOFFRUCFuncs() {
    close();
}

RGY_ERR NVEncNVOFFRUCFuncs::load() {
    hModule = RGY_LOAD_LIBRARY(NVENC_NVOFFRUC_MODULENAME);
    if (!hModule) {
        return RGY_ERR_NULL_PTR;
    }

#define LOAD_PROC(proc, procName) { \
    proc = (decltype(procName) *)RGY_GET_PROC_ADDRESS(hModule, #procName); \
    if (!proc) { \
        close(); \
        return RGY_ERR_NULL_PTR; \
    } \
}

    LOAD_PROC(fcreate,           NVEncNVOFFRUCCreate);
    LOAD_PROC(fload,             NVEncNVOFFRUCLoad);
    LOAD_PROC(fdelete,           NVEncNVOFFRUCDelete);
    LOAD_PROC(fcreateHandle,     NVEncNVOFFRUCCreateFURCHandle);
    LOAD_PROC(fregisterResource, NVEncNVOFFRUCRegisterResource);
    LOAD_PROC(fcloseHandle,      NVEncNVOFFRUCCloseFURCHandle);
    LOAD_PROC(fproc,             NVEncNVOFFRUCProc);
#undef LOAD_PROC
    return RGY_ERR_NONE;
}
void NVEncNVOFFRUCFuncs::close() {
    if (hModule) {
        RGY_FREE_LIBRARY(hModule);
        hModule = nullptr;
    }
}

tstring NVEncFilterParamNVOFFRUC::print() const {
    return fruc.print();
}

NVEncFilterNVOFFRUC::NVEncFilterNVOFFRUC() :
    m_func(),
    m_frucBuf(),
    m_frucHandles(),
    m_frucCsp(RGY_CSP_NV12),
    m_prevTimestamp(-1),
    m_targetFps(),
    m_timebase(),
    m_srcCrop(),
    m_dstCrop(),
    m_inputFrames(0) {
    m_name = _T("nvof-fruc");
}
NVEncFilterNVOFFRUC::~NVEncFilterNVOFFRUC() {
    close();
}

RGY_ERR NVEncFilterNVOFFRUC::checkParam([[maybe_unused]] const NVEncFilterParam *param) {
    return RGY_ERR_NONE;
}

RGY_ERR NVEncFilterNVOFFRUC::init(shared_ptr<NVEncFilterParam> pParam, shared_ptr<RGYLog> pPrintMes) {
    RGY_ERR sts = RGY_ERR_NONE;
    m_pLog = pPrintMes;
#if !ENABLE_NVOFFRUC
    AddMessage(RGY_LOG_ERROR, _T("nv optical flow filters are not supported on x86 exec file, please use x64 exec file.\n"));
    return RGY_ERR_UNSUPPORTED;
#else
    auto prm = dynamic_cast<NVEncFilterParamNVOFFRUC*>(pParam.get());
    if (!prm) {
        AddMessage(RGY_LOG_ERROR, _T("Invalid parameter type.\n"));
        return RGY_ERR_INVALID_PARAM;
    }
    if (prm->compute_capability.first < 7) {
        AddMessage(RGY_LOG_ERROR, _T("NVOF FRUC filters require Turing GPUs (CC:7.0) or later: current CC %d.%d.\n"), prm->compute_capability.first, prm->compute_capability.second);
        return RGY_ERR_UNSUPPORTED;
    }
    AddMessage(RGY_LOG_DEBUG, _T("GPU CC: %d.%d.\n"),
        prm->compute_capability.first, prm->compute_capability.second);

    if (rgy_csp_has_alpha(pParam->frameIn.csp)) {
        AddMessage(RGY_LOG_ERROR, _T("vpp-fruc does not support alpha channel.\n"));
        return RGY_ERR_UNSUPPORTED;
    }

    if (!m_func) {
        m_func = std::make_unique<NVEncNVOFFRUCFuncs>();
        if ((sts = m_func->load()) != RGY_ERR_NONE) {
            AddMessage(RGY_LOG_ERROR, _T("Cannot find NVOF FRUC library %s: %s.\n"), NVENC_NVOFFRUC_MODULENAME, get_err_mes(sts));
            return RGY_ERR_NULL_PTR;
        }
        AddMessage(RGY_LOG_DEBUG, _T("Loaded NVOF FRUC library %s.\n"), NVENC_NVOFFRUC_MODULENAME);
    }
    if (m_frucHandles.size() == 0) {
        NVEncNVOFFRUCHandle frucHandle = nullptr;
        if ((sts = m_func->fcreate(&frucHandle)) != RGY_ERR_NONE) {
            AddMessage(RGY_LOG_ERROR, _T("Failed to create FRUC handle: %s.\n"), get_err_mes(sts));
            return sts;
        }

        if ((sts = m_func->fload(frucHandle)) != RGY_ERR_NONE) {
            AddMessage(RGY_LOG_ERROR, _T("Failed to load NVOF FRUC library %s: %s.\n"), NVOFFRUC_MODULENAME, get_err_mes(sts));
            return sts;
        }
        AddMessage(RGY_LOG_DEBUG, _T("Loaded NVOF FRUC library %s: %s.\n"), NVOFFRUC_MODULENAME, get_err_mes(sts));
        m_frucHandles.push_back(NVEncFilterFRUCHandle(unique_fruc_handle(frucHandle, m_func->fcloseHandle)));
    }

    sts = checkParam(pParam.get());
    if (sts != RGY_ERR_NONE) {
        return sts;
    }
    m_frucCsp = (RGY_CSP_CHROMA_FORMAT[pParam->frameIn.csp] == RGY_CHROMAFMT_YUV444) ? RGY_CSP_RGB32 : RGY_CSP_NV12;
    if (!m_srcCrop
        || m_srcCrop->GetFilterParam()->frameIn.width  != pParam->frameIn.width
        || m_srcCrop->GetFilterParam()->frameIn.height != pParam->frameIn.height) {
        AddMessage(RGY_LOG_DEBUG, _T("Create input csp conversion filter.\n"));
        unique_ptr<NVEncFilterCspCrop> filter(new NVEncFilterCspCrop());
        shared_ptr<NVEncFilterParamCrop> paramCrop(new NVEncFilterParamCrop());
        paramCrop->frameIn = pParam->frameIn;
        paramCrop->frameOut = paramCrop->frameIn;
        paramCrop->frameOut.csp = m_frucCsp;
        paramCrop->baseFps = pParam->baseFps;
        paramCrop->frameIn.mem_type = RGY_MEM_TYPE_GPU;
        paramCrop->frameOut.mem_type = RGY_MEM_TYPE_GPU;
        paramCrop->bOutOverwrite = false;
        sts = filter->init(paramCrop, m_pLog);
        if (sts != RGY_ERR_NONE) {
            return sts;
        }
        m_srcCrop = std::move(filter);
        AddMessage(RGY_LOG_DEBUG, _T("created %s.\n"), m_srcCrop->GetInputMessage().c_str());
    }
    if (!m_dstCrop
        || m_dstCrop->GetFilterParam()->frameOut.width  != pParam->frameOut.width
        || m_dstCrop->GetFilterParam()->frameOut.height != pParam->frameOut.height) {
        AddMessage(RGY_LOG_DEBUG, _T("Create output csp conversion filter.\n"));
        unique_ptr<NVEncFilterCspCrop> filter(new NVEncFilterCspCrop());
        shared_ptr<NVEncFilterParamCrop> paramCrop(new NVEncFilterParamCrop());
        paramCrop->frameIn = pParam->frameOut;
        paramCrop->frameIn.csp = m_frucCsp;
        paramCrop->frameOut = pParam->frameOut;
        paramCrop->baseFps = pParam->baseFps;
        paramCrop->frameIn.mem_type = RGY_MEM_TYPE_GPU;
        paramCrop->frameOut.mem_type = RGY_MEM_TYPE_GPU;
        paramCrop->bOutOverwrite = false;
        sts = filter->init(paramCrop, m_pLog);
        if (sts != RGY_ERR_NONE) {
            return sts;
        }
        m_dstCrop = std::move(filter);
        AddMessage(RGY_LOG_DEBUG, _T("created %s.\n"), m_dstCrop->GetInputMessage().c_str());
    }

    if (!m_frucBuf.front()
        || m_frucBuf.front()->width() != pParam->frameIn.width
        || m_frucBuf.front()->height() != pParam->frameIn.height
        || m_frucBuf.front()->csp() != m_srcCrop->GetFilterParam()->frameOut.csp) {
        for (auto& buf : m_frucBuf) {
            buf = std::make_unique<CUFrameDevPtr>(pParam->frameIn.width, pParam->frameIn.height, m_srcCrop->GetFilterParam()->frameOut.csp);
            // NVOF FRUCに渡すフレームは連続確保かつpitch=widthが必要
            if ((sts = buf->alloc(true, 1)) != RGY_ERR_NONE) {
                AddMessage(RGY_LOG_ERROR, _T("failed to allocate memory: %s.\n"), get_err_mes(sts));
                return sts;
            }
        }
    }
    switch (prm->fruc.mode) {
    case VppFrucMode::NVOFFRUCx2:
        m_targetFps = pParam->baseFps * 2;
        break;
    case VppFrucMode::NVOFFRUCFps:
        m_targetFps = prm->fruc.targetFps;
        break;
    default:
        AddMessage(RGY_LOG_ERROR, _T("Invalid fruc mode: %d.\n"), prm->fruc.mode);
        return RGY_ERR_INVALID_PARAM;
    }
    m_timebase = prm->timebase;
    if (m_frameBuf.size() == 0
        || !cmpFrameInfoCspResolution(&m_frameBuf[0]->frame, &pParam->frameOut)) {
        sts = AllocFrameBuf(pParam->frameOut, 1);
        if (sts != RGY_ERR_NONE) {
            AddMessage(RGY_LOG_ERROR, _T("failed to allocate memory: %s.\n"), get_err_mes(sts));
            return RGY_ERR_MEMORY_ALLOC;
        }
    }
    for (int i = 0; i < RGY_CSP_PLANES[pParam->frameOut.csp]; i++) {
        pParam->frameOut.pitch[i] = m_frameBuf[0]->frame.pitch[i];
    }
    tstring info = m_name + _T(": ");
    if (m_srcCrop) {
        info += m_srcCrop->GetInputMessage() + _T("\n");
    }
    tstring nameBlank(m_name.length() + _tcslen(_T(": ")), _T(' '));
    info += tstring(INFO_INDENT) + nameBlank + pParam->print();
    if (m_dstCrop) {
        info += tstring(_T("\n")) + tstring(INFO_INDENT) + nameBlank + m_dstCrop->GetInputMessage();
    }
    setFilterInfo(info);
    pParam->baseFps = m_targetFps;
    m_pathThrough &= (~(FILTER_PATHTHROUGH_TIMESTAMP));
    m_param = pParam;
    return sts;
#endif
}

std::pair<RGY_ERR, unique_fruc_handle> NVEncFilterNVOFFRUC::createFRUCHandle() {
    NVEncNVOFFRUCHandle frucHandle = nullptr;
    auto sts = m_func->fcreate(&frucHandle);
    if (sts != RGY_ERR_NONE) {
        AddMessage(RGY_LOG_ERROR, _T("Failed to create FRUC handle: %s.\n"), get_err_mes(sts));
        return std::make_pair(sts, unique_fruc_handle(nullptr, nullptr));
    }
    return std::make_pair(RGY_ERR_NONE, unique_fruc_handle(frucHandle, m_func->fcloseHandle));
}

RGYFrameInfo *NVEncFilterNVOFFRUC::getNextOutFrame(RGYFrameInfo **ppOutputFrames, int *pOutputFrameNum) {
    if (*pOutputFrameNum >= (int)m_frameBuf.size()) {
        auto uptr = std::make_unique<CUFrameBuf>(m_frameBuf.front()->frame.width, m_frameBuf.front()->frame.height, m_frameBuf.front()->frame.csp);
        auto ret = uptr->alloc();
        if (ret != RGY_ERR_NONE) {
            m_frameBuf.clear();
            return nullptr;
        }
        m_frameBuf.push_back(std::move(uptr));
    }
    int outFrameIdx = pOutputFrameNum[0]++;
    auto ptr = &m_frameBuf[outFrameIdx]->frame;
    ppOutputFrames[outFrameIdx] = ptr;
    return ptr;
}

RGY_ERR NVEncFilterNVOFFRUC::run_filter(const RGYFrameInfo *pInputFrame, RGYFrameInfo **ppOutputFrames, int *pOutputFrameNum, cudaStream_t stream) {
    RGY_ERR sts = RGY_ERR_NONE;
    if (pInputFrame->ptr[0] == nullptr) {
        return sts;
    }

    //const auto memcpyKind = getCudaMemcpyKind(pInputFrame->mem_type, ppOutputFrames[0]->mem_type);
    //if (memcpyKind != cudaMemcpyDeviceToDevice) {
    //    AddMessage(RGY_LOG_ERROR, _T("only supported on device memory.\n"));
    //    return RGY_ERR_INVALID_PARAM;
    //}
    if (m_param->frameOut.csp != m_param->frameIn.csp) {
        AddMessage(RGY_LOG_ERROR, _T("csp does not match.\n"));
        return RGY_ERR_INVALID_PARAM;
    }

    auto bufferIn   = m_frucBuf[m_inputFrames % 2].get();
    auto bufferPrev = m_frucBuf[std::abs(m_inputFrames - 1) % 2].get();

    int cropFilterOutputNum = 0;
    RGYFrameInfo *outInfo[1] = { &bufferIn->frame };
    RGYFrameInfo cropInput = *pInputFrame;
    auto sts_filter = m_srcCrop->filter(&cropInput, (RGYFrameInfo **)&outInfo, &cropFilterOutputNum, stream);
    if (outInfo[0] == nullptr || cropFilterOutputNum != 1) {
        AddMessage(RGY_LOG_ERROR, _T("Unknown behavior \"%s\".\n"), m_srcCrop->name().c_str());
        return sts_filter;
    }
    if (sts_filter != RGY_ERR_NONE || cropFilterOutputNum != 1) {
        AddMessage(RGY_LOG_ERROR, _T("Error while running filter \"%s\".\n"), m_srcCrop->name().c_str());
        return sts_filter;
    }
    copyFramePropWithoutRes(&bufferIn->frame, pInputFrame);

    *pOutputFrameNum = 0;
    if (m_inputFrames++ == 0) {
        // 出力するフレームは自分自身のみ
        auto outFrame = getNextOutFrame(ppOutputFrames, pOutputFrameNum);
        sts = copyFrameAsync(outFrame, pInputFrame, stream);
        if (sts != RGY_ERR_NONE) {
            AddMessage(RGY_LOG_ERROR, _T("Failed to copy frame: %s.\n"), get_err_mes(sts));
            return sts;
        }
        copyFramePropWithoutRes(outFrame, pInputFrame);
        for (auto& frucHandle : m_frucHandles) {
            sts = setFirstFrame(frucHandle, bufferIn);
            if (sts != RGY_ERR_NONE) {
                return sts;
            }
        }
        m_prevTimestamp = pInputFrame->timestamp;
        return RGY_ERR_NONE;
    }

    auto prm = dynamic_cast<NVEncFilterParamNVOFFRUC*>(m_param.get());
    if (!prm) {
        AddMessage(RGY_LOG_ERROR, _T("Invalid parameter type.\n"));
        return RGY_ERR_INVALID_PARAM;
    }

    size_t frucHandleIdx = 0;
    auto prevTimestamp = m_prevTimestamp;
    for (int i = 1; ; i++) {
        int64_t nextPts = -1;
        int64_t duration = 0;
        bool isNearToCurrentFrame = false;
        if (prm->fruc.mode == VppFrucMode::NVOFFRUCx2) {
            nextPts = m_prevTimestamp + i * (pInputFrame->timestamp - m_prevTimestamp) / 2;
            duration = nextPts - prevTimestamp;
            isNearToCurrentFrame = (i >= 2);
        } else if (prm->fruc.mode == VppFrucMode::NVOFFRUCFps) {
            const auto frameOffset = m_timebase.inv() / m_targetFps * i;
            nextPts = m_prevTimestamp + (frameOffset.d() == 1 ? frameOffset.n() : (int64_t)(frameOffset.qdouble() + 0.5));
            duration = nextPts - prevTimestamp;
            const int64_t timestampDiff = nextPts - (int64_t)bufferIn->timestamp();
            isNearToCurrentFrame = std::abs(timestampDiff) <= std::max<int64_t>((int64_t)((m_timebase.inv() / m_targetFps * rgy_rational<int>(1, 8)).qdouble() + 0.5), 1);
        } else {
            AddMessage(RGY_LOG_ERROR, _T("Invalid fruc mode: %d.\n"), prm->fruc.mode);
            return RGY_ERR_INVALID_PARAM;
        }
        // 入力フレームよりも後のフレームを生成する場合は終了
        if (!isNearToCurrentFrame && nextPts > (int64_t)bufferIn->timestamp()) {
            break;
        }
        auto outFrame = getNextOutFrame(ppOutputFrames, pOutputFrameNum);
        copyFramePropWithoutRes(outFrame, pInputFrame);
        if (isNearToCurrentFrame) {
            sts = copyFrameAsync(outFrame, pInputFrame, stream);
            if (sts != RGY_ERR_NONE) {
                AddMessage(RGY_LOG_ERROR, _T("Failed to copy frame: %s.\n"), get_err_mes(sts));
                return sts;
            }
            outFrame->timestamp = nextPts;
            outFrame->duration = duration;
            prevTimestamp = nextPts;
            break;
        }
        // 途中のフレームを生成
        sts = genFrame(frucHandleIdx++, outFrame, bufferPrev, bufferIn, nextPts, stream);
        if (sts != RGY_ERR_NONE) {
            return sts;
        }
        outFrame->timestamp = nextPts;
        outFrame->duration = duration;
        prevTimestamp = nextPts;
    }
    m_prevTimestamp = prevTimestamp;
    return RGY_ERR_NONE;
}

RGY_ERR NVEncFilterNVOFFRUC::setFirstFrame(NVEncFilterFRUCHandle& frucHandle, const CUFrameDevPtr *firstframe) {
    CUFrameDevPtr *work = m_frucBuf.back().get();
    if (firstframe) {
        auto sts = m_func->fcreateHandle(frucHandle.handle.get(), firstframe->width(), firstframe->height(), m_frucCsp == RGY_CSP_NV12);
        if (sts != RGY_ERR_NONE) {
            AddMessage(RGY_LOG_ERROR, _T("Failed to create fruc handle: %s.\n"), get_err_mes(sts));
            return sts;
        }

        // NVOF FRUCにデバイスポインタを渡すときは、デバイスポインタそのものでなく、
        // デバイスポインタのあるアドレスを指定する必要がある
        sts = m_func->fregisterResource(frucHandle.handle.get(), &m_frucBuf[0]->frame.ptr[0], &m_frucBuf[1]->frame.ptr[0], &m_frucBuf[2]->frame.ptr[0]);
        if (sts != RGY_ERR_NONE) {
            AddMessage(RGY_LOG_ERROR, _T("Failed to register resources: %s.\n"), get_err_mes(sts));
            return sts;
        }

        // NVOF FRUCにデバイスポインタを渡すときは、デバイスポインタそのものでなく、
        // デバイスポインタのあるアドレスを指定する必要がある
        NVEncNVOFFRUCParams procParam = { 0 };
        procParam.frameIn = (void *)&firstframe->frame.ptr[0];
        procParam.timestampIn = firstframe->timestamp();
        procParam.frameOut = (void *)&work->frame.ptr[0];
        procParam.timestampOut = (frucHandle.prevFramePts >= 0) ? ((frucHandle.prevFramePts + firstframe->timestamp()) / 2) : work->timestamp();
        sts = m_func->fproc(frucHandle.handle.get(), &procParam);
        if (sts != RGY_ERR_NONE) {
            AddMessage(RGY_LOG_ERROR, _T("FRUC: Process(0) %lld: %s.\n"),
                firstframe->timestamp(), get_err_mes(sts));
            return sts;
        }
        frucHandle.prevFramePts = firstframe->timestamp();
    }
    return RGY_ERR_NONE;
}

RGY_ERR NVEncFilterNVOFFRUC::genFrame(const size_t frucHandleIdx, RGYFrameInfo *outFrame, const CUFrameDevPtr *prev, const CUFrameDevPtr *curr, const int64_t genPts, cudaStream_t stream) {
    CUFrameDevPtr *work = m_frucBuf.back().get();
    copyFramePropWithoutRes(&work->frame, &curr->frame);
    work->setTimestamp(genPts);
    work->setInputFrameId(curr->inputFrameId());
    if (frucHandleIdx > m_frucHandles.size()) {
        return RGY_ERR_UNKNOWN;
    } else if (frucHandleIdx == m_frucHandles.size()) {
        auto sts = RGY_ERR_NONE;
        NVEncNVOFFRUCHandle frucHandle = nullptr;
        if ((sts = m_func->fcreate(&frucHandle)) != RGY_ERR_NONE) {
            AddMessage(RGY_LOG_ERROR, _T("Failed to create FRUC handle: %s.\n"), get_err_mes(sts));
            return sts;
        }

        if ((sts = m_func->fload(frucHandle)) != RGY_ERR_NONE) {
            AddMessage(RGY_LOG_ERROR, _T("Failed to load NVOF FRUC library %s: %s.\n"), NVOFFRUC_MODULENAME, get_err_mes(sts));
            return sts;
        }
        m_frucHandles.push_back(NVEncFilterFRUCHandle(unique_fruc_handle(frucHandle, m_func->fcloseHandle)));
        AddMessage(RGY_LOG_DEBUG, _T("Loaded NVOF FRUC library %s: %s.\n"), NVOFFRUC_MODULENAME, get_err_mes(sts));
    }
    auto& frucHandle = m_frucHandles[frucHandleIdx];
    if (frucHandle.prevFramePts < (int64_t)prev->timestamp()) {
        auto sts = setFirstFrame(frucHandle, prev);
        if (sts != RGY_ERR_NONE) {
            return sts;
        }
    }
    {
        // NVOF FRUCにデバイスポインタを渡すときは、デバイスポインタそのものでなく、
        // 「デバイスポインタのあるアドレス」を指定する必要がある
        NVEncNVOFFRUCParams procParam = { 0 };
        procParam.frameIn = (void *)&curr->frame.ptr[0];
        procParam.timestampIn = curr->timestamp();
        procParam.frameOut = (void *)&work->frame.ptr[0];
        procParam.timestampOut = work->timestamp();
        auto sts = m_func->fproc(frucHandle.handle.get(), &procParam);
        if (sts != RGY_ERR_NONE) {
            AddMessage(RGY_LOG_ERROR, _T("FRUC: Process(1) error %lld - %lld -> %lld: %s.\n"),
                prev->timestamp(), curr->timestamp(), work->timestamp(), get_err_mes(sts));
            return sts;
        }
        frucHandle.prevFramePts = curr->timestamp();
    }
    {
        int outputFrames = 0;
        RGYFrameInfo *outInfo[1] = { outFrame };
        auto sts_filter = m_dstCrop->filter(&work->frame, outInfo, &outputFrames, stream);
        if (outInfo[0] == nullptr || outputFrames != 1) {
            AddMessage(RGY_LOG_ERROR, _T("Unknown behavior \"%s\".\n"), m_dstCrop->name().c_str());
            return sts_filter;
        }
        if (sts_filter != RGY_ERR_NONE || outputFrames != 1) {
            AddMessage(RGY_LOG_ERROR, _T("Error while running filter \"%s\".\n"), m_dstCrop->name().c_str());
            return sts_filter;
        }
        copyFramePropWithoutRes(outFrame, &work->frame);
    }
    return RGY_ERR_NONE;
}

void NVEncFilterNVOFFRUC::close() {
    m_srcCrop.reset();
    m_dstCrop.reset();
    for (auto& buf : m_frucBuf) {
        buf.reset();
    }
    m_frucHandles.clear();
    m_func.reset();
    m_frameBuf.clear();
}
