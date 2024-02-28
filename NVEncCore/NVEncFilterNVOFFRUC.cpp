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

#if defined(_WIN32) || defined(_WIN64)
static const TCHAR * NVOFFRUC_MODULENAME = _T("NVEncNVOFFRUC.dll");
#else
static const TCHAR * NVOFFRUC_MODULENAME = _T("libNVEncNVOFFRUC.so");
#endif

NVEncNVOFFRUCFuncs::NVEncNVOFFRUCFuncs() :
    hModule(),
    fcreate(nullptr),
    fdelete(nullptr),
    fcreateHandle(nullptr),
    fcloseHandle(nullptr),
    fproc(nullptr) {
}
NVEncNVOFFRUCFuncs::~NVEncNVOFFRUCFuncs() {
    close();
}

RGY_ERR NVEncNVOFFRUCFuncs::load() {
    hModule = RGY_LOAD_LIBRARY(_T("NVEncNVOFFRUC.dll"));
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

    LOAD_PROC(fcreate,  NVEncNVOptFlowCreate);
    LOAD_PROC(fdelete,  NVEncNVOptFlowDelete);
    LOAD_PROC(fcreateHandle,  NVEncNVOptFlowCreateFURCHandle);
    LOAD_PROC(fcloseHandle,  NVEncNVOptFlowCloseFURCHandle);
    LOAD_PROC(fproc,  NVEncNVOptFlowProc);
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
    m_frucHandle(unique_fruc_handle(nullptr, nullptr)),
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

    if (!m_func) {
        m_func = std::make_unique<NVEncNVOFFRUCFuncs>();
        if (m_func->load() != RGY_ERR_NONE) {
            AddMessage(RGY_LOG_ERROR, _T("Cannot find NVOF FRUC library: %s.\n"), NVOFFRUC_MODULENAME);
            return RGY_ERR_NULL_PTR;
        }
        AddMessage(RGY_LOG_DEBUG, _T("Loaded NVOF FRUC library.\n"));

        NVEncNVOFFRUCHandle frucHandle = nullptr;
        sts = m_func->fcreate(&frucHandle);
        if (sts != RGY_ERR_NONE) {
            AddMessage(RGY_LOG_ERROR, _T("Failed to create FRUC handle: %s.\n"), get_err_mes(sts));
            return sts;
        }
        m_frucHandle = unique_fruc_handle(frucHandle, m_func->fcloseHandle);
    }

    sts = checkParam(pParam.get());
    if (sts != RGY_ERR_NONE) {
        return sts;
    }
    if (!m_srcCrop
        || m_srcCrop->GetFilterParam()->frameIn.width  != pParam->frameIn.width
        || m_srcCrop->GetFilterParam()->frameIn.height != pParam->frameIn.height) {
        AddMessage(RGY_LOG_DEBUG, _T("Create input csp conversion filter.\n"));
        unique_ptr<NVEncFilterCspCrop> filter(new NVEncFilterCspCrop());
        shared_ptr<NVEncFilterParamCrop> paramCrop(new NVEncFilterParamCrop());
        paramCrop->frameIn = pParam->frameIn;
        paramCrop->frameOut = paramCrop->frameIn;
        paramCrop->frameOut.csp = RGY_CSP_YUVA444;
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
        paramCrop->frameIn.csp = RGY_CSP_YUVA444;
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
            buf = std::make_unique<CUFrameBuf>(pParam->frameIn.width, pParam->frameIn.height, m_srcCrop->GetFilterParam()->frameOut.csp);
            if ((sts = buf->alloc(true, 1)) != RGY_ERR_NONE) {
                AddMessage(RGY_LOG_ERROR, _T("failed to allocate memory: %s.\n"), get_err_mes(sts));
                return sts;
            }
        }
    }
    m_targetFps = prm->fruc.targetFps;
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
    setFilterInfo(pParam->print());
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
    copyFrameProp(&bufferIn->frame, pInputFrame);

    *pOutputFrameNum = 0;
    if (m_inputFrames++ == 0) {
        // 出力するフレームは自分自身のみ
        auto outFrame = getNextOutFrame(ppOutputFrames, pOutputFrameNum);
        sts = copyFrameAsync(outFrame, pInputFrame, stream);
        if (sts != RGY_ERR_NONE) {
            AddMessage(RGY_LOG_ERROR, _T("Failed to copy frame: %s.\n"), get_err_mes(sts));
            return sts;
        }
        copyFrameProp(outFrame, pInputFrame);
        m_prevTimestamp = pInputFrame->timestamp;
        return RGY_ERR_NONE;
    }

    sts = m_func->fcreateHandle(m_frucHandle.get(), bufferIn->frame.width, bufferIn->frame.height);
    if (sts != RGY_ERR_NONE) {
        AddMessage(RGY_LOG_ERROR, _T("Failed to create fruc handle: %s.\n"), get_err_mes(sts));
        return sts;
    }

    auto prevTimestamp = m_prevTimestamp;
    for (int i = 0; ; i++) {
        const auto frameOffset = m_targetFps * m_timebase * i;
        const auto nextPts = m_prevTimestamp + frameOffset.d() == 1 ? frameOffset.n() : (int64_t)(frameOffset.qdouble() + 0.5);
        const auto duration = nextPts - prevTimestamp;
        const int64_t timestampDiff = nextPts - (int64_t)bufferIn->timestamp();
        const auto isNearToCurrentFrame = std::abs(timestampDiff) < (int64_t)((m_targetFps * m_timebase * rgy_rational<int>(1, 8)).qdouble() + 0.5);
        if (!isNearToCurrentFrame && nextPts > bufferIn->timestamp()) {
            break;
        }
        auto outFrame = getNextOutFrame(ppOutputFrames, pOutputFrameNum);
        copyFrameProp(outFrame, pInputFrame);
        if (isNearToCurrentFrame) {
            sts = copyFrameAsync(outFrame, pInputFrame, stream);
            if (sts != RGY_ERR_NONE) {
                AddMessage(RGY_LOG_ERROR, _T("Failed to copy frame: %s.\n"), get_err_mes(sts));
                return sts;
            }
            outFrame->timestamp = nextPts;
            outFrame->duration = duration;
            break;
        }
        // 途中のフレームを生成
        sts = genFrame(outFrame, bufferPrev, bufferIn, nextPts, stream);
        if (sts != RGY_ERR_NONE) {
            return sts;
        }
        outFrame->timestamp = nextPts;
        outFrame->duration = duration;
        prevTimestamp = nextPts;
    }
    m_prevTimestamp = pInputFrame->timestamp;
    return RGY_ERR_NONE;
}

RGY_ERR NVEncFilterNVOFFRUC::genFrame(RGYFrameInfo *outFrame, const CUFrameBuf *prev, const CUFrameBuf *curr, const int64_t genPts, cudaStream_t stream) {
    bool ignored = false;

    CUFrameBuf *work = m_frucBuf.back().get();
    copyFrameProp(&work->frame, &curr->frame);
    work->setTimestamp(genPts);
    work->setInputFrameId(curr->inputFrameId());
    {
        NVEncNVOFFRUCParams procParam = { 0 };
        procParam.frameIn = prev->ptrY();
        procParam.timestampIn = prev->timestamp();
        procParam.frameOut = work->ptrY();
        procParam.timestampOut = prev->timestamp();
        auto sts = m_func->fproc(m_frucHandle.get(), &procParam);
        if (sts != RGY_ERR_NONE) {
            AddMessage(RGY_LOG_ERROR, _T("FRUC: Process(0) error %lld - %lld -> %lld: %s.\n"),
                prev->timestamp(), curr->timestamp(), work->timestamp(), get_err_mes(sts));
            return sts;
        }
    }
    {
        NVEncNVOFFRUCParams procParam = { 0 };
        procParam.frameIn = curr->ptrY();
        procParam.timestampIn = curr->timestamp();
        procParam.frameOut = work->ptrY();
        procParam.timestampOut = work->timestamp();
        auto sts = m_func->fproc(m_frucHandle.get(), &procParam);
        if (sts != RGY_ERR_NONE) {
            AddMessage(RGY_LOG_ERROR, _T("FRUC: Process(1) error %lld - %lld -> %lld: %s.\n"),
                prev->timestamp(), curr->timestamp(), work->timestamp(), get_err_mes(sts));
            return sts;
        }
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
        copyFrameProp(outFrame, &work->frame);
    }
    return RGY_ERR_NONE;
}

void NVEncFilterNVOFFRUC::close() {
    m_srcCrop.reset();
    m_dstCrop.reset();
    for (auto& buf : m_frucBuf) {
        buf.reset();
    }
    m_frucHandle.reset();
    m_func.reset();
    m_frameBuf.clear();
}
