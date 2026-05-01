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

#include "convert_csp.h"
#include "NVEncFilterBwdif.h"
#include "NVEncParam.h"

static const int BWDIF_CACHE_SIZE = 3;

NVEncFilterBwdif::NVEncFilterBwdif() :
    m_cacheFrames(),
    m_inputCount(0),
    m_drained(false),
    m_defaultTff(true) {
    m_name = _T("bwdif");
}

NVEncFilterBwdif::~NVEncFilterBwdif() {
    close();
}

RGY_ERR NVEncFilterBwdif::check_param(shared_ptr<NVEncFilterParamBwdif> prm) {
    if (prm->frameOut.height <= 0 || prm->frameOut.width <= 0) {
        AddMessage(RGY_LOG_ERROR, _T("Invalid parameter.\n"));
        return RGY_ERR_INVALID_PARAM;
    }
    const int height_mul = (RGY_CSP_CHROMA_FORMAT[prm->frameOut.csp] == RGY_CHROMAFMT_YUV420) ? 4 : 2;
    if ((prm->frameOut.height % height_mul) != 0) {
        AddMessage(RGY_LOG_ERROR, _T("Height must be multiple of %d.\n"), height_mul);
        return RGY_ERR_INVALID_PARAM;
    }
    if (prm->bwdif.thr < 0.0f || prm->bwdif.thr > 100.0f) {
        AddMessage(RGY_LOG_ERROR, _T("Invalid thr=%.3f: must be in [0.0, 100.0].\n"), prm->bwdif.thr);
        return RGY_ERR_INVALID_PARAM;
    }
    return RGY_ERR_NONE;
}

RGY_ERR NVEncFilterBwdif::init(shared_ptr<NVEncFilterParam> pParam, shared_ptr<RGYLog> pPrintMes) {
    RGY_ERR sts = RGY_ERR_NONE;
    m_pLog = pPrintMes;
    auto prm = std::dynamic_pointer_cast<NVEncFilterParamBwdif>(pParam);
    if (!prm) {
        AddMessage(RGY_LOG_ERROR, _T("Invalid parameter type.\n"));
        return RGY_ERR_INVALID_PARAM;
    }
    sts = check_param(prm);
    if (sts != RGY_ERR_NONE) {
        return sts;
    }

    prm->frameOut.picstruct = RGY_PICSTRUCT_FRAME;
    m_pathThrough &= ~(FILTER_PATHTHROUGH_PICSTRUCT | FILTER_PATHTHROUGH_FLAGS | FILTER_PATHTHROUGH_TIMESTAMP);
    if (prm->bwdif.isbob()) {
        prm->baseFps *= 2;
    }

    sts = AllocFrameBuf(prm->frameOut, prm->bwdif.isbob() ? 2 : 1);
    if (sts != RGY_ERR_NONE) {
        AddMessage(RGY_LOG_ERROR, _T("failed to allocate memory: %s.\n"), get_err_mes(sts));
        return sts;
    }
    for (int i = 0; i < RGY_CSP_PLANES[prm->frameOut.csp]; i++) {
        prm->frameOut.pitch[i] = m_frameBuf[0]->frame.pitch[i];
    }

    auto prmPrev = std::dynamic_pointer_cast<NVEncFilterParamBwdif>(m_param);
    if (!prmPrev || cmpFrameInfoCspResolution(&m_cacheFrames[0].frame, &prm->frameIn)) {
        for (auto& cache : m_cacheFrames) {
            cache.frame = prm->frameIn;
            sts = cache.alloc();
            if (sts != RGY_ERR_NONE) {
                cache.clear();
                AddMessage(RGY_LOG_ERROR, _T("failed to allocate cache frame: %s.\n"), get_err_mes(sts));
                return sts;
            }
        }
    }

    m_defaultTff = (prm->frameIn.picstruct & RGY_PICSTRUCT_BFF) == 0;

    if (!prmPrev || prmPrev->bwdif != prm->bwdif) {
        m_inputCount = 0;
        m_drained = false;
    }

    setFilterInfo(prm->print() + _T("\n                         auto-order-fallback=") + (m_defaultTff ? _T("tff") : _T("bff")));
    m_param = prm;
    return sts;
}

bool NVEncFilterBwdif::getInputTff(const RGYFrameInfo *frame) const {
    auto prm = std::dynamic_pointer_cast<NVEncFilterParamBwdif>(m_param);
    if (!prm) {
        return m_defaultTff;
    }
    if (prm->bwdif.order == VppBwdifOrder::TFF) {
        return true;
    }
    if (prm->bwdif.order == VppBwdifOrder::BFF) {
        return false;
    }
    if (frame) {
        if (frame->picstruct & RGY_PICSTRUCT_BFF) {
            return false;
        }
        if (frame->picstruct & RGY_PICSTRUCT_TFF) {
            return true;
        }
    }
    return m_defaultTff;
}

bool NVEncFilterBwdif::shouldPassthrough(const RGYFrameInfo *frame) const {
    auto prm = std::dynamic_pointer_cast<NVEncFilterParamBwdif>(m_param);
    return prm
        && prm->bwdif.order == VppBwdifOrder::Auto
        && frame
        && (frame->picstruct & RGY_PICSTRUCT_INTERLACED) == 0;
}

RGY_ERR NVEncFilterBwdif::reconstructFrame(int idx_prev, int idx_cur, int idx_next,
    bool inputTff, int preserveTopField, int outputSlot, cudaStream_t stream) {
    auto prm = std::dynamic_pointer_cast<NVEncFilterParamBwdif>(m_param);
    if (!prm) {
        AddMessage(RGY_LOG_ERROR, _T("Invalid parameter type.\n"));
        return RGY_ERR_INVALID_PARAM;
    }

    const bool xorFlag = (preserveTopField != (inputTff ? 1 : 0));
    const int idx_prev2 = xorFlag ? idx_prev : idx_cur;
    const int idx_next2 = xorFlag ? idx_cur  : idx_next;

    const RGYFrameInfo *prev2 = &m_cacheFrames[idx_prev2].frame;
    const RGYFrameInfo *prev  = &m_cacheFrames[idx_prev ].frame;
    const RGYFrameInfo *cur   = &m_cacheFrames[idx_cur  ].frame;
    const RGYFrameInfo *next  = &m_cacheFrames[idx_next ].frame;
    const RGYFrameInfo *next2 = &m_cacheFrames[idx_next2].frame;

    const int bitDepth = RGY_CSP_BIT_DEPTH[cur->csp];
    const int maxVal   = (1 << bitDepth) - 1;
    const int thr      = (int)(prm->bwdif.thr * (float)maxVal / 100.0f + 0.5f);

    auto sts = run_bwdif_frame(&m_frameBuf[outputSlot]->frame, prev2, prev, cur, next, next2, preserveTopField, thr, stream);
    if (sts != RGY_ERR_NONE) {
        AddMessage(RGY_LOG_ERROR, _T("error at run_bwdif_frame: %s.\n"), get_err_mes(sts));
        return sts;
    }
    return RGY_ERR_NONE;
}

RGY_ERR NVEncFilterBwdif::generateOutput(int idx_prev, int idx_cur, int idx_next,
    RGYFrameInfo **ppOutputFrames, int *pOutputFrameNum, cudaStream_t stream) {
    auto prm = std::dynamic_pointer_cast<NVEncFilterParamBwdif>(m_param);
    if (!prm) {
        AddMessage(RGY_LOG_ERROR, _T("Invalid parameter type.\n"));
        return RGY_ERR_INVALID_PARAM;
    }

    const bool bob = prm->bwdif.isbob();
    const RGYFrameInfo *curF = &m_cacheFrames[idx_cur].frame;
    const bool inputTff = getInputTff(curF);
    const int firstFieldParity  = inputTff ? 1 : 0;
    const int secondFieldParity = inputTff ? 0 : 1;

    if (shouldPassthrough(curF)) {
        auto err = copyFrameAsync(&m_frameBuf[0]->frame, curF, stream);
        if (err != RGY_ERR_NONE) {
            AddMessage(RGY_LOG_ERROR, _T("failed to copy progressive frame: %s.\n"), get_err_mes(err));
            return err;
        }
        auto pOut0 = &m_frameBuf[0]->frame;
        pOut0->picstruct = RGY_PICSTRUCT_FRAME;
        pOut0->flags = curF->flags;
        ppOutputFrames[0] = pOut0;
        *pOutputFrameNum = 1;
        if (bob) {
            err = copyFrameAsync(&m_frameBuf[1]->frame, curF, stream);
            if (err != RGY_ERR_NONE) {
                AddMessage(RGY_LOG_ERROR, _T("failed to duplicate progressive frame for bob: %s.\n"), get_err_mes(err));
                return err;
            }
            auto pOut1 = &m_frameBuf[1]->frame;
            pOut1->picstruct = RGY_PICSTRUCT_FRAME;
            pOut1->flags = curF->flags;
            ppOutputFrames[1] = pOut1;
            *pOutputFrameNum = 2;
            setBobTimestamp(curF, ppOutputFrames);
        } else {
            pOut0->timestamp = curF->timestamp;
            pOut0->duration = curF->duration;
            pOut0->inputFrameId = curF->inputFrameId;
        }
        return RGY_ERR_NONE;
    }

    auto err = reconstructFrame(idx_prev, idx_cur, idx_next, inputTff, firstFieldParity, 0, stream);
    if (err != RGY_ERR_NONE) {
        return err;
    }

    auto pOut0 = &m_frameBuf[0]->frame;
    pOut0->picstruct = RGY_PICSTRUCT_FRAME;
    pOut0->flags     = curF->flags;
    ppOutputFrames[0] = pOut0;
    *pOutputFrameNum  = 1;

    if (bob) {
        err = reconstructFrame(idx_prev, idx_cur, idx_next, inputTff, secondFieldParity, 1, stream);
        if (err != RGY_ERR_NONE) {
            return err;
        }
        auto pOut1 = &m_frameBuf[1]->frame;
        pOut1->picstruct = RGY_PICSTRUCT_FRAME;
        pOut1->flags     = curF->flags;
        ppOutputFrames[1] = pOut1;
        *pOutputFrameNum  = 2;
        setBobTimestamp(curF, ppOutputFrames);
    } else {
        pOut0->timestamp    = curF->timestamp;
        pOut0->duration     = curF->duration;
        pOut0->inputFrameId = curF->inputFrameId;
    }
    return RGY_ERR_NONE;
}

void NVEncFilterBwdif::setBobTimestamp(const RGYFrameInfo *pInputFrame, RGYFrameInfo **ppOutputFrames) {
    auto prm = std::dynamic_pointer_cast<NVEncFilterParamBwdif>(m_param);
    auto frameDuration = pInputFrame->duration;
    if (frameDuration == 0 && prm && prm->timebase.is_valid()) {
        frameDuration = (decltype(frameDuration))((prm->timebase.inv() / prm->baseFps * 2).qdouble() + 0.5);
    }
    ppOutputFrames[0]->timestamp    = pInputFrame->timestamp;
    ppOutputFrames[0]->duration     = (frameDuration + 1) / 2;
    ppOutputFrames[1]->timestamp    = ppOutputFrames[0]->timestamp + ppOutputFrames[0]->duration;
    ppOutputFrames[1]->duration     = frameDuration - ppOutputFrames[0]->duration;
    ppOutputFrames[0]->inputFrameId = pInputFrame->inputFrameId;
    ppOutputFrames[1]->inputFrameId = pInputFrame->inputFrameId;
}

RGY_ERR NVEncFilterBwdif::run_filter(const RGYFrameInfo *pInputFrame, RGYFrameInfo **ppOutputFrames, int *pOutputFrameNum, cudaStream_t stream) {
    *pOutputFrameNum  = 0;
    ppOutputFrames[0] = nullptr;
    ppOutputFrames[1] = nullptr;

    auto prm = std::dynamic_pointer_cast<NVEncFilterParamBwdif>(m_param);
    if (!prm) {
        AddMessage(RGY_LOG_ERROR, _T("Invalid parameter type.\n"));
        return RGY_ERR_INVALID_PARAM;
    }

    const bool hasInput = (pInputFrame && pInputFrame->ptr[0]);

    if (hasInput) {
        const auto memcpyKind = getCudaMemcpyKind(pInputFrame->mem_type, m_frameBuf[0]->frame.mem_type);
        if (memcpyKind != cudaMemcpyDeviceToDevice) {
            AddMessage(RGY_LOG_ERROR, _T("only supported on device memory.\n"));
            return RGY_ERR_INVALID_CALL;
        }
        if (m_param->frameOut.csp != m_param->frameIn.csp) {
            AddMessage(RGY_LOG_ERROR, _T("csp does not match.\n"));
            return RGY_ERR_INVALID_PARAM;
        }

        const int slot = m_inputCount % BWDIF_CACHE_SIZE;
        RGYFrameInfo *pSlot = &m_cacheFrames[slot].frame;
        auto copyErr = copyFrameAsync(pSlot, pInputFrame, stream);
        if (copyErr != RGY_ERR_NONE) {
            AddMessage(RGY_LOG_ERROR, _T("failed to copy input to cache slot %d: %s.\n"), slot, get_err_mes(copyErr));
            return copyErr;
        }
        copyFrameProp(pSlot, pInputFrame);

        m_inputCount++;
        if (m_inputCount < 2) {
            return RGY_ERR_NONE;
        }

        const int idx_cur  = (m_inputCount - 2) % BWDIF_CACHE_SIZE;
        const int idx_next = (m_inputCount - 1) % BWDIF_CACHE_SIZE;
        const int idx_prev = (m_inputCount >= 3) ? (m_inputCount - 3) % BWDIF_CACHE_SIZE : idx_cur;
        return generateOutput(idx_prev, idx_cur, idx_next, ppOutputFrames, pOutputFrameNum, stream);
    }

    if (!m_drained && m_inputCount >= 1) {
        m_drained = true;
        const int idx_cur  = (m_inputCount - 1) % BWDIF_CACHE_SIZE;
        const int idx_next = idx_cur;
        const int idx_prev = (m_inputCount >= 2) ? (m_inputCount - 2) % BWDIF_CACHE_SIZE : idx_cur;
        return generateOutput(idx_prev, idx_cur, idx_next, ppOutputFrames, pOutputFrameNum, stream);
    }

    return RGY_ERR_NONE;
}

void NVEncFilterBwdif::close() {
    for (auto& cache : m_cacheFrames) {
        cache.clear();
    }
    m_inputCount = 0;
    m_drained    = false;
    m_frameBuf.clear();
    AddMessage(RGY_LOG_DEBUG, _T("closed bwdif filter.\n"));
}
