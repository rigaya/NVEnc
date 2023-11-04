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
#include "convert_csp.h"
#include "NVEncFilterRff.h"
#include "NVEncParam.h"
#pragma warning (push)

static const int FRAME_OUT_INDEX = FRAME_BUF_SIZE;

tstring NVEncFilterParamRff::print() const {
    return rff.print();
}

NVEncFilterRff::NVEncFilterRff() :
    m_nFieldBufUsed(0),
    m_nFieldBufPicStruct({ RGY_FRAME_FLAG_NONE, RGY_FRAME_FLAG_NONE }),
    m_ptsOffset(0),
    m_prevInputTimestamp(-1),
    m_prevInputFlags(RGY_FRAME_FLAG_NONE),
    m_fpLog() {
    m_sFilterName = _T("rff");
}

NVEncFilterRff::~NVEncFilterRff() {
    close();
}

RGY_ERR NVEncFilterRff::checkParam(const NVEncFilterParam *param) {
    auto prm = dynamic_cast<const NVEncFilterParamRff *>(param);
    if (!prm) {
        AddMessage(RGY_LOG_ERROR, _T("Invalid parameter type.\n"));
        return RGY_ERR_INVALID_PARAM;
    }
    //パラメータチェック
    if (prm->frameOut.height <= 0 || prm->frameOut.width <= 0) {
        AddMessage(RGY_LOG_ERROR, _T("Invalid parameter.\n"));
        return RGY_ERR_INVALID_PARAM;
    }
    return RGY_ERR_NONE;
}

RGY_ERR NVEncFilterRff::init(shared_ptr<NVEncFilterParam> pParam, shared_ptr<RGYLog> pPrintMes) {
    RGY_ERR sts = RGY_ERR_NONE;
    m_pPrintMes = pPrintMes;
    auto prm = std::dynamic_pointer_cast<NVEncFilterParamRff>(pParam);
    if (!prm) {
        AddMessage(RGY_LOG_ERROR, _T("Invalid parameter type.\n"));
        return RGY_ERR_INVALID_PARAM;
    }

    prm->frameOut.pitch = prm->frameIn.pitch;

    if (!m_pParam || cmpFrameInfoCspResolution(&m_pParam->frameOut, &prm->frameOut)) {
        auto cudaerr = AllocFrameBuf(prm->frameOut, FRAME_BUF_SIZE + 1);
        if (cudaerr != cudaSuccess) {
            AddMessage(RGY_LOG_ERROR, _T("failed to allocate memory: %s.\n"), char_to_tstring(cudaGetErrorName(cudaerr)).c_str());
            return RGY_ERR_MEMORY_ALLOC;
        }
        if (prm->rff.log) {
            m_fpLog = std::unique_ptr<FILE, fp_deleter>(_tfopen((prm->outFilename + _T(".rff.log")).c_str(), _T("w")), fp_deleter());
        }
        m_nFieldBufUsed = 0;
        m_nFieldBufPicStruct.fill(RGY_FRAME_FLAG_NONE);
        m_ptsOffset = 0;
        m_prevInputTimestamp = -1;
        m_prevInputFlags = RGY_FRAME_FLAG_NONE;
    }

    m_nPathThrough &= (~(FILTER_PATHTHROUGH_PICSTRUCT | FILTER_PATHTHROUGH_FLAGS | FILTER_PATHTHROUGH_TIMESTAMP));

    setFilterInfo(pParam->print());
    m_pParam = pParam;
    return sts;
}

std::tuple<RGY_ERR, int, bool> NVEncFilterRff::copyFieldFromBuffer(RGYFrameInfo *dst, const int idx, cudaStream_t& stream) {
    const int targetIdx = idx % FRAME_BUF_SIZE;
    const bool copyTopField = (m_nFieldBufPicStruct[targetIdx] & RGY_FRAME_FLAG_RFF_TFF) != 0;
    const int inputFrameId = m_pFrameBuf[targetIdx]->frame.inputFrameId;
    // m_cl->copyFrameはframe情報をsrcからコピーする
    // dst側の情報を維持するため、あらかじめdstの情報をコピーしておく
    copyFrameProp(&m_pFrameBuf[targetIdx]->frame, dst);
    auto err = err_to_rgy(copyFrameFieldAsync(dst, &m_pFrameBuf[targetIdx]->frame, copyTopField, copyTopField, stream));
    m_nFieldBufPicStruct[targetIdx] = RGY_FRAME_FLAG_NONE;
    return { err, inputFrameId, copyTopField };
}

RGY_ERR NVEncFilterRff::copyFieldToBuffer(const RGYFrameInfo *src, const bool copyTopField, cudaStream_t& stream) {
    const int targetIdx = (m_nFieldBufUsed++) % m_nFieldBufPicStruct.size();
    if (copyTopField) {
        m_nFieldBufPicStruct[targetIdx] = RGY_FRAME_FLAG_RFF | RGY_FRAME_FLAG_RFF_TFF;
    } else {
        m_nFieldBufPicStruct[targetIdx] = RGY_FRAME_FLAG_RFF | RGY_FRAME_FLAG_RFF_BFF;
    }
    auto err = err_to_rgy(copyFrameFieldAsync(&m_pFrameBuf[targetIdx]->frame, src, copyTopField, copyTopField, stream));
    copyFrameProp(&m_pFrameBuf[targetIdx]->frame, src);
    return err;
}

int64_t NVEncFilterRff::getInputDuration(const RGYFrameInfo *pInputFrame) {
    if (pInputFrame->duration) return pInputFrame->duration;
    // durationがない場合は、前のフレームから推定する
    if (m_prevInputTimestamp >= 0) {
        auto est_duration = rgy_rational<decltype(m_prevInputTimestamp)>(pInputFrame->timestamp - m_prevInputTimestamp, 1);
        if (pInputFrame->flags & RGY_FRAME_FLAG_RFF) {
            est_duration *= rgy_rational<decltype(m_prevInputTimestamp)>(3, 2);
        }
        if (m_prevInputFlags & RGY_FRAME_FLAG_RFF) {
            est_duration *= rgy_rational<decltype(m_prevInputTimestamp)>(2, 3);
        }
        return est_duration.round();
    }
    // わからない場合はfpsから推定する
    auto prm = std::dynamic_pointer_cast<NVEncFilterParamRff>(m_pParam);
    if (!prm) {
        AddMessage(RGY_LOG_ERROR, _T("Invalid parameter type.\n"));
        return -1;
    }
    return (prm->timebase.inv() / prm->inFps).round();
}

RGY_FRAME_FLAGS NVEncFilterRff::getPrevBufFlags() const {
    return m_nFieldBufPicStruct[(m_nFieldBufUsed - 1) % m_nFieldBufPicStruct.size()];
}

RGY_ERR NVEncFilterRff::run_filter(const RGYFrameInfo *pInputFrame, RGYFrameInfo **ppOutputFrames, int *pOutputFrameNum, cudaStream_t stream) {
    UNREFERENCED_PARAMETER(pOutputFrameNum);
    RGY_ERR sts = RGY_ERR_NONE;
    if (pInputFrame->ptr == nullptr) {
        return sts;
    }

    auto prm = std::dynamic_pointer_cast<NVEncFilterParamRff>(m_pParam);
    if (!prm) {
        AddMessage(RGY_LOG_ERROR, _T("Invalid parameter type.\n"));
        return RGY_ERR_INVALID_PARAM;
    }

    const int64_t input_duration = getInputDuration(pInputFrame);

    const auto prevBufFieldPicStruct = getPrevBufFlags();

    const int inputTFF = (pInputFrame->flags & RGY_FRAME_FLAG_RFF_TFF) ? 1 : 0;
    const int inputRFF = (pInputFrame->flags & RGY_FRAME_FLAG_RFF) ? 1 : 0;
    const int prevFieldCached = (prevBufFieldPicStruct & RGY_FRAME_FLAG_RFF) ? 1 : 0;
    const auto outputPicstruct = ((inputTFF + prevFieldCached) & 1) ? RGY_PICSTRUCT_FRAME_TFF : RGY_PICSTRUCT_FRAME_BFF;
    const auto prevInputFlags = m_prevInputFlags;

    auto log_mes = strsprintf(_T("%6d, %12lld: %12s %s %s"),
        pInputFrame->inputFrameId, pInputFrame->timestamp, picstrcut_to_str(pInputFrame->picstruct),
        inputTFF ? _T("TFF") : _T("   "),
        inputRFF ? _T("RFF") : _T("   "));

    m_prevInputTimestamp = pInputFrame->timestamp;
    m_prevInputFlags = pInputFrame->flags;

    const RGY_FRAME_FLAGS rff_flags = RGY_FRAME_FLAG_RFF | RGY_FRAME_FLAG_RFF_COPY | RGY_FRAME_FLAG_RFF_TFF | RGY_FRAME_FLAG_RFF_BFF;
    if (!(prevBufFieldPicStruct & RGY_FRAME_FLAG_RFF)) { //バッファが使われていない場合
        // 入力フレームはそのまま (入力フレームと出力フレームは同じなので、コピーの必要はない)
        if (!ppOutputFrames[0]->duration) {
            ppOutputFrames[0]->duration = input_duration;
        }
        // RFF_TFFかRFF_BFFがあれば、RFFフラグの適用区間なので、RFF_XXXに合わせてpicstructを変更する
        if (((pInputFrame->flags | prevInputFlags) & (RGY_FRAME_FLAG_RFF_TFF | RGY_FRAME_FLAG_RFF_BFF))
            || (m_prevInputPicStruct & RGY_PICSTRUCT_INTERLACED)) {
            ppOutputFrames[0]->picstruct = outputPicstruct;
        }
        if ((pInputFrame->flags & RGY_FRAME_FLAG_RFF) != 0) {
            // RFFがある場合、対応するフィールドをコピー
            const auto copyTopField = (pInputFrame->flags & RGY_FRAME_FLAG_RFF_TFF)  != 0;
            if ((sts = copyFieldToBuffer(pInputFrame, copyTopField, stream)) != RGY_ERR_NONE) { return sts; }
            //rffを展開する場合、時間を補正する
            m_ptsOffset = -1 * input_duration / 3;
            ppOutputFrames[0]->duration += m_ptsOffset;
        }
        log_mes += strsprintf(_T(" -> %12lld: [%6d/%6d]: %12s\n"), ppOutputFrames[0]->timestamp,
            ppOutputFrames[0]->inputFrameId, ppOutputFrames[0]->inputFrameId, picstrcut_to_str(ppOutputFrames[0]->picstruct));
    } else { //バッファが使われている場合
        if (pInputFrame->flags & RGY_FRAME_FLAG_RFF) {
            // RFFがある場合、自分をコピー
            *pOutputFrameNum = 2;
            ppOutputFrames[1] = &m_pFrameBuf[FRAME_OUT_INDEX]->frame;
            sts = err_to_rgy(copyFrameAsync(ppOutputFrames[1], pInputFrame, stream));
            if (sts != RGY_ERR_NONE) { return sts; }

            // m_nFieldBufPicStruct側をバッファからコピー
            auto [err, bufInputFrameId, copiedTopField] = copyFieldFromBuffer(ppOutputFrames[0], m_nFieldBufUsed - 1, stream);
            if (err != RGY_ERR_NONE) { return err; }

            ppOutputFrames[0]->picstruct = outputPicstruct;
            ppOutputFrames[0]->duration = input_duration * 2 / 3;
            ppOutputFrames[0]->timestamp += m_ptsOffset;

            ppOutputFrames[1]->picstruct = outputPicstruct;
            ppOutputFrames[1]->duration = input_duration - m_ptsOffset - ppOutputFrames[0]->duration;
            ppOutputFrames[1]->timestamp = ppOutputFrames[0]->timestamp + ppOutputFrames[0]->duration;
            ppOutputFrames[1]->inputFrameId = ppOutputFrames[0]->inputFrameId;
            m_ptsOffset = 0;

            const auto log_mes_len = log_mes.length();

            log_mes += strsprintf(_T(" -> %12lld: [%6d/%6d]: %12s\n"), ppOutputFrames[0]->timestamp,
                (copiedTopField) ? bufInputFrameId : ppOutputFrames[0]->inputFrameId,
                (copiedTopField) ? ppOutputFrames[0]->inputFrameId : bufInputFrameId,
                picstrcut_to_str(ppOutputFrames[0]->picstruct));
            log_mes += decltype(log_mes)(log_mes_len, _T(' '));
            log_mes += strsprintf(_T("  + %12lld: [%6d/%6d]: %12s\n"), ppOutputFrames[1]->timestamp,
                ppOutputFrames[1]->inputFrameId, ppOutputFrames[1]->inputFrameId,
                picstrcut_to_str(ppOutputFrames[1]->picstruct));
        } else {
            const auto copyTopField = (prevBufFieldPicStruct & RGY_FRAME_FLAG_RFF_TFF) != 0;
            if ((sts = copyFieldToBuffer(pInputFrame, copyTopField, stream)) != RGY_ERR_NONE) { return sts; }

            // m_nFieldBufPicStruct側をバッファからコピー
            auto [err, bufInputFrameId, copiedTopField] = copyFieldFromBuffer(ppOutputFrames[0], m_nFieldBufUsed - 2, stream);
            if (err != RGY_ERR_NONE) { return sts; }
            ppOutputFrames[0]->picstruct = outputPicstruct;
            ppOutputFrames[0]->timestamp += m_ptsOffset;
            if (!ppOutputFrames[0]->duration) {
                ppOutputFrames[0]->duration = input_duration;
            }

            log_mes += strsprintf(_T(" -> %12lld: [%6d/%6d]: %12s\n"), ppOutputFrames[0]->timestamp,
                (copiedTopField) ? bufInputFrameId : ppOutputFrames[0]->inputFrameId,
                (copiedTopField) ? ppOutputFrames[0]->inputFrameId : bufInputFrameId,
                picstrcut_to_str(ppOutputFrames[0]->picstruct));
        }
    }
    ppOutputFrames[0]->flags &= ~(rff_flags);
    m_prevInputPicStruct = outputPicstruct;
    if (m_fpLog) {
        fprintf(m_fpLog.get(), "%s", tchar_to_string(log_mes).c_str());
    }
    //AddMessage(RGY_LOG_WARN, _T("%s"), log_mes.c_str());

    return sts;
}

void NVEncFilterRff::close() {
    m_pFrameBuf.clear();
    m_fpLog.reset();
}
