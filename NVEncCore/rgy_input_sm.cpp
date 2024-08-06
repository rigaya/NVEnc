// -----------------------------------------------------------------------------------------
// QSVEnc/NVEnc by rigaya
// -----------------------------------------------------------------------------------------
// The MIT License
//
// Copyright (c) 2011-2016 rigaya
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

#include "rgy_input_sm.h"

#if ENABLE_SM_READER


RGYInputSMPrm::RGYInputSMPrm(RGYInputPrm base) :
    RGYInputPrm(base),
    parentProcessID(false) {

}

RGYInputSM::RGYInputSM() :
    m_prm(),
    m_sm(),
    m_heBufEmpty(),
    m_heBufFilled(),
    m_parentProcess(NULL),
    m_droppedInAviutl(0) {
    m_readerName = _T("sm");
}

RGYInputSM::~RGYInputSM() {
    Close();
}

void RGYInputSM::Close() {
    for (size_t i = 0; i < m_heBufEmpty.size(); i++) {
        m_heBufEmpty[i] = NULL;
    }
    for (size_t i = 0; i < m_heBufFilled.size(); i++) {
        m_heBufFilled[i] = NULL;
    }
    m_parentProcess = NULL;
    for (auto& mem : m_sm) {
        mem.reset();
    }
    RGYInput::Close();
}

rgy_rational<int> RGYInputSM::getInputTimebase() {
    return rgy_rational<int>(m_inputVideoInfo.fpsN, m_inputVideoInfo.fpsD).inv() * rgy_rational<int>(1, 4);
}

bool RGYInputSM::isAfs() {
    RGYInputSMSharedData* prmsm = (RGYInputSMSharedData*)m_prm->ptr();
    return prmsm->afs;
}

#pragma warning(push)
#pragma warning(disable: 4312) //'型キャスト': 'uint32_t' からより大きいサイズの 'HANDLE' へ変換します。
RGY_ERR RGYInputSM::Init(const TCHAR *strFileName, VideoInfo *pInputInfo, const RGYInputPrm *prm) {
    UNREFERENCED_PARAMETER(strFileName);
    m_inputVideoInfo = *pInputInfo;

    m_readerName = _T("sm");
    if (m_timecode) {
        AddMessage(RGY_LOG_WARN, _T("--tcfile-in ignored with sm reader.\n"));
        m_timecode.reset();
    }

    m_convert = std::make_unique<RGYConvertCSP>(prm->threadCsp, prm->threadParamCsp);


    const RGYInputSMPrm *prmSM = dynamic_cast<const RGYInputSMPrm *>(prm);

    auto nOutputCSP = m_inputVideoInfo.csp;

    m_prm = std::unique_ptr<RGYSharedMemWin>(new RGYSharedMemWin(strsprintf("%s_%08x", RGYInputSMPrmSM, prmSM->parentProcessID).c_str(), sizeof(RGYInputSMSharedData)));
    if (!m_prm->is_open()) {
        AddMessage(RGY_LOG_ERROR, _T("could not open params for input: %s.\n"), char_to_tstring(m_prm->name()).c_str());
        return RGY_ERR_INVALID_HANDLE;
    }
    AddMessage(RGY_LOG_DEBUG, _T("Opened parameter struct %s, size: %u.\n"), char_to_tstring(m_prm->name()).c_str(), m_prm->size());

    m_parentProcess = OpenProcess(SYNCHRONIZE | PROCESS_VM_READ, FALSE, prmSM->parentProcessID);
    if (m_parentProcess == NULL) {
        AddMessage(RGY_LOG_ERROR, _T("could not open parent process handle.\n"));
        return RGY_ERR_INVALID_HANDLE;
    }
    AddMessage(RGY_LOG_DEBUG, _T("Parent process handle: 0x%08p.\n"), m_parentProcess);

    RGYInputSMSharedData *prmsm = (RGYInputSMSharedData *)m_prm->ptr();
    prmsm->pitch = ALIGN(prmsm->w, 128) * (RGY_CSP_BIT_DEPTH[prmsm->csp] > 8 ? 2 : 1);
    m_inputVideoInfo.srcWidth = prmsm->w;
    m_inputVideoInfo.srcHeight = prmsm->h;
    m_inputVideoInfo.fpsN = prmsm->fpsN;
    m_inputVideoInfo.fpsD = prmsm->fpsD;
    m_inputVideoInfo.srcPitch = prmsm->pitch;
    m_inputVideoInfo.picstruct = prmsm->picstruct;
    m_inputVideoInfo.frames = prmsm->frames;
    m_inputCsp = m_inputVideoInfo.csp = prmsm->csp;
    for (size_t i = 0; i < m_heBufEmpty.size(); i++) {
        m_heBufEmpty[i] = (HANDLE)prmsm->heBufEmpty[i];
    }
    for (size_t i = 0; i < m_heBufFilled.size(); i++) {
        m_heBufFilled[i] = (HANDLE)prmsm->heBufFilled[i];
    }
    AddMessage(RGY_LOG_DEBUG, _T("Got event handle empty: 0x%08p, 0x%08p, filled: 0x%08p, 0x%08p\n"), m_heBufEmpty[0], m_heBufEmpty[1], m_heBufFilled[0], m_heBufFilled[1]);

    RGY_CSP output_csp_if_lossless = RGY_CSP_NA;
    uint32_t bufferSize = 0;
    switch (m_inputCsp) {
    case RGY_CSP_NV12:
    case RGY_CSP_YV12:
        bufferSize = m_inputVideoInfo.srcPitch * m_inputVideoInfo.srcHeight * 3 / 2;
        output_csp_if_lossless = RGY_CSP_NV12;
        break;
    case RGY_CSP_P010:
        bufferSize = m_inputVideoInfo.srcPitch * m_inputVideoInfo.srcHeight * 3;
        output_csp_if_lossless = RGY_CSP_P010;
        break;
    case RGY_CSP_YV12_09:
    case RGY_CSP_YV12_10:
    case RGY_CSP_YV12_12:
    case RGY_CSP_YV12_14:
    case RGY_CSP_YV12_16:
        bufferSize = m_inputVideoInfo.srcPitch * m_inputVideoInfo.srcHeight * 3;
        output_csp_if_lossless = RGY_CSP_P010;
        break;
    case RGY_CSP_YUV422:
        bufferSize = m_inputVideoInfo.srcPitch * m_inputVideoInfo.srcHeight * 2;
        if (ENCODER_VCEENC) {
            AddMessage(RGY_LOG_ERROR, _T("yuv422 not supported as input color format.\n"));
            return RGY_ERR_INVALID_FORMAT;
        }
        //yuv422読み込みは、出力フォーマットへの直接変換を持たないのでNV16に変換する
        nOutputCSP = RGY_CSP_NV16;
        output_csp_if_lossless = RGY_CSP_YUV444;
        break;
    case RGY_CSP_YUV422_09:
    case RGY_CSP_YUV422_10:
    case RGY_CSP_YUV422_12:
    case RGY_CSP_YUV422_14:
    case RGY_CSP_YUV422_16:
        bufferSize = m_inputVideoInfo.srcPitch * m_inputVideoInfo.srcHeight * 4;
        if (ENCODER_VCEENC) {
            AddMessage(RGY_LOG_ERROR, _T("yuv422 not supported as input color format.\n"));
            return RGY_ERR_INVALID_FORMAT;
        }
        //yuv422読み込みは、出力フォーマットへの直接変換を持たないのでP210に変換する
        nOutputCSP = RGY_CSP_P210;
        output_csp_if_lossless = RGY_CSP_YUV444_16;
        break;
    case RGY_CSP_YUV444:
        bufferSize = m_inputVideoInfo.srcPitch * m_inputVideoInfo.srcHeight * 3;
        output_csp_if_lossless = RGY_CSP_YUV444;
        break;
    case RGY_CSP_YUV444_09:
    case RGY_CSP_YUV444_10:
    case RGY_CSP_YUV444_12:
    case RGY_CSP_YUV444_14:
    case RGY_CSP_YUV444_16:
        bufferSize = m_inputVideoInfo.srcPitch * m_inputVideoInfo.srcHeight * 6;
        output_csp_if_lossless = RGY_CSP_YUV444_16;
        break;
    case RGY_CSP_RGB:
        bufferSize = m_inputVideoInfo.srcPitch * m_inputVideoInfo.srcHeight * 3;
        output_csp_if_lossless = RGY_CSP_RGB;
        nOutputCSP = RGY_CSP_RGB;
        break;
    default:
        AddMessage(RGY_LOG_ERROR, _T("Unknown color foramt.\n"));
        return RGY_ERR_INVALID_COLOR_FORMAT;
    }
    AddMessage(RGY_LOG_DEBUG, _T("%s, %dx%d, pitch:%d, bufferSize:%d.\n"), RGY_CSP_NAMES[m_inputVideoInfo.csp],
        m_inputVideoInfo.srcWidth, m_inputVideoInfo.srcHeight, m_inputVideoInfo.srcPitch, bufferSize);

    if (nOutputCSP != RGY_CSP_NA) {
        m_inputVideoInfo.csp =
            (ENCODER_NVENC
                && RGY_CSP_BIT_PER_PIXEL[m_inputCsp] < RGY_CSP_BIT_PER_PIXEL[nOutputCSP])
            ? output_csp_if_lossless : nOutputCSP;
    } else {
        //ロスレスの場合は、入力側で出力フォーマットを決める
        m_inputVideoInfo.csp = output_csp_if_lossless;
    }
    //m_inputVideoInfo.shiftも出力フォーマットに対応する値でなく入力フォーマットに対するものに
    m_inputVideoInfo.bitdepth = RGY_CSP_BIT_DEPTH[m_inputVideoInfo.csp];
    if (cspShiftUsed(m_inputVideoInfo.csp) && RGY_CSP_BIT_DEPTH[m_inputVideoInfo.csp] > RGY_CSP_BIT_DEPTH[m_inputCsp]) {
        m_inputVideoInfo.bitdepth = RGY_CSP_BIT_DEPTH[m_inputCsp];
    }

    prmsm->bufSize = bufferSize;
    for (size_t i = 0; i < m_sm.size(); i++) {
        m_sm[i] = std::unique_ptr<RGYSharedMemWin>(new RGYSharedMemWin(strsprintf("%s_%08x_%d", RGYInputSMBuffer, prmSM->parentProcessID, i).c_str(), bufferSize));
        if (!m_sm[i]->is_open()) {
            AddMessage(RGY_LOG_ERROR, _T("Failed to allocate input buffer %s.\n"), char_to_tstring(m_prm->name()).c_str());
            return RGY_ERR_NULL_PTR;
        }
        AddMessage(RGY_LOG_DEBUG, _T("Created input buffer[%d] %s, size = %d.\n"), i, char_to_tstring(m_prm->name()).c_str(), m_prm->size());
    }
    for (size_t i = 0; i < m_heBufEmpty.size(); i++) {
        if (SetEvent(m_heBufEmpty[i]) == FALSE) {
            AddMessage(RGY_LOG_ERROR, _T("Failed to set event!\n"));
            return RGY_ERR_UNKNOWN;
        }
        AddMessage(RGY_LOG_DEBUG, _T("SetEvent: heBufEmpty[%d].\n"), i);
    }

    if (m_convert->getFunc(m_inputCsp, m_inputVideoInfo.csp, false, prm->simdCsp) == nullptr) {
        AddMessage(RGY_LOG_ERROR, _T("sm: color conversion not supported: %s -> %s.\n"),
            RGY_CSP_NAMES[m_inputCsp], RGY_CSP_NAMES[m_inputVideoInfo.csp]);
        return RGY_ERR_INVALID_COLOR_FORMAT;
    }

    CreateInputInfo(m_readerName.c_str(), RGY_CSP_NAMES[m_convert->getFunc()->csp_from], RGY_CSP_NAMES[m_convert->getFunc()->csp_to], get_simd_str(m_convert->getFunc()->simd), &m_inputVideoInfo);
    AddMessage(RGY_LOG_DEBUG, m_inputInfo);
    *pInputInfo = m_inputVideoInfo;
    return RGY_ERR_NONE;
}
#pragma warning(pop)

RGY_ERR RGYInputSM::LoadNextFrameInternal(RGYFrame *pSurface) {
    //m_encSatusInfo->m_nInputFramesがtrimの結果必要なフレーム数を大きく超えたら、エンコードを打ち切る
    //ちょうどのところで打ち切ると他のストリームに影響があるかもしれないので、余分に取得しておく
    if (getVideoTrimMaxFramIdx() < (int)m_encSatusInfo->m_sData.frameIn - TRIM_OVERREAD_FRAMES) {
        return RGY_ERR_MORE_DATA;
    }
    RGYInputSMSharedData *prmsm = (RGYInputSMSharedData *)m_prm->ptr();
    if (prmsm->abort) {
        return RGY_ERR_MORE_DATA;
    }

    DWORD waiterr = 0;
    while ((waiterr = WaitForSingleObject(m_heBufFilled[m_encSatusInfo->m_sData.frameIn&1], 1000)) != WAIT_OBJECT_0) {
        if (prmsm->abort) {
            return RGY_ERR_MORE_DATA;
        }
        if (waiterr == WAIT_FAILED) {
            AddMessage(RGY_LOG_ERROR, _T("Waiting for filling buffer has failed!\n"));
            return RGY_ERR_UNKNOWN;
        }
        if (WaitForSingleObject(m_parentProcess, 0) == WAIT_OBJECT_0) {
            AddMessage(RGY_LOG_ERROR, _T("Parent Process has terminated!\n"));
            return RGY_ERR_ABORTED;
        }
    }
    if (prmsm->abort) {
        return RGY_ERR_MORE_DATA;
    }

    void *dst_array[RGY_MAX_PLANES];
    pSurface->ptrArray(dst_array);

    const void *src_array[RGY_MAX_PLANES];
    src_array[0] = m_sm[m_encSatusInfo->m_sData.frameIn & 1]->ptr();
    src_array[1] = (uint8_t *)src_array[0] + m_inputVideoInfo.srcPitch * m_inputVideoInfo.srcHeight;
    switch (m_convert->getFunc()->csp_from) {
    case RGY_CSP_YV12:
    case RGY_CSP_YV12_09:
    case RGY_CSP_YV12_10:
    case RGY_CSP_YV12_12:
    case RGY_CSP_YV12_14:
    case RGY_CSP_YV12_16:
        src_array[2] = (uint8_t *)src_array[1] + m_inputVideoInfo.srcPitch * m_inputVideoInfo.srcHeight / 4;
        break;
    case RGY_CSP_YUV422:
    case RGY_CSP_YUV422_09:
    case RGY_CSP_YUV422_10:
    case RGY_CSP_YUV422_12:
    case RGY_CSP_YUV422_14:
    case RGY_CSP_YUV422_16:
        src_array[2] = (uint8_t *)src_array[1] + m_inputVideoInfo.srcPitch * m_inputVideoInfo.srcHeight / 2;
        break;
    case RGY_CSP_YUV444:
    case RGY_CSP_YUV444_09:
    case RGY_CSP_YUV444_10:
    case RGY_CSP_YUV444_12:
    case RGY_CSP_YUV444_14:
    case RGY_CSP_YUV444_16:
    case RGY_CSP_RGB:
        src_array[2] = (uint8_t *)src_array[1] + m_inputVideoInfo.srcPitch * m_inputVideoInfo.srcHeight;
        break;
    case RGY_CSP_NV12:
    case RGY_CSP_P010:
    default:
        break;
    }
    src_array[3] = nullptr;

    int src_uv_pitch = m_inputVideoInfo.srcPitch;
    switch (RGY_CSP_CHROMA_FORMAT[m_convert->getFunc()->csp_from]) {
    case RGY_CHROMAFMT_YUV422:
        src_uv_pitch >>= 1;
        break;
    case RGY_CHROMAFMT_YUV444:
        break;
    case RGY_CHROMAFMT_RGB:
    case RGY_CHROMAFMT_RGB_PACKED:
        break;
    case RGY_CHROMAFMT_YUV420:
    default:
        src_uv_pitch >>= 1;
        break;
    }
    m_convert->run((m_inputVideoInfo.picstruct & RGY_PICSTRUCT_INTERLACED) ? 1 : 0,
        dst_array, src_array, m_inputVideoInfo.srcWidth, m_inputVideoInfo.srcPitch,
        src_uv_pitch, pSurface->pitch(), m_inputVideoInfo.srcHeight, m_inputVideoInfo.srcHeight, m_inputVideoInfo.crop.c);

    pSurface->setTimestamp(prmsm->timestamp[m_encSatusInfo->m_sData.frameIn & 1]);
    pSurface->setDuration(prmsm->duration[m_encSatusInfo->m_sData.frameIn & 1]);
    m_droppedInAviutl = prmsm->dropped[m_encSatusInfo->m_sData.frameIn & 1];

    if (SetEvent(m_heBufEmpty[m_encSatusInfo->m_sData.frameIn & 1]) == FALSE) {
        AddMessage(RGY_LOG_ERROR, _T("Failed to set event!\n"));
        return RGY_ERR_UNKNOWN;
    }
    m_encSatusInfo->m_sData.frameIn++;
    return m_encSatusInfo->UpdateDisplay();
}

#endif //#if ENABLE_SM_READER
