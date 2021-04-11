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

#include <sstream>
#include <fcntl.h>
#include "rgy_input_raw.h"

#if ENABLE_RAW_READER

RGY_ERR RGYInputRaw::ParseY4MHeader(char *buf, VideoInfo *pInfo) {
    char *p, *q = nullptr;

    for (p = buf; (p = strtok_s(p, " ", &q)) != nullptr; ) {
        switch (*p) {
        case 'W':
        {
            char *eptr = nullptr;
            int w = strtol(p+1, &eptr, 10);
            if (*eptr == '\0' && w)
                pInfo->srcWidth = w;
        }
        break;
        case 'H':
        {
            char *eptr = nullptr;
            int h = strtol(p+1, &eptr, 10);
            if (*eptr == '\0' && h)
                pInfo->srcHeight = h;
        }
        break;
        case 'F':
        {
            int rate = 0, scale = 0;
            if ((pInfo->fpsN == 0 || pInfo->fpsD == 0)
                && sscanf_s(p+1, "%d:%d", &rate, &scale) == 2) {
                if (rate && scale) {
                    pInfo->fpsN = rate;
                    pInfo->fpsD = scale;
                }
            }
        }
        break;
        case 'A':
        {
            int sar_x = 0, sar_y = 0;
            if ((pInfo->sar[0] == 0 || pInfo->sar[1] == 0)
                && sscanf_s(p+1, "%d:%d", &sar_x, &sar_y) == 2) {
                if (sar_x && sar_y) {
                    pInfo->sar[0] = sar_x;
                    pInfo->sar[1] = sar_y;
                }
            }
        }
        break;
        case 'I':
            switch (*(p+1)) {
            case 'b':
                pInfo->picstruct = RGY_PICSTRUCT_TFF;
                break;
            case 't':
            case 'm':
                pInfo->picstruct = RGY_PICSTRUCT_FRAME;
                break;
            default:
                break;
            }
            break;
        case 'C':
            if (0 == _strnicmp(p+1, "420p9", strlen("420p9"))) {
                pInfo->csp = RGY_CSP_YV12_09;
            } else if (0 == _strnicmp(p+1, "420p10", strlen("420p10"))) {
                pInfo->csp = RGY_CSP_YV12_10;
            } else if (0 == _strnicmp(p+1, "420p12", strlen("420p12"))) {
                pInfo->csp = RGY_CSP_YV12_12;
            } else if (0 == _strnicmp(p+1, "420p14", strlen("420p14"))) {
                pInfo->csp = RGY_CSP_YV12_14;
            } else if (0 == _strnicmp(p+1, "420p16", strlen("420p16"))) {
                pInfo->csp = RGY_CSP_YV12_16;
            } else if (0 == _strnicmp(p+1, "420mpeg2", strlen("420mpeg2"))
                    || 0 == _strnicmp(p+1, "420jpeg",  strlen("420jpeg"))
                    || 0 == _strnicmp(p+1, "420paldv", strlen("420paldv"))
                    || 0 == _strnicmp(p+1, "420",      strlen("420"))) {
                pInfo->csp = RGY_CSP_YV12;
            } else if (0 == _strnicmp(p+1, "422p9", strlen("422p9"))) {
                pInfo->csp = RGY_CSP_YUV422_09;
            } else if (0 == _strnicmp(p+1, "422p10", strlen("422p10"))) {
                pInfo->csp = RGY_CSP_YUV422_10;
            } else if (0 == _strnicmp(p+1, "422p12", strlen("422p12"))) {
                pInfo->csp = RGY_CSP_YUV422_12;
            } else if (0 == _strnicmp(p+1, "422p14", strlen("422p14"))) {
                pInfo->csp = RGY_CSP_YUV422_14;
            } else if (0 == _strnicmp(p+1, "422p16", strlen("422p16"))) {
                pInfo->csp = RGY_CSP_YUV422_16;
            } else if (0 == _strnicmp(p+1, "422", strlen("422"))) {
                pInfo->csp = RGY_CSP_YUV422;
            } else if (0 == _strnicmp(p+1, "444p9", strlen("444p9"))) {
                pInfo->csp = RGY_CSP_YUV444_09;
            } else if (0 == _strnicmp(p+1, "444p10", strlen("444p10"))) {
                pInfo->csp = RGY_CSP_YUV444_10;
            } else if (0 == _strnicmp(p+1, "444p12", strlen("444p12"))) {
                pInfo->csp = RGY_CSP_YUV444_12;
            } else if (0 == _strnicmp(p+1, "444p14", strlen("444p14"))) {
                pInfo->csp = RGY_CSP_YUV444_14;
            } else if (0 == _strnicmp(p+1, "444p16", strlen("444p16"))) {
                pInfo->csp = RGY_CSP_YUV444_16;
            } else if (0 == _strnicmp(p+1, "444", strlen("444"))) {
                pInfo->csp = RGY_CSP_YUV444;
            } else if (0 == _strnicmp(p+1, "nv12", strlen("nv12"))) {
                pInfo->csp = RGY_CSP_NV12;
            } else if (0 == _strnicmp(p+1, "p010", strlen("p010"))) {
                pInfo->csp = RGY_CSP_P010;
            } else {
                return RGY_ERR_INVALID_COLOR_FORMAT;
            }
            break;
        default:
            break;
        }
        p = nullptr;
    }
    if (pInfo->fpsN > 0 && pInfo->fpsD > 0) {
        rgy_reduce(pInfo->fpsN, pInfo->fpsD);
    }
    pInfo->srcPitch = pInfo->srcWidth * ((RGY_CSP_BIT_DEPTH[pInfo->csp] > 8) ? 2 : 1);
    return RGY_ERR_NONE;
}

RGYInputRaw::RGYInputRaw() :
    m_fSource(NULL),
    m_nBufSize(0),
    m_pBuffer() {
    m_readerName = _T("raw");
}

RGYInputRaw::~RGYInputRaw() {
    Close();
}

void RGYInputRaw::Close() {
    if (m_fSource) {
        fclose(m_fSource);
        m_fSource = NULL;
    }
    m_pBuffer.reset();
    m_nBufSize = 0;
    RGYInput::Close();
}

RGY_ERR RGYInputRaw::Init(const TCHAR *strFileName, VideoInfo *pInputInfo, const RGYInputPrm *prm) {
    m_inputVideoInfo = *pInputInfo;
    m_readerName = (m_inputVideoInfo.type == RGY_INPUT_FMT_Y4M) ? _T("y4m") : _T("raw");

    m_convert = std::make_unique<RGYConvertCSP>(prm->threadCsp);

    bool use_stdin = _tcscmp(strFileName, _T("-")) == 0;
    if (use_stdin) {
        m_fSource = stdin;
#if defined(_WIN32) || defined(_WIN64)
        if (_setmode(_fileno(stdin), _O_BINARY) < 0) {
            AddMessage(RGY_LOG_ERROR, _T("failed to switch stdin to binary mode.\n"));
            return RGY_ERR_UNDEFINED_BEHAVIOR;
        }
#endif //#if defined(_WIN32) || defined(_WIN64)
        AddMessage(RGY_LOG_DEBUG, _T("output to stdout.\n"));
    } else {
        int error = 0;
        if (0 != (error = _tfopen_s(&m_fSource, strFileName, _T("rb"))) || m_fSource == nullptr) {
            AddMessage(RGY_LOG_ERROR, _T("Failed to open file \"%s\": %s.\n"), strFileName, _tcserror(error));
            return RGY_ERR_FILE_OPEN;
        } else {
            AddMessage(RGY_LOG_DEBUG, _T("Opened file: \"%s\".\n"), strFileName);
        }
    }

    auto nOutputCSP = m_inputVideoInfo.csp; //RGYInputRawがエンコーダに渡すべき色空間
    m_inputCsp = RGY_CSP_YV12;
    if (m_inputVideoInfo.type == RGY_INPUT_FMT_Y4M) {
        //read y4m header
        char buf[128] = { 0 };
        if (fread(buf, 1, strlen("YUV4MPEG2"), m_fSource) != strlen("YUV4MPEG2")
            || strcmp(buf, "YUV4MPEG2") != 0
            || !fgets(buf, sizeof(buf), m_fSource)
            || RGY_ERR_NONE != ParseY4MHeader(buf, &m_inputVideoInfo)) {
            AddMessage(RGY_LOG_ERROR, _T("failed to parse y4m header."));
            return RGY_ERR_INVALID_FORMAT;
        }
        m_inputCsp = m_inputVideoInfo.csp;
    } else {
        auto rawprm = reinterpret_cast<const RGYInputPrmRaw *>(prm);
        m_inputCsp = (rawprm->inputCsp == RGY_CSP_NA) ? RGY_CSP_YV12 : rawprm->inputCsp; //input-cspで指定されたraw読み込みの値
        m_inputVideoInfo.srcPitch = m_inputVideoInfo.srcWidth * (RGY_CSP_BIT_DEPTH[m_inputCsp] > 8 ? 2 : 1);
    }

    RGY_CSP output_csp_if_lossless = RGY_CSP_NA;
    uint32_t bufferSize = 0;
    switch (m_inputCsp) {
    case RGY_CSP_NV12:
    case RGY_CSP_YV12:
        bufferSize = m_inputVideoInfo.srcWidth * m_inputVideoInfo.srcHeight * 3 / 2;
        output_csp_if_lossless = RGY_CSP_NV12;
        break;
    case RGY_CSP_P010:
        bufferSize = m_inputVideoInfo.srcWidth * m_inputVideoInfo.srcHeight * 3;
        output_csp_if_lossless = RGY_CSP_P010;
        break;
    case RGY_CSP_YV12_09:
    case RGY_CSP_YV12_10:
    case RGY_CSP_YV12_12:
    case RGY_CSP_YV12_14:
    case RGY_CSP_YV12_16:
        bufferSize = m_inputVideoInfo.srcWidth * m_inputVideoInfo.srcHeight * 3;
        output_csp_if_lossless = RGY_CSP_P010;
        break;
    case RGY_CSP_YUV422:
        bufferSize = m_inputVideoInfo.srcWidth * m_inputVideoInfo.srcHeight * 2;
        if (ENCODER_QSV || ENCODER_VCEENC) {
            nOutputCSP = RGY_CSP_NV12;
            output_csp_if_lossless = RGY_CSP_NV12;
        } else {
            //yuv422読み込みは、出力フォーマットへの直接変換を持たないのでNV16に変換する
            nOutputCSP = RGY_CSP_NV16;
            output_csp_if_lossless = RGY_CSP_YUV444;
        }
        break;
    case RGY_CSP_YUV422_09:
    case RGY_CSP_YUV422_10:
    case RGY_CSP_YUV422_12:
    case RGY_CSP_YUV422_14:
    case RGY_CSP_YUV422_16:
        bufferSize = m_inputVideoInfo.srcWidth * m_inputVideoInfo.srcHeight * 4;
        if (ENCODER_QSV || ENCODER_VCEENC) {
            //yuv422読み込みは、出力フォーマットへの直接変換を持たないのでP010に変換する
            nOutputCSP = RGY_CSP_P010;
            output_csp_if_lossless = RGY_CSP_P010;
        } else {
            //yuv422読み込みは、出力フォーマットへの直接変換を持たないのでP210に変換する
            nOutputCSP = RGY_CSP_P210;
            output_csp_if_lossless = RGY_CSP_YUV444_16;
        }
        break;
    case RGY_CSP_YUV444:
        bufferSize = m_inputVideoInfo.srcWidth * m_inputVideoInfo.srcHeight * 3;
        output_csp_if_lossless = RGY_CSP_YUV444;
        break;
    case RGY_CSP_YUV444_09:
    case RGY_CSP_YUV444_10:
    case RGY_CSP_YUV444_12:
    case RGY_CSP_YUV444_14:
    case RGY_CSP_YUV444_16:
        bufferSize = m_inputVideoInfo.srcWidth * m_inputVideoInfo.srcHeight * 6;
        output_csp_if_lossless = RGY_CSP_YUV444_16;
        break;
    default:
        AddMessage(RGY_LOG_ERROR, _T("Unknown color foramt.\n"));
        return RGY_ERR_INVALID_COLOR_FORMAT;
    }
    AddMessage(RGY_LOG_DEBUG, _T("%dx%d, pitch:%d, bufferSize:%d.\n"), m_inputVideoInfo.srcWidth, m_inputVideoInfo.srcHeight, m_inputVideoInfo.srcPitch, bufferSize);

    if (nOutputCSP != RGY_CSP_NA) {
        m_inputVideoInfo.csp =
            (ENCODER_NVENC
                && RGY_CSP_BIT_PER_PIXEL[m_inputCsp] < RGY_CSP_BIT_PER_PIXEL[nOutputCSP])
            ? output_csp_if_lossless : nOutputCSP;
    } else {
        //ロスレスの場合は、入力側で出力フォーマットを決める
        m_inputVideoInfo.csp = output_csp_if_lossless;
    }

    m_inputVideoInfo.bitdepth = RGY_CSP_BIT_DEPTH[m_inputVideoInfo.csp];
    if (cspShiftUsed(m_inputVideoInfo.csp) && RGY_CSP_BIT_DEPTH[m_inputVideoInfo.csp] > RGY_CSP_BIT_DEPTH[m_inputCsp]) {
        m_inputVideoInfo.bitdepth = RGY_CSP_BIT_DEPTH[m_inputCsp];
    }
    m_pBuffer = std::shared_ptr<uint8_t>((uint8_t *)_aligned_malloc(bufferSize, 32), aligned_malloc_deleter());
    if (!m_pBuffer) {
        AddMessage(RGY_LOG_ERROR, _T("Failed to allocate input buffer.\n"));
        return RGY_ERR_NULL_PTR;
    }

    if (m_convert->getFunc(m_inputCsp, m_inputVideoInfo.csp, false, prm->simdCsp) == nullptr) {
        AddMessage(RGY_LOG_ERROR, _T("raw/y4m: color conversion not supported: %s -> %s.\n"),
            RGY_CSP_NAMES[m_inputCsp], RGY_CSP_NAMES[m_inputVideoInfo.csp]);
        return RGY_ERR_INVALID_COLOR_FORMAT;
    }

    CreateInputInfo(m_readerName.c_str(), RGY_CSP_NAMES[m_convert->getFunc()->csp_from], RGY_CSP_NAMES[m_convert->getFunc()->csp_to], get_simd_str(m_convert->getFunc()->simd), &m_inputVideoInfo);
    AddMessage(RGY_LOG_DEBUG, m_inputInfo);
    *pInputInfo = m_inputVideoInfo;
    return RGY_ERR_NONE;
}

RGY_ERR RGYInputRaw::LoadNextFrame(RGYFrame *pSurface) {
    //m_encSatusInfo->m_nInputFramesがtrimの結果必要なフレーム数を大きく超えたら、エンコードを打ち切る
    //ちょうどのところで打ち切ると他のストリームに影響があるかもしれないので、余分に取得しておく
    if (getVideoTrimMaxFramIdx() < (int)m_encSatusInfo->m_sData.frameIn - TRIM_OVERREAD_FRAMES) {
        return RGY_ERR_MORE_DATA;
    }

    if (m_inputVideoInfo.type == RGY_INPUT_FMT_Y4M) {
        uint8_t y4m_buf[8] = { 0 };
        if (_fread_nolock(y4m_buf, 1, strlen("FRAME"), m_fSource) != strlen("FRAME")) {
            AddMessage(RGY_LOG_DEBUG, _T("header1: finish.\n"));
            return RGY_ERR_MORE_DATA;
        }
        if (memcmp(y4m_buf, "FRAME", strlen("FRAME")) != 0) {
            AddMessage(RGY_LOG_DEBUG, _T("header2: finish.\n"));
            return RGY_ERR_MORE_DATA;
        }
        int i;
        for (i = 0; _fgetc_nolock(m_fSource) != '\n'; i++) {
            if (i >= 64) {
                AddMessage(RGY_LOG_DEBUG, _T("header3: finish.\n"));
                return RGY_ERR_MORE_DATA;
            }
        }
    }

    uint32_t frameSize = 0;
    switch (m_convert->getFunc()->csp_from) {
    case RGY_CSP_NV12:
    case RGY_CSP_YV12:
        frameSize = m_inputVideoInfo.srcWidth * m_inputVideoInfo.srcHeight * 3 / 2; break;
    case RGY_CSP_P010:
        frameSize = m_inputVideoInfo.srcWidth * m_inputVideoInfo.srcHeight * 3; break;
    case RGY_CSP_YUV422:
        frameSize = m_inputVideoInfo.srcWidth * m_inputVideoInfo.srcHeight * 2; break;
    case RGY_CSP_YUV444:
        frameSize = m_inputVideoInfo.srcWidth * m_inputVideoInfo.srcHeight * 3; break;
    case RGY_CSP_YV12_09:
    case RGY_CSP_YV12_10:
    case RGY_CSP_YV12_12:
    case RGY_CSP_YV12_14:
    case RGY_CSP_YV12_16:
        frameSize = m_inputVideoInfo.srcWidth * m_inputVideoInfo.srcHeight * 3; break;
    case RGY_CSP_YUV422_09:
    case RGY_CSP_YUV422_10:
    case RGY_CSP_YUV422_12:
    case RGY_CSP_YUV422_14:
    case RGY_CSP_YUV422_16:
        frameSize = m_inputVideoInfo.srcWidth * m_inputVideoInfo.srcHeight * 4; break;
    case RGY_CSP_YUV444_09:
    case RGY_CSP_YUV444_10:
    case RGY_CSP_YUV444_12:
    case RGY_CSP_YUV444_14:
    case RGY_CSP_YUV444_16:
        frameSize = m_inputVideoInfo.srcWidth * m_inputVideoInfo.srcHeight * 6; break;
    default:
        AddMessage(RGY_LOG_ERROR, _T("Unknown color foramt.\n"));
        return RGY_ERR_INVALID_COLOR_FORMAT;
    }
    if (frameSize != _fread_nolock(m_pBuffer.get(), 1, frameSize, m_fSource)) {
        AddMessage(RGY_LOG_DEBUG, _T("fread: finish: %d.\n"), frameSize);
        return RGY_ERR_MORE_DATA;
    }

    void *dst_array[3];
    pSurface->ptrArray(dst_array, m_convert->getFunc()->csp_to == RGY_CSP_RGB24 || m_convert->getFunc()->csp_to == RGY_CSP_RGB32);

    const void *src_array[3];
    src_array[0] = m_pBuffer.get();
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
        src_array[2] = (uint8_t *)src_array[1] + m_inputVideoInfo.srcPitch * m_inputVideoInfo.srcHeight;
        break;
    case RGY_CSP_NV12:
    case RGY_CSP_P010:
    default:
        break;
    }

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

    m_encSatusInfo->m_sData.frameIn++;
    return m_encSatusInfo->UpdateDisplay();
}

#endif
