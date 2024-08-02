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

#include "rgy_input_avi.h"
#if ENABLE_AVI_READER
#pragma warning(disable:4312)
#pragma warning(disable:4838)
#pragma warning(disable:4201)
#include "Aviriff.h"

static const auto FOURCC_CSP = make_array<std::pair<uint32_t, RGY_CSP>>(
    std::make_pair(FCC('YUY2'), RGY_CSP_YUY2),
    std::make_pair(FCC('YV12'), RGY_CSP_YV12),
    std::make_pair(FCC('NV12'), RGY_CSP_NV12)
    );

MAP_PAIR_0_1(codec, fcc, uint32_t, rgy, RGY_CSP, FOURCC_CSP, 0u, RGY_CSP_NA);

RGYInputAvi::RGYInputAvi() :
    m_pAviFile(nullptr),
    m_pAviStream(nullptr),
    m_pGetFrame(nullptr),
    m_pBitmapInfoHeader(nullptr),
    m_nYPitchMultiplizer(1),
    m_nBufSize(0),
    m_pBuffer() {
    m_readerName = _T("avi");
}

RGYInputAvi::~RGYInputAvi() {
    Close();
}

RGY_ERR RGYInputAvi::Init(const TCHAR *strFileName, VideoInfo *pInputInfo, const RGYInputPrm *prm) {
    m_inputVideoInfo = *pInputInfo;

    m_convert = std::make_unique<RGYConvertCSP>(prm->threadCsp, prm->threadParamCsp);

    AVIFileInit();

    if (0 != AVIFileOpen(&m_pAviFile, strFileName, OF_READ | OF_SHARE_DENY_NONE, NULL)) {
        AddMessage(RGY_LOG_ERROR, _T("failed to open avi file: \"%s\"\n"), strFileName);
        return RGY_ERR_FILE_OPEN;
    }
    AddMessage(RGY_LOG_DEBUG, _T("openend avi file: \"%s\"\n"), strFileName);

    AVIFILEINFO finfo = { 0 };
    if (0 != AVIFileInfo(m_pAviFile, &finfo, sizeof(AVIFILEINFO))) {
        AddMessage(RGY_LOG_ERROR, _T("failed to get avi file info.\n"));
        return RGY_ERR_INVALID_HANDLE;
    }

    tstring strFcc;
    for (uint32_t i_stream = 0; i_stream < finfo.dwStreams; i_stream++) {
        if (0 != AVIFileGetStream(m_pAviFile, &m_pAviStream, 0, i_stream))
            return RGY_ERR_INVALID_HANDLE;
        AVISTREAMINFO sinfo = { 0 };
        if (0 == AVIStreamInfo(m_pAviStream, &sinfo, sizeof(AVISTREAMINFO)) && sinfo.fccType == streamtypeVIDEO) {
            rgy_reduce(sinfo.dwRate, sinfo.dwScale);
            m_inputVideoInfo.srcWidth = sinfo.rcFrame.right - sinfo.rcFrame.left;
            m_inputVideoInfo.srcHeight = sinfo.rcFrame.bottom - sinfo.rcFrame.top;
            if (!rgy_rational<int>(m_inputVideoInfo.fpsN, m_inputVideoInfo.fpsD).is_valid()) {
                m_inputVideoInfo.fpsN = sinfo.dwRate;
                m_inputVideoInfo.fpsD = sinfo.dwScale;
            }
            if (m_inputVideoInfo.frames == 0) {
                m_inputVideoInfo.frames = std::numeric_limits<decltype(m_inputVideoInfo.frames)>::max();
            }
            m_inputVideoInfo.frames = std::min(m_inputVideoInfo.frames, (int)(sinfo.dwLength - sinfo.dwStart));

            m_inputCsp = codec_fcc_to_rgy(sinfo.fccHandler);

            char temp[5] = { 0 };
            memcpy(temp, &sinfo.fccHandler, sizeof(sinfo.fccHandler));
            strFcc = char_to_tstring(temp);
            break;
        }
        AVIStreamRelease(m_pAviStream);
        m_pAviStream = NULL;
    }
    if (m_pAviStream == NULL) {
        AddMessage(RGY_LOG_ERROR, _T("failed to get valid stream from avi file.\n"));
        return RGY_ERR_INVALID_HANDLE;
    }
    AddMessage(RGY_LOG_DEBUG, _T("found video stream from avi file.\n"));

    if (   m_inputCsp == RGY_CSP_YUY2
        || m_inputCsp == RGY_CSP_YV12) {
        //何もしない
    } else {
        BITMAPINFOHEADER bih[4] = {
            { sizeof(BITMAPINFOHEADER), 0, 0, 1, 12, FCC('YV12'), m_inputVideoInfo.srcWidth * m_inputVideoInfo.srcHeight * 3/2, 0, 0, 0, 0 },
            { sizeof(BITMAPINFOHEADER), 0, 0, 1, 16, FCC('YUY2'), m_inputVideoInfo.srcWidth * m_inputVideoInfo.srcHeight * 2,   0, 0, 0, 0 },
            { sizeof(BITMAPINFOHEADER), 0, 0, 1, 24, BI_RGB,      m_inputVideoInfo.srcWidth * m_inputVideoInfo.srcHeight * 3,   0, 0, 0, 0 },
            { sizeof(BITMAPINFOHEADER), 0, 0, 1, 32, BI_RGB,      m_inputVideoInfo.srcWidth * m_inputVideoInfo.srcHeight * 3,   0, 0, 0, 0 }
        };
        for (int i = 0; i < _countof(bih); i++) {
            if (NULL == (m_pGetFrame = AVIStreamGetFrameOpen(m_pAviStream, &bih[i]))) {
                continue;
            }
            if (bih[i].biCompression == BI_RGB) {
                m_inputCsp = (bih[i].biBitCount == 24) ? RGY_CSP_RGB24R : RGY_CSP_RGB32R;
            } else {
                m_inputCsp = codec_fcc_to_rgy(bih[i].biCompression);
                if (m_inputCsp == RGY_CSP_NA) {
                    AddMessage(RGY_LOG_ERROR, _T("Invalid Color format.\n"));
                    return RGY_ERR_INVALID_COLOR_FORMAT;
                }
            }
            break;
        }

        if (m_pGetFrame == nullptr) {
            if (   nullptr == (m_pGetFrame = AVIStreamGetFrameOpen(m_pAviStream, NULL))
                && nullptr == (m_pGetFrame = AVIStreamGetFrameOpen(m_pAviStream, (BITMAPINFOHEADER *)AVIGETFRAMEF_BESTDISPLAYFMT))) {
                AddMessage(RGY_LOG_ERROR, _T("\nfailed to decode avi file.\n"));
                return RGY_ERR_INVALID_HANDLE;
            }
            BITMAPINFOHEADER *bmpInfoHeader = (BITMAPINFOHEADER *)AVIStreamGetFrame(m_pGetFrame, 0);
            if (NULL == bmpInfoHeader || bmpInfoHeader->biCompression != 0) {
                AddMessage(RGY_LOG_ERROR, _T("\nfailed to decode avi file.\n"));
                return RGY_ERR_MORE_DATA;
            }

            m_inputCsp = (bmpInfoHeader->biBitCount == 24) ? RGY_CSP_RGB24R : RGY_CSP_RGB32R;
        }
    }

    switch (m_inputCsp) {
    case RGY_CSP_YUY2:   m_nYPitchMultiplizer = 2; break;
    case RGY_CSP_RGB24R: m_nYPitchMultiplizer = 3; break;
    case RGY_CSP_RGB32R: m_nYPitchMultiplizer = 4; break;
    case RGY_CSP_YV12:
    default: m_nYPitchMultiplizer = 1; break;
    }

    if (m_inputCsp == RGY_CSP_RGB32R) {
        m_inputVideoInfo.csp = RGY_CSP_RGB32;
    } else if (m_inputCsp == RGY_CSP_RGB24R) {
        m_inputVideoInfo.csp = (ENCODER_NVENC) ? RGY_CSP_RGB : RGY_CSP_RGB32;
    } else {
        m_inputVideoInfo.csp = RGY_CSP_NV12;
    }
    if (m_convert->getFunc(m_inputCsp, m_inputVideoInfo.csp, false, prm->simdCsp) == nullptr) {
        AddMessage(RGY_LOG_ERROR, _T("color conversion not supported: %s -> %s.\n"),
            RGY_CSP_NAMES[m_inputCsp], RGY_CSP_NAMES[m_inputVideoInfo.csp]);
        return RGY_ERR_INVALID_COLOR_FORMAT;
    }
    CreateInputInfo(tstring(_T("avi: ") + strFcc).c_str(), RGY_CSP_NAMES[m_convert->getFunc()->csp_from], RGY_CSP_NAMES[m_convert->getFunc()->csp_to], get_simd_str(m_convert->getFunc()->simd), &m_inputVideoInfo);
    AddMessage(RGY_LOG_DEBUG, m_inputInfo);
    *pInputInfo = m_inputVideoInfo;
    return RGY_ERR_NONE;
}

void RGYInputAvi::Close() {
    AddMessage(RGY_LOG_DEBUG, _T("Closing...\n"));
    if (m_pGetFrame) {
        AVIStreamGetFrameClose(m_pGetFrame);
    }
    if (m_pAviStream) {
        AVIStreamRelease(m_pAviStream);
    }
    if (m_pAviFile) {
        AVIFileRelease(m_pAviFile);
    }
    AVIFileExit();

    m_pAviFile = nullptr;
    m_pAviStream = nullptr;
    m_pGetFrame = nullptr;
    m_pBitmapInfoHeader = nullptr;
    m_nYPitchMultiplizer = 1;
    m_nBufSize = 0;
    m_pBuffer.reset();

    AddMessage(RGY_LOG_DEBUG, _T("Closed.\n"));
    m_encSatusInfo.reset();
}

RGY_ERR RGYInputAvi::LoadNextFrameInternal(RGYFrame *pSurface) {
    if ((int)m_encSatusInfo->m_sData.frameIn >= m_inputVideoInfo.frames
        //m_encSatusInfo->m_nInputFramesがtrimの結果必要なフレーム数を大きく超えたら、エンコードを打ち切る
        //ちょうどのところで打ち切ると他のストリームに影響があるかもしれないので、余分に取得しておく
        || getVideoTrimMaxFramIdx() < (int)m_encSatusInfo->m_sData.frameIn - TRIM_OVERREAD_FRAMES) {
        return RGY_ERR_MORE_DATA;
    }

    uint8_t *ptr_src = nullptr;
    if (m_pGetFrame) {
        if (nullptr == (ptr_src = (uint8_t *)AVIStreamGetFrame(m_pGetFrame, m_encSatusInfo->m_sData.frameIn))) {
            return RGY_ERR_MORE_DATA;
        }
        ptr_src += sizeof(BITMAPINFOHEADER);
    } else {
        uint32_t required_bufsize = m_inputVideoInfo.srcWidth * m_inputVideoInfo.srcHeight * 3;
        if (m_nBufSize < required_bufsize) {
            m_pBuffer.reset();
            m_pBuffer = std::shared_ptr<uint8_t>((uint8_t *)_aligned_malloc(required_bufsize, 16), aligned_malloc_deleter());
            if (!m_pBuffer.get()) {
                return RGY_ERR_MEMORY_ALLOC;
            }
            m_nBufSize = required_bufsize;
        }
        LONG sizeRead = 0;
        if (0 != AVIStreamRead(m_pAviStream, m_encSatusInfo->m_sData.frameIn, 1, m_pBuffer.get(), (LONG)m_nBufSize, &sizeRead, NULL))
            return RGY_ERR_MORE_DATA;
        ptr_src = m_pBuffer.get();
    }

    void *dst_array[RGY_MAX_PLANES];
    pSurface->ptrArray(dst_array);
    const void *src_array[RGY_MAX_PLANES] = {
        ptr_src,
        ptr_src + m_inputVideoInfo.srcWidth * m_inputVideoInfo.srcHeight * 5 / 4,
        ptr_src + m_inputVideoInfo.srcWidth * m_inputVideoInfo.srcHeight,
        nullptr
    };

    m_convert->run((m_inputVideoInfo.picstruct & RGY_PICSTRUCT_INTERLACED) ? 1 : 0,
        dst_array, src_array,
        m_inputVideoInfo.srcWidth, m_inputVideoInfo.srcWidth * m_nYPitchMultiplizer, m_inputVideoInfo.srcWidth/2, pSurface->pitch(),
        m_inputVideoInfo.srcHeight, m_inputVideoInfo.srcHeight, m_inputVideoInfo.crop.c);

    m_encSatusInfo->m_sData.frameIn++;
    // display update
    return m_encSatusInfo->UpdateDisplay();
}

#endif //ENABLE_AVI_READER
