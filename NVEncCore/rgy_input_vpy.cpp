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

#include "rgy_input_vpy.h"
#include "rgy_vapoursynth_wrapper.h"
#include "rgy_filesystem.h"
#if ENABLE_VAPOURSYNTH_READER
#include <algorithm>
#include <sstream>
#include <map>
#include <fstream>
#include "rgy_codepage.h"
#include "rgy_util.h"


RGYInputVpyPrm::RGYInputVpyPrm(RGYInputPrm base) :
    RGYInputPrm(base),
    vsdir(),
    seekRatio(0.0f) {

}

RGYInputVpy::RGYInputVpy() :
    m_pAsyncBuffer(),
    m_hAsyncEventFrameSetFin(),
    m_hAsyncEventFrameSetStart(),
    m_bAbortAsync(false),
    m_nCopyOfInputFrames(0),
    m_vs(),
    m_asyncThreads(0),
    m_asyncFrames(0),
    m_startFrame(0) {
    memset(m_pAsyncBuffer, 0, sizeof(m_pAsyncBuffer));
    memset(m_hAsyncEventFrameSetFin,   0, sizeof(m_hAsyncEventFrameSetFin));
    memset(m_hAsyncEventFrameSetStart, 0, sizeof(m_hAsyncEventFrameSetStart));
    m_readerName = _T("vpy");
}

RGYInputVpy::~RGYInputVpy() {
    Close();
}

// VapourSynth v3/v4 specific initialization is handled by RGYVapourSynthWrapper.

int RGYInputVpy::initAsyncEvents() {
    for (int i = 0; i < _countof(m_hAsyncEventFrameSetFin); i++) {
        if (   NULL == (m_hAsyncEventFrameSetFin[i]   = CreateEvent(NULL, FALSE, FALSE, NULL))
            || NULL == (m_hAsyncEventFrameSetStart[i] = CreateEvent(NULL, FALSE, TRUE,  NULL)))
            return 1;
    }
    return 0;
}

void RGYInputVpy::closeAsyncEvents() {
    m_bAbortAsync = true;
    if (m_vs) {
        for (int i_frame = m_nCopyOfInputFrames; i_frame < m_asyncFrames; i_frame++) {
            const void *src_frame = getFrameFromAsyncBuffer(i_frame);
            if (src_frame) {
                m_vs->freeFrame(src_frame);
            }
        }
    }
    for (int i = 0; i < _countof(m_hAsyncEventFrameSetFin); i++) {
        if (m_hAsyncEventFrameSetFin[i])
            CloseEvent(m_hAsyncEventFrameSetFin[i]);
        if (m_hAsyncEventFrameSetStart[i])
            CloseEvent(m_hAsyncEventFrameSetStart[i]);
    }
    memset(m_hAsyncEventFrameSetFin,   0, sizeof(m_hAsyncEventFrameSetFin));
    memset(m_hAsyncEventFrameSetStart, 0, sizeof(m_hAsyncEventFrameSetStart));
    m_bAbortAsync = false;
}

static void vapourSynthFrameDoneThunk(void *userData, const void *frame, int n, const char *errorMsg) {
    (void)errorMsg;
    reinterpret_cast<RGYInputVpy*>(userData)->setFrameToAsyncBuffer(n, frame);
}

void RGYInputVpy::setFrameToAsyncBuffer(int n, const void* f) {
    WaitForSingleObject(m_hAsyncEventFrameSetStart[n & (ASYNC_BUFFER_SIZE-1)], INFINITE);
    m_pAsyncBuffer[n & (ASYNC_BUFFER_SIZE-1)] = f;
    SetEvent(m_hAsyncEventFrameSetFin[n & (ASYNC_BUFFER_SIZE-1)]);

    if (m_vs && m_asyncFrames < m_inputVideoInfo.frames && !m_bAbortAsync) {
        m_vs->getFrameAsync(m_asyncFrames, vapourSynthFrameDoneThunk, this);
        m_asyncFrames++;
    }
}

#pragma warning(push)
#pragma warning(disable:4127) //warning C4127: 条件式が定数です。
RGY_ERR RGYInputVpy::Init(const TCHAR *strFileName, VideoInfo *pInputInfo, const RGYInputPrm *prm) {
    m_inputVideoInfo = *pInputInfo;

    auto vpyPrm = reinterpret_cast<const RGYInputVpyPrm *>(prm);
    m_vs = CreateVapourSynthWrapper(vpyPrm->vsdir, m_printMes.get());
    if (!m_vs) return RGY_ERR_NULL_PTR;

    m_convert = std::make_unique<RGYConvertCSP>((m_inputVideoInfo.type == RGY_INPUT_FMT_VPY_MT) ? 1 : prm->threadCsp, prm->threadParamCsp);

    //ファイルデータ読み込み
    std::ifstream inputFile(strFileName);
    if (inputFile.bad()) {
        AddMessage(RGY_LOG_ERROR, _T("Failed to open vpy file \"%s\".\n"), strFileName);
        return RGY_ERR_FILE_OPEN;
    }
    AddMessage(RGY_LOG_DEBUG, _T("Opened file \"%s\""), strFileName);
    std::istreambuf_iterator<char> data_begin(inputFile);
    std::istreambuf_iterator<char> data_end;
    std::string script_data = std::string(data_begin, data_end);
    inputFile.close();

    if (initAsyncEvents()) {
        AddMessage(RGY_LOG_ERROR, _T("Failed to initialize async events.\n"));
        return RGY_ERR_NULL_PTR;
    }
    const auto filename_utf8 = tchar_to_string(strFileName, CODE_PAGE_UTF8);
    if (m_vs->openScriptFromBuffer(script_data, filename_utf8) != RGY_ERR_NONE) {
        AddMessage(RGY_LOG_ERROR, _T("VapourSynth script error.\n"));
        return RGY_ERR_NULL_PTR;
    }
    const auto& vsvideoinfo = m_vs->videoInfo();

    if (vsvideoinfo.height <= 0 || vsvideoinfo.width <= 0) {
        AddMessage(RGY_LOG_ERROR, _T("Variable resolution is not supported.\n"));
        return RGY_ERR_INCOMPATIBLE_VIDEO_PARAM;
    }

    if (vsvideoinfo.numFrames == 0) {
        AddMessage(RGY_LOG_ERROR, _T("Length of input video is unknown.\n"));
        return RGY_ERR_INCOMPATIBLE_VIDEO_PARAM;
    }

    if (!vsvideoinfo.isYUV || !vsvideoinfo.isInteger || vsvideoinfo.bitsPerSample <= 0) {
        AddMessage(RGY_LOG_ERROR, _T("Invalid colorformat.\n"));
        return RGY_ERR_INVALID_COLOR_FORMAT;
    }

    struct CSPMap {
        int bits;
        int subW;
        int subH;
        RGY_CSP in, out;
        constexpr CSPMap(int b, int sw, int sh, RGY_CSP i, RGY_CSP o) : bits(b), subW(sw), subH(sh), in(i), out(o) {};
    };

    static constexpr auto valid_csp_list = make_array<CSPMap>(
        CSPMap( 8, 1, 1,  RGY_CSP_YV12,      RGY_CSP_NV12 ),
        CSPMap(10, 1, 1,  RGY_CSP_YV12_10,   RGY_CSP_P010 ),
        CSPMap(12, 1, 1,  RGY_CSP_YV12_12,   RGY_CSP_P010 ),
        CSPMap(14, 1, 1,  RGY_CSP_YV12_14,   RGY_CSP_P010 ),
        CSPMap(16, 1, 1,  RGY_CSP_YV12_16,   RGY_CSP_P010 ),
#if ENCODER_QSV || ENCODER_VCEENC
        CSPMap( 8, 1, 0,  RGY_CSP_YUV422,    RGY_CSP_NV12 ),
        CSPMap(10, 1, 0,  RGY_CSP_YUV422_10, RGY_CSP_P010 ),
        CSPMap(12, 1, 0,  RGY_CSP_YUV422_12, RGY_CSP_P010 ),
        CSPMap(14, 1, 0,  RGY_CSP_YUV422_14, RGY_CSP_P010 ),
        CSPMap(16, 1, 0,  RGY_CSP_YUV422_16, RGY_CSP_P010 ),
#else
        CSPMap( 8, 1, 0,  RGY_CSP_YUV422,    RGY_CSP_NV16 ),
        CSPMap(10, 1, 0,  RGY_CSP_YUV422_10, RGY_CSP_P210 ),
        CSPMap(12, 1, 0,  RGY_CSP_YUV422_12, RGY_CSP_P210 ),
        CSPMap(14, 1, 0,  RGY_CSP_YUV422_14, RGY_CSP_P210 ),
        CSPMap(16, 1, 0,  RGY_CSP_YUV422_16, RGY_CSP_P210 ),
#endif
        CSPMap( 8, 0, 0,  RGY_CSP_YUV444,    RGY_CSP_YUV444 ),
        CSPMap(10, 0, 0,  RGY_CSP_YUV444_10, RGY_CSP_YUV444_16 ),
        CSPMap(12, 0, 0,  RGY_CSP_YUV444_12, RGY_CSP_YUV444_16 ),
        CSPMap(14, 0, 0,  RGY_CSP_YUV444_14, RGY_CSP_YUV444_16 ),
        CSPMap(16, 0, 0,  RGY_CSP_YUV444_16, RGY_CSP_YUV444_16 )
    );

    const RGY_CSP prefered_csp = m_inputVideoInfo.csp;
    m_inputCsp = RGY_CSP_NA;
    for (const auto& csp : valid_csp_list) {
        if (csp.bits == vsvideoinfo.bitsPerSample && csp.subW == vsvideoinfo.subSamplingW && csp.subH == vsvideoinfo.subSamplingH) {
            m_inputCsp = csp.in;
            if (prefered_csp == RGY_CSP_NA) {
                //ロスレスの場合は、入力側で出力フォーマットを決める
                m_inputVideoInfo.csp = csp.out;
            } else {
                m_inputVideoInfo.csp = (m_convert->getFunc(m_inputCsp, prefered_csp, false, prm->simdCsp) != nullptr) ? prefered_csp : csp.out;
                //csp.outがYUV422に関しては可能ならcsp.outを優先する
                if (RGY_CSP_CHROMA_FORMAT[csp.out] == RGY_CHROMAFMT_YUV422
                    && m_convert->getFunc(m_inputCsp, csp.out, false, prm->simdCsp) != nullptr) {
                    m_inputVideoInfo.csp = csp.out;
                }
                //QSVではNV16->P010がサポートされていない
                if (ENCODER_QSV && m_inputVideoInfo.csp == RGY_CSP_NV16 && prefered_csp == RGY_CSP_P010) {
                    m_inputVideoInfo.csp = RGY_CSP_P210;
                }
                //なるべく軽いフォーマットでGPUに転送するように
                if (ENCODER_NVENC
                    && RGY_CSP_BIT_PER_PIXEL[csp.out] < RGY_CSP_BIT_PER_PIXEL[prefered_csp]
                    && m_convert->getFunc(m_inputCsp, csp.out, false, prm->simdCsp) != nullptr) {
                    m_inputVideoInfo.csp = csp.out;
                }
            }
            if (m_convert->getFunc(m_inputCsp, m_inputVideoInfo.csp, false, prm->simdCsp) == nullptr && m_inputCsp == RGY_CSP_YUY2) {
                //YUY2用の特別処理
                m_inputVideoInfo.csp = RGY_CSP_CHROMA_FORMAT[csp.out] == RGY_CHROMAFMT_YUV420 ? RGY_CSP_NV12 : RGY_CSP_YUV444;
                m_convert->getFunc(m_inputCsp, m_inputVideoInfo.csp, false, prm->simdCsp);
            }
            break;
        }
    }

    if (m_inputCsp == RGY_CSP_NA) {
        AddMessage(RGY_LOG_ERROR, _T("invalid colorformat.\n"));
        return RGY_ERR_INVALID_COLOR_FORMAT;
    }
    if (m_convert->getFunc() == nullptr) {
        AddMessage(RGY_LOG_ERROR, _T("color conversion not supported: %s -> %s.\n"),
            RGY_CSP_NAMES[m_inputCsp], RGY_CSP_NAMES[m_inputVideoInfo.csp]);
        return RGY_ERR_INVALID_COLOR_FORMAT;
    }

    if (vsvideoinfo.fpsNum <= 0 || vsvideoinfo.fpsDen <= 0) {
        AddMessage(RGY_LOG_ERROR, _T("Invalid framerate.\n"));
        return RGY_ERR_INCOMPATIBLE_VIDEO_PARAM;
    }

    m_inputVideoInfo.srcWidth = vsvideoinfo.width;
    m_inputVideoInfo.srcHeight = vsvideoinfo.height;
    if (!rgy_rational<int>(m_inputVideoInfo.fpsN, m_inputVideoInfo.fpsD).is_valid()) {
        const auto fps_gcd = rgy_gcd((int)vsvideoinfo.fpsNum, (int)vsvideoinfo.fpsDen);
        m_inputVideoInfo.fpsN = (int)(vsvideoinfo.fpsNum / fps_gcd);
        m_inputVideoInfo.fpsD = (int)(vsvideoinfo.fpsDen / fps_gcd);
    }
    if (m_inputVideoInfo.frames == 0) {
        m_inputVideoInfo.frames = std::numeric_limits<decltype(m_inputVideoInfo.frames)>::max();
    }
    m_inputVideoInfo.frames = std::min(m_inputVideoInfo.frames, vsvideoinfo.numFrames);
    m_inputVideoInfo.bitdepth = RGY_CSP_BIT_DEPTH[m_inputVideoInfo.csp];
    if (cspShiftUsed(m_inputVideoInfo.csp) && RGY_CSP_BIT_DEPTH[m_inputVideoInfo.csp] > RGY_CSP_BIT_DEPTH[m_inputCsp]) {
        m_inputVideoInfo.bitdepth = RGY_CSP_BIT_DEPTH[m_inputCsp];
    }

    m_startFrame = 0;
    if (vpyPrm->seekRatio > 0.0f) {
        m_startFrame = (int)(vpyPrm->seekRatio * m_inputVideoInfo.frames);
    }
    m_asyncThreads = vsvideoinfo.numFrames - m_startFrame;
    m_asyncThreads = (std::min)(m_asyncThreads, vsvideoinfo.numThreads);
    m_asyncThreads = (std::min)(m_asyncThreads, ASYNC_BUFFER_SIZE-1);
    if (m_inputVideoInfo.type != RGY_INPUT_FMT_VPY_MT) {
        m_asyncThreads = 1;
    }
    m_asyncFrames = m_startFrame + m_asyncThreads;

    for (int i = m_startFrame; i < m_asyncFrames; i++) {
        m_vs->getFrameAsync(i, vapourSynthFrameDoneThunk, this);
    }

    tstring vs_ver = _T("VapourSynth");
    vs_ver += (m_vs->apiMajor() >= 4) ? _T("4") : _T("");
    if (m_inputVideoInfo.type == RGY_INPUT_FMT_VPY_MT) vs_ver += _T("MT");

    CreateInputInfo(vs_ver.c_str(), RGY_CSP_NAMES[m_convert->getFunc()->csp_from], RGY_CSP_NAMES[m_convert->getFunc()->csp_to], get_simd_str(m_convert->getFunc()->simd), &m_inputVideoInfo);
    AddMessage(RGY_LOG_DEBUG, m_inputInfo);
    *pInputInfo = m_inputVideoInfo;
    return RGY_ERR_NONE;
}
#pragma warning(pop)

int64_t RGYInputVpy::GetVideoFirstKeyPts() const {
    auto inputFps = rgy_rational<int>(m_inputVideoInfo.fpsN, m_inputVideoInfo.fpsD);
    return rational_rescale(m_startFrame, getInputTimebase().inv(), inputFps);
}

void RGYInputVpy::Close() {
    AddMessage(RGY_LOG_DEBUG, _T("Closing...\n"));
    closeAsyncEvents();
    if (m_vs) {
        m_vs->close();
    }

    m_bAbortAsync = false;
    m_nCopyOfInputFrames = 0;

    m_vs.reset();
    m_asyncThreads = 0;
    m_asyncFrames = 0;
    m_encSatusInfo.reset();
    AddMessage(RGY_LOG_DEBUG, _T("Closed.\n"));
}

RGY_ERR RGYInputVpy::LoadNextFrameInternal(RGYFrame *pSurface) {
    if ((int)(m_encSatusInfo->m_sData.frameIn + m_startFrame) >= m_inputVideoInfo.frames
        //m_encSatusInfo->m_nInputFramesがtrimの結果必要なフレーム数を大きく超えたら、エンコードを打ち切る
        //ちょうどのところで打ち切ると他のストリームに影響があるかもしれないので、余分に取得しておく
        || getVideoTrimMaxFramIdx() < (int)(m_encSatusInfo->m_sData.frameIn + m_startFrame) - TRIM_OVERREAD_FRAMES) {
        return RGY_ERR_MORE_DATA;
    }

    const void *src_frame = getFrameFromAsyncBuffer(m_encSatusInfo->m_sData.frameIn + m_startFrame);
    if (src_frame == nullptr) {
        return RGY_ERR_MORE_DATA;
    }
    if (pSurface) {
        void *dst_array[RGY_MAX_PLANES];
        pSurface->ptrArray(dst_array);
        const void *src_array[RGY_MAX_PLANES] = { m_vs->getReadPtr(src_frame, 0), m_vs->getReadPtr(src_frame, 1), m_vs->getReadPtr(src_frame, 2), nullptr };
        m_convert->run((m_inputVideoInfo.picstruct & RGY_PICSTRUCT_INTERLACED) ? 1 : 0,
            dst_array, src_array,
            m_inputVideoInfo.srcWidth, (int)m_vs->getStride(src_frame, 0), (int)m_vs->getStride(src_frame, 1),
            pSurface->pitch(), pSurface->pitch(RGY_PLANE_C), m_inputVideoInfo.srcHeight, m_inputVideoInfo.srcHeight, m_inputVideoInfo.crop.c);

        auto inputFps = rgy_rational<int>(m_inputVideoInfo.fpsN, m_inputVideoInfo.fpsD);
        pSurface->setDuration(rational_rescale(1, getInputTimebase().inv(), inputFps));
        pSurface->setTimestamp(rational_rescale(m_encSatusInfo->m_sData.frameIn + m_startFrame, getInputTimebase().inv(), inputFps));
    }
    m_vs->freeFrame(src_frame);

    m_encSatusInfo->m_sData.frameIn++;
    m_nCopyOfInputFrames = m_encSatusInfo->m_sData.frameIn + m_startFrame;

    return m_encSatusInfo->UpdateDisplay();
}

#endif //ENABLE_VAPOURSYNTH_READER
