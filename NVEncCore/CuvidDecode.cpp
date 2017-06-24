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

#include "NVEncCore.h"
#include "CuvidDecode.h"
#include "helper_cuda.h"
#include "NVEncUtil.h"
#if ENABLE_AVSW_READER

bool check_if_nvcuvid_dll_available() {
    //check for nvcuvid.dll
    HMODULE hModule = LoadLibrary(_T("nvcuvid.dll"));
    if (hModule == NULL)
        return false;
    FreeLibrary(hModule);
    return true;
}

CodecCsp getHWDecCodecCsp() {

    CUVIDDECODECAPS caps_test;
    memset(&caps_test, 0, sizeof(caps_test));

    static const auto test_target = make_array<RGY_CSP>(
        RGY_CSP_NV12,
        RGY_CSP_YV12,
        RGY_CSP_YV12_09,
        RGY_CSP_YV12_10,
        RGY_CSP_YV12_12,
        RGY_CSP_YV12_14,
        RGY_CSP_YV12_16,
        RGY_CSP_YUV444,
        RGY_CSP_YUV444_09,
        RGY_CSP_YUV444_10,
        RGY_CSP_YUV444_12,
        RGY_CSP_YUV444_14,
        RGY_CSP_YUV444_16
        );

    CodecCsp HWDecCodecCsp;

    for (int i = 0; i < _countof(HW_DECODE_LIST); i++) {
        std::vector<RGY_CSP> supported_csp;
        caps_test.eCodecType = codec_rgy_to_enc(HW_DECODE_LIST[i].rgy_codec);
        for (auto csp : test_target) {
            caps_test.nBitDepthMinus8 = RGY_CSP_BIT_DEPTH[csp] - 8;
            caps_test.eChromaFormat = chromafmt_rgy_to_enc(RGY_CSP_CHROMA_FORMAT[csp]);
            auto ret = cuvidGetDecoderCaps(&caps_test);
            if (ret == CUDA_SUCCESS && caps_test.bIsSupported) {
                supported_csp.push_back(csp);
            }
        }
        if (supported_csp.size() > 0) {
            HWDecCodecCsp[HW_DECODE_LIST[i].rgy_codec] = supported_csp;
        }
    }

    //もし、なんらかの原因で正常に取得されていなければ、
    //基本的なコーデックはデコード可能だと返す
    std::vector<RGY_CODEC> basic_codec_list = { RGY_CODEC_H264, RGY_CODEC_MPEG2, RGY_CODEC_MPEG1 };
    std::vector<RGY_CSP> basic_csp_list = { RGY_CSP_NV12, RGY_CSP_YV12 };
    for (auto codec : basic_codec_list) {
        if (HWDecCodecCsp.count(codec) == 0) {
            HWDecCodecCsp[codec] = basic_csp_list;
        }
    }
    return HWDecCodecCsp;
}

static int CUDAAPI HandleVideoData(void *pUserData, CUVIDSOURCEDATAPACKET *pPacket) {
    assert(pUserData);
    return ((CuvidDecode*)pUserData)->DecVideoData(pPacket);
}

static int CUDAAPI HandleVideoSequence(void *pUserData, CUVIDEOFORMAT *pFormat) {
    assert(pUserData);
    return ((CuvidDecode*)pUserData)->DecVideoSequence(pFormat);
}

static int CUDAAPI HandlePictureDecode(void *pUserData, CUVIDPICPARAMS *pPicParams) {
    assert(pUserData);
    return ((CuvidDecode*)pUserData)->DecPictureDecode(pPicParams);
}

static int CUDAAPI HandlePictureDisplay(void *pUserData, CUVIDPARSERDISPINFO *pPicParams) {
    assert(pUserData);
    return ((CuvidDecode*)pUserData)->DecPictureDisplay(pPicParams);
}

CuvidDecode::CuvidDecode() :
    m_pFrameQueue(nullptr), m_decodedFrames(0), m_videoParser(nullptr), m_videoDecoder(nullptr),
    m_ctxLock(nullptr), m_pPrintMes(), m_bIgnoreDynamicFormatChange(false), m_bError(false), m_videoInfo(), m_nDecType(0) {
    memset(&m_videoDecodeCreateInfo, 0, sizeof(m_videoDecodeCreateInfo));
    memset(&m_videoFormatEx, 0, sizeof(m_videoFormatEx));
}

CuvidDecode::~CuvidDecode() {
    CloseDecoder();
}

int CuvidDecode::DecVideoData(CUVIDSOURCEDATAPACKET *pPacket) {
    CUresult curesult = CUDA_SUCCESS;
    //cuvidCtxLock(m_ctxLock, 0);
    __try {
        curesult = cuvidParseVideoData(m_videoParser, pPacket);
    } __except(1) {
        AddMessage(RGY_LOG_ERROR, _T("cuvidParseVideoData error\n"));
        curesult = CUDA_ERROR_UNKNOWN;
    }
    //cuvidCtxUnlock(m_ctxLock, 0);
    if (curesult != CUDA_SUCCESS) {
        m_bError = true;
    }
    return (curesult == CUDA_SUCCESS);
}

int CuvidDecode::DecPictureDecode(CUVIDPICPARAMS *pPicParams) {
    AddMessage(RGY_LOG_TRACE, _T("DecPictureDecode idx: %d\n"), pPicParams->CurrPicIdx);
    m_pFrameQueue->waitUntilFrameAvailable(pPicParams->CurrPicIdx);
    CUresult curesult = CUDA_SUCCESS;
    //cuvidCtxLock(m_ctxLock, 0);
    __try {
        curesult = cuvidDecodePicture(m_videoDecoder, pPicParams);
    } __except(1) {
        AddMessage(RGY_LOG_ERROR, _T("cuvidDecodePicture error\n"));
        curesult = CUDA_ERROR_UNKNOWN;
    }
    //cuvidCtxUnlock(m_ctxLock, 0);
    if (curesult != CUDA_SUCCESS) {
        m_bError = true;
    }
    return (curesult == CUDA_SUCCESS);
}

int CuvidDecode::DecVideoSequence(CUVIDEOFORMAT *pFormat) {
    AddMessage(RGY_LOG_TRACE, _T("DecVideoSequence\n"));
    if (   (pFormat->codec         != m_videoDecodeCreateInfo.CodecType)
        || (pFormat->chroma_format != m_videoDecodeCreateInfo.ChromaFormat)) {
        if (m_videoDecodeCreateInfo.CodecType != cudaVideoCodec_NumCodecs) {
            AddMessage(RGY_LOG_DEBUG, _T("dynamic video format changing detected\n"));
        }
        CreateDecoder(pFormat);
        return 1;
    }
    if (   (pFormat->coded_width   != m_videoDecodeCreateInfo.ulWidth)
        || (pFormat->coded_height  != m_videoDecodeCreateInfo.ulHeight)) {
        AddMessage(RGY_LOG_DEBUG, _T("dynamic video format changing detected\n"));
        m_videoDecodeCreateInfo.CodecType    = pFormat->codec;
        m_videoDecodeCreateInfo.ulWidth      = pFormat->coded_width;
        m_videoDecodeCreateInfo.ulHeight     = pFormat->coded_height;
        m_videoDecodeCreateInfo.ChromaFormat = pFormat->chroma_format;
        if (pFormat->coded_width != m_videoDecodeCreateInfo.ulWidth && pFormat->coded_height != m_videoDecodeCreateInfo.ulHeight) {
            memcpy(&m_videoDecodeCreateInfo.display_area, &pFormat->display_area, sizeof(pFormat->display_area));
        }
        return 0;
    }
    return 1;
}

int CuvidDecode::DecPictureDisplay(CUVIDPARSERDISPINFO *pPicParams) {
    AddMessage(RGY_LOG_TRACE, _T("DecPictureDisplay idx: %d, %I64d\n"), pPicParams->picture_index, pPicParams->timestamp);
    m_pFrameQueue->enqueue(pPicParams);
    m_decodedFrames++;

    return 1;
}

void CuvidDecode::CloseDecoder() {
    if (m_videoDecoder) {
        cuvidDestroyDecoder(m_videoDecoder);
        m_videoDecoder = nullptr;
    }
    if (m_videoParser) {
        cuvidDestroyVideoParser(m_videoParser);
        m_videoParser = nullptr;
    }
    m_ctxLock = nullptr;
    m_pPrintMes.reset();
    if (m_pFrameQueue) {
        delete m_pFrameQueue;
        m_pFrameQueue = nullptr;
    }
    m_decodedFrames = 0;
    m_bError = false;
}

CUresult CuvidDecode::CreateDecoder() {
    CUresult curesult = CUDA_SUCCESS;
    __try {
        curesult = cuvidCreateDecoder(&m_videoDecoder, &m_videoDecodeCreateInfo);
    } __except (1) {
        AddMessage(RGY_LOG_ERROR, _T("cuvidCreateDecoder error\n"));
        curesult = CUDA_ERROR_UNKNOWN;
    }
    return curesult;
}


CUresult CuvidDecode::CreateDecoder(CUVIDEOFORMAT *pFormat) {
    if (m_videoDecoder) {
        cuvidDestroyDecoder(m_videoDecoder);
        m_videoDecoder = nullptr;
    }

    m_videoDecodeCreateInfo.CodecType = pFormat->codec;
    m_videoDecodeCreateInfo.ChromaFormat = pFormat->chroma_format;
    m_videoDecodeCreateInfo.ulWidth   = pFormat->coded_width;
    m_videoDecodeCreateInfo.ulHeight  = pFormat->coded_height;
    m_videoDecodeCreateInfo.bitDepthMinus8 = (RGY_CSP_BIT_DEPTH[m_videoInfo.csp] - m_videoInfo.shift) - 8;

    if (m_videoInfo.dstWidth > 0 && m_videoInfo.dstHeight > 0) {
        m_videoDecodeCreateInfo.ulTargetWidth  = m_videoInfo.dstWidth;
        m_videoDecodeCreateInfo.ulTargetHeight = m_videoInfo.dstHeight;
    } else {
        m_videoDecodeCreateInfo.ulTargetWidth  = m_videoInfo.srcWidth - m_videoInfo.crop.e.right - m_videoInfo.crop.e.left;
        m_videoDecodeCreateInfo.ulTargetHeight = m_videoInfo.srcHeight - m_videoInfo.crop.e.up - m_videoInfo.crop.e.bottom;
    }
    m_videoDecodeCreateInfo.target_rect.left = 0;
    m_videoDecodeCreateInfo.target_rect.top = 0;
    m_videoDecodeCreateInfo.target_rect.right = (short)m_videoDecodeCreateInfo.ulTargetWidth;
    m_videoDecodeCreateInfo.target_rect.bottom = (short)m_videoDecodeCreateInfo.ulTargetHeight;

    m_videoDecodeCreateInfo.display_area.left   = (short)(pFormat->display_area.left + m_videoInfo.crop.e.left);
    m_videoDecodeCreateInfo.display_area.top    = (short)(pFormat->display_area.top + m_videoInfo.crop.e.up);
    m_videoDecodeCreateInfo.display_area.right  = (short)(pFormat->display_area.right - m_videoInfo.crop.e.right);
    m_videoDecodeCreateInfo.display_area.bottom = (short)(pFormat->display_area.bottom - m_videoInfo.crop.e.bottom);

    cuvidCtxLock(m_ctxLock, 0);
    m_videoDecodeCreateInfo.CodecType = pFormat->codec;
    CUresult curesult = CreateDecoder();
    cuvidCtxUnlock(m_ctxLock, 0);
    if (CUDA_SUCCESS != curesult) {
        AddMessage(RGY_LOG_ERROR, _T("Failed cuvidCreateDecoder %d (%s)\n"), curesult, char_to_tstring(_cudaGetErrorEnum(curesult)).c_str());
        return curesult;
    }
    AddMessage(RGY_LOG_DEBUG, _T("created decoder (mode: %s)\n"), get_chr_from_value(list_cuvid_mode, m_nDecType));
    return curesult;
}

CUresult CuvidDecode::InitDecode(CUvideoctxlock ctxLock, const VideoInfo *input, const VppParam *vpp, shared_ptr<RGYLog> pLog, int nDecType, bool bCuvidResize, bool ignoreDynamicFormatChange) {
    //初期化
    CloseDecoder();

    m_videoInfo = *input;
    if (!bCuvidResize) {
        m_videoInfo.dstWidth = 0;
        m_videoInfo.dstHeight = 0;
    }
    m_nDecType = nDecType;
    m_pPrintMes = pLog;
    m_bIgnoreDynamicFormatChange = ignoreDynamicFormatChange;
    m_deinterlaceMode = vpp->deinterlace;

    if (!check_if_nvcuvid_dll_available()) {
        AddMessage(RGY_LOG_ERROR, _T("nvcuvid.dll does not exist.\n"));
        return CUDA_ERROR_NOT_FOUND;
    }
    AddMessage(RGY_LOG_DEBUG, _T("nvcuvid.dll available\n"));

    if (!ctxLock) {
        AddMessage(RGY_LOG_ERROR, _T("invalid ctxLock.\n"));
        return CUDA_ERROR_INVALID_VALUE;
    }

    m_ctxLock = ctxLock;

    if (nullptr == (m_pFrameQueue = new CUVIDFrameQueue(m_ctxLock))) {
        AddMessage(RGY_LOG_ERROR, _T("Failed to alloc frame queue for decoder.\n"));
        return CUDA_ERROR_OUT_OF_MEMORY;
    }
    m_pFrameQueue->init(input->srcWidth, input->srcHeight);
    AddMessage(RGY_LOG_DEBUG, _T("created frame queue\n"));

    //init video parser
    memset(&m_videoFormatEx, 0, sizeof(CUVIDEOFORMATEX));
    if (input->codecExtra && input->codecExtraSize) {
        if (input->codecExtraSize > sizeof(m_videoFormatEx.raw_seqhdr_data)) {
            AddMessage(RGY_LOG_ERROR, _T("Parsed header too large!\n"));
            return CUDA_ERROR_INVALID_VALUE;
        }
        memcpy(m_videoFormatEx.raw_seqhdr_data, input->codecExtra, input->codecExtraSize);
        m_videoFormatEx.format.seqhdr_data_length = input->codecExtraSize;
    }

    CUVIDPARSERPARAMS oVideoParserParameters;
    memset(&oVideoParserParameters, 0, sizeof(CUVIDPARSERPARAMS));
    oVideoParserParameters.CodecType              = codec_rgy_to_enc(input->codec);
    oVideoParserParameters.ulMaxNumDecodeSurfaces = FrameQueue::cnMaximumSize;
    oVideoParserParameters.ulMaxDisplayDelay      = 1;
    oVideoParserParameters.pUserData              = this;
    oVideoParserParameters.pfnSequenceCallback    = HandleVideoSequence;
    oVideoParserParameters.pfnDecodePicture       = HandlePictureDecode;
    oVideoParserParameters.pfnDisplayPicture      = HandlePictureDisplay;
    oVideoParserParameters.pExtVideoInfo          = &m_videoFormatEx;

    CUresult curesult = CUDA_SUCCESS;
    if (CUDA_SUCCESS != (curesult = cuvidCreateVideoParser(&m_videoParser, &oVideoParserParameters))) {
        AddMessage(RGY_LOG_ERROR, _T("Failed cuvidCreateVideoParser %d (%s)\n"), curesult, char_to_tstring(_cudaGetErrorEnum(curesult)).c_str());
        return curesult;
    }
    AddMessage(RGY_LOG_DEBUG, _T("created video parser\n"));

    cuvidCtxLock(m_ctxLock, 0);
    memset(&m_videoDecodeCreateInfo, 0, sizeof(CUVIDDECODECREATEINFO));
    m_videoDecodeCreateInfo.CodecType = cudaVideoCodec_NV12; // codec_rgy_to_enc(input->codec);
    m_videoDecodeCreateInfo.ulWidth   = input->codedWidth  ? input->codedWidth  : input->srcWidth;
    m_videoDecodeCreateInfo.ulHeight  = input->codedHeight ? input->codedHeight : input->srcHeight;
    m_videoDecodeCreateInfo.ulNumDecodeSurfaces = FrameQueue::cnMaximumSize;

    m_videoDecodeCreateInfo.ChromaFormat = cudaVideoChromaFormat_420;
    m_videoDecodeCreateInfo.OutputFormat = (input->csp == RGY_CSP_P010) ? cudaVideoSurfaceFormat_P016 : cudaVideoSurfaceFormat_NV12;
    m_videoDecodeCreateInfo.DeinterlaceMode = vpp->deinterlace;

    if (m_videoInfo.dstWidth > 0 && m_videoInfo.dstHeight > 0) {
        m_videoDecodeCreateInfo.ulTargetWidth  = m_videoInfo.dstWidth;
        m_videoDecodeCreateInfo.ulTargetHeight = m_videoInfo.dstHeight;
    } else {
        m_videoDecodeCreateInfo.ulTargetWidth  = m_videoInfo.srcWidth - input->crop.e.right - input->crop.e.left;
        m_videoDecodeCreateInfo.ulTargetHeight = m_videoInfo.srcHeight - input->crop.e.up - input->crop.e.bottom;
    }

    m_videoDecodeCreateInfo.display_area.left   = (short)input->crop.e.left;
    m_videoDecodeCreateInfo.display_area.top    = (short)input->crop.e.up;
    m_videoDecodeCreateInfo.display_area.right  = (short)(input->srcWidth - input->crop.e.right);
    m_videoDecodeCreateInfo.display_area.bottom = (short)(input->srcHeight - input->crop.e.bottom);

    m_videoDecodeCreateInfo.ulNumOutputSurfaces = 1;
    m_videoDecodeCreateInfo.ulCreationFlags = (nDecType == NV_ENC_AVCUVID_CUDA) ? cudaVideoCreate_PreferCUDA : cudaVideoCreate_PreferCUVID;
    m_videoDecodeCreateInfo.vidLock = m_ctxLock;
#if 0
    curesult = CreateDecoder();
    if (CUDA_SUCCESS != curesult) {
        AddMessage(RGY_LOG_ERROR, _T("Failed cuvidCreateDecoder %d (%s)\n"), curesult, char_to_tstring(_cudaGetErrorEnum(curesult)).c_str());
        return curesult;
    }
    AddMessage(RGY_LOG_DEBUG, _T("created decoder (mode: %s)\n"), get_chr_from_value(list_cuvid_mode, nDecType));

    if (m_videoFormatEx.raw_seqhdr_data && m_videoFormatEx.format.seqhdr_data_length) {
        if (CUDA_SUCCESS != (curesult = DecodePacket(m_videoFormatEx.raw_seqhdr_data, m_videoFormatEx.format.seqhdr_data_length, AV_NOPTS_VALUE, HW_NATIVE_TIMEBASE))) {
            AddMessage(RGY_LOG_ERROR, _T("Failed to decode header %d (%s).\n"), curesult, char_to_tstring(_cudaGetErrorEnum(curesult)).c_str());
            return curesult;
        }
    }
#else
    if (m_videoFormatEx.format.seqhdr_data_length > 0) {
        CUVIDSOURCEDATAPACKET pCuvidPacket;
        memset(&pCuvidPacket, 0, sizeof(pCuvidPacket));
        pCuvidPacket.payload = m_videoFormatEx.raw_seqhdr_data;
        pCuvidPacket.payload_size = m_videoFormatEx.format.seqhdr_data_length;
        curesult = cuvidParseVideoData(m_videoParser, &pCuvidPacket);
    }
#endif
    cuvidCtxUnlock(m_ctxLock, 0);
    AddMessage(RGY_LOG_DEBUG, _T("DecodePacket: success\n"));
    return curesult;
}

CUresult CuvidDecode::FlushParser() {
    CUVIDSOURCEDATAPACKET pCuvidPacket;
    memset(&pCuvidPacket, 0, sizeof(pCuvidPacket));

    pCuvidPacket.flags |= CUVID_PKT_ENDOFSTREAM;
    CUresult result = CUDA_SUCCESS;

    //cuvidCtxLock(m_ctxLock, 0);
    __try {
        result = cuvidParseVideoData(m_videoParser, &pCuvidPacket);
    } __except (1) {
        AddMessage(RGY_LOG_ERROR, _T("cuvidParseVideoData error\n"));
        result = CUDA_ERROR_UNKNOWN;
    }
    //cuvidCtxUnlock(m_ctxLock, 0);
    m_pFrameQueue->endDecode();
    return result;
}

CUresult CuvidDecode::DecodePacket(uint8_t *data, size_t nSize, int64_t timestamp, AVRational streamtimebase) {
    if (data == nullptr || nSize == 0) {
        return FlushParser();
    }

    CUVIDSOURCEDATAPACKET pCuvidPacket;
    memset(&pCuvidPacket, 0, sizeof(pCuvidPacket));
    pCuvidPacket.payload      = data;
    pCuvidPacket.payload_size = (uint32_t)nSize;
    CUresult result = CUDA_SUCCESS;

    if (timestamp != AV_NOPTS_VALUE) {
        pCuvidPacket.flags     |= CUVID_PKT_TIMESTAMP;
        pCuvidPacket.timestamp  = av_rescale_q(timestamp, streamtimebase, HW_NATIVE_TIMEBASE);
    }

    //cuvidCtxLock(m_ctxLock, 0);
    __try {
        result = cuvidParseVideoData(m_videoParser, &pCuvidPacket);
    } __except (1) {
        AddMessage(RGY_LOG_ERROR, _T("cuvidParseVideoData error\n"));
        result = CUDA_ERROR_UNKNOWN;
    }
    //cuvidCtxUnlock(m_ctxLock, 0);
    return result;
}

FrameInfo CuvidDecode::GetDecFrameInfo() {
    FrameInfo frame;
    frame.ptr = nullptr;
    frame.csp = m_videoInfo.csp;
    frame.width = m_videoDecodeCreateInfo.ulTargetWidth;
    frame.height = m_videoDecodeCreateInfo.ulTargetHeight;
    frame.pitch = 0; //この段階では取得できない、cuvidMapVideoFrameで取得
    frame.timestamp = (uint64_t)AV_NOPTS_VALUE;
    frame.deivce_mem = true;
    return frame;
}

#endif // #if ENABLE_AVSW_READER