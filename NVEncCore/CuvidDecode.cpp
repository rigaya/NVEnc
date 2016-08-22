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
#if ENABLE_AVCUVID_READER

bool check_if_nvcuvid_dll_available() {
    //check for nvcuvid.dll
    HMODULE hModule = LoadLibrary(_T("nvcuvid.dll"));
    if (hModule == NULL)
        return false;
    FreeLibrary(hModule);
    return true;
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
    m_ctxLock(nullptr), m_pPrintMes(), m_bIgnoreDynamicFormatChange(false), m_bError(false) {
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
        AddMessage(NV_LOG_ERROR, _T("cuvidParseVideoData error\n"));
        curesult = CUDA_ERROR_UNKNOWN;
    }
    //cuvidCtxUnlock(m_ctxLock, 0);
    if (curesult != CUDA_SUCCESS) {
        m_bError = true;
    }
    return (curesult == CUDA_SUCCESS);
}

int CuvidDecode::DecPictureDecode(CUVIDPICPARAMS *pPicParams) {
    AddMessage(NV_LOG_TRACE, _T("DecPictureDecode idx: %d\n"), pPicParams->CurrPicIdx);
    m_pFrameQueue->waitUntilFrameAvailable(pPicParams->CurrPicIdx);
    CUresult curesult = CUDA_SUCCESS;
    //cuvidCtxLock(m_ctxLock, 0);
    __try {
        curesult = cuvidDecodePicture(m_videoDecoder, pPicParams);
    } __except(1) {
        AddMessage(NV_LOG_ERROR, _T("cuvidDecodePicture error\n"));
        curesult = CUDA_ERROR_UNKNOWN;
    }
    //cuvidCtxUnlock(m_ctxLock, 0);
    if (curesult != CUDA_SUCCESS) {
        m_bError = true;
    }
    return (curesult == CUDA_SUCCESS);
}

int CuvidDecode::DecVideoSequence(CUVIDEOFORMAT *pFormat) {
    AddMessage(NV_LOG_TRACE, _T("DecVideoSequence\n"));
    if (   (pFormat->codec         != m_videoDecodeCreateInfo.CodecType)
        || (pFormat->chroma_format != m_videoDecodeCreateInfo.ChromaFormat)) {
        AddMessage(NV_LOG_ERROR, _T("dynamic video format changing detected\n"));
        m_bError = true;
        return 0;
    }
    if (   (pFormat->coded_width   != m_videoDecodeCreateInfo.ulWidth)
        || (pFormat->coded_height  != m_videoDecodeCreateInfo.ulHeight)) {
        AddMessage(NV_LOG_DEBUG, _T("dynamic video format changing detected\n"));
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
    AddMessage(NV_LOG_TRACE, _T("DecPictureDisplay idx: %d, %I64d\n"), pPicParams->picture_index, pPicParams->timestamp);
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
        AddMessage(NV_LOG_ERROR, _T("cuvidCreateDecoder error\n"));
        curesult = CUDA_ERROR_UNKNOWN;
    }
    return curesult;
}

CUresult CuvidDecode::InitDecode(CUvideoctxlock ctxLock, const InputVideoInfo *input, const VppParam *vpp, shared_ptr<CNVEncLog> pLog, bool ignoreDynamicFormatChange) {
    //初期化
    CloseDecoder();

    m_pPrintMes = pLog;
    m_bIgnoreDynamicFormatChange = ignoreDynamicFormatChange;
    m_deinterlaceMode = vpp->deinterlace;

    if (!check_if_nvcuvid_dll_available()) {
        AddMessage(NV_LOG_ERROR, _T("nvcuvid.dll does not exist.\n"));
        return CUDA_ERROR_NOT_FOUND;
    }
    AddMessage(NV_LOG_DEBUG, _T("nvcuvid.dll available\n"));

    if (!ctxLock) {
        AddMessage(NV_LOG_ERROR, _T("invalid ctxLock.\n"));
        return CUDA_ERROR_INVALID_VALUE;
    }

    m_ctxLock = ctxLock;

    if (nullptr == (m_pFrameQueue = new CUVIDFrameQueue(m_ctxLock))) {
        AddMessage(NV_LOG_ERROR, _T("Failed to alloc frame queue for decoder.\n"));
        return CUDA_ERROR_OUT_OF_MEMORY;
    }
    m_pFrameQueue->init(input->width, input->height);
    AddMessage(NV_LOG_DEBUG, _T("created frame queue\n"));

    //init video parser
    memset(&m_videoFormatEx, 0, sizeof(CUVIDEOFORMATEX));
    if (input->codecExtra && input->codecExtraSize) {
        if (input->codecExtraSize > sizeof(m_videoFormatEx.raw_seqhdr_data)) {
            AddMessage(NV_LOG_ERROR, _T("Parsed header too large!\n"));
            return CUDA_ERROR_INVALID_VALUE;
        }
        memcpy(m_videoFormatEx.raw_seqhdr_data, input->codecExtra, input->codecExtraSize);
        m_videoFormatEx.format.seqhdr_data_length = input->codecExtraSize;
    }

    CUVIDPARSERPARAMS oVideoParserParameters;
    memset(&oVideoParserParameters, 0, sizeof(CUVIDPARSERPARAMS));
    oVideoParserParameters.CodecType              = input->codec;
    oVideoParserParameters.ulMaxNumDecodeSurfaces = FrameQueue::cnMaximumSize;
    oVideoParserParameters.ulMaxDisplayDelay      = 1;
    oVideoParserParameters.pUserData              = this;
    oVideoParserParameters.pfnSequenceCallback    = HandleVideoSequence;
    oVideoParserParameters.pfnDecodePicture       = HandlePictureDecode;
    oVideoParserParameters.pfnDisplayPicture      = HandlePictureDisplay;
    oVideoParserParameters.pExtVideoInfo          = &m_videoFormatEx;

    CUresult curesult = CUDA_SUCCESS;
    if (CUDA_SUCCESS != (curesult = cuvidCreateVideoParser(&m_videoParser, &oVideoParserParameters))) {
        AddMessage(NV_LOG_ERROR, _T("Failed cuvidCreateVideoParser %d (%s)\n"), curesult, char_to_tstring(_cudaGetErrorEnum(curesult)).c_str());
        return curesult;
    }
    AddMessage(NV_LOG_DEBUG, _T("created video parser\n"));

    cuvidCtxLock(m_ctxLock, 0);
    memset(&m_videoDecodeCreateInfo, 0, sizeof(CUVIDDECODECREATEINFO));
    m_videoDecodeCreateInfo.CodecType = input->codec;
    m_videoDecodeCreateInfo.ulWidth   = input->codedWidth  ? input->codedWidth  : input->width;
    m_videoDecodeCreateInfo.ulHeight  = input->codedHeight ? input->codedHeight : input->height;
    m_videoDecodeCreateInfo.ulNumDecodeSurfaces = FrameQueue::cnMaximumSize;

    m_videoDecodeCreateInfo.ChromaFormat = cudaVideoChromaFormat_420;
    m_videoDecodeCreateInfo.OutputFormat = cudaVideoSurfaceFormat_NV12;
    m_videoDecodeCreateInfo.DeinterlaceMode = vpp->deinterlace;

    if (input->dstWidth > 0 && input->dstHeight > 0) {
        m_videoDecodeCreateInfo.ulTargetWidth  = input->dstWidth;
        m_videoDecodeCreateInfo.ulTargetHeight = input->dstHeight;
    } else {
        m_videoDecodeCreateInfo.ulTargetWidth  = input->width;
        m_videoDecodeCreateInfo.ulTargetHeight = input->height;
    }

    m_videoDecodeCreateInfo.display_area.left   = (short)input->crop.e.left;
    m_videoDecodeCreateInfo.display_area.top    = (short)input->crop.e.up;
    m_videoDecodeCreateInfo.display_area.right  = (short)(input->crop.e.right  + input->codedWidth  - input->width);
    m_videoDecodeCreateInfo.display_area.bottom = (short)(input->crop.e.bottom + input->codedHeight - input->height);

    m_videoDecodeCreateInfo.ulNumOutputSurfaces = 1;
    m_videoDecodeCreateInfo.ulCreationFlags = (input->cuvidType == NV_ENC_AVCUVID_CUDA) ? cudaVideoCreate_PreferCUDA : cudaVideoCreate_PreferCUVID;
    m_videoDecodeCreateInfo.vidLock = m_ctxLock;
    curesult = CreateDecoder();
    cuvidCtxUnlock(m_ctxLock, 0);
    if (CUDA_SUCCESS != curesult) {
        AddMessage(NV_LOG_ERROR, _T("Failed cuvidCreateDecoder %d (%s)\n"), curesult, char_to_tstring(_cudaGetErrorEnum(curesult)).c_str());
        return curesult;
    }
    AddMessage(NV_LOG_DEBUG, _T("created decoder (mode: %s)\n"), get_chr_from_value(list_cuvid_mode, input->cuvidType));

    if (m_videoFormatEx.raw_seqhdr_data && m_videoFormatEx.format.seqhdr_data_length) {
        if (CUDA_SUCCESS != (curesult = DecodePacket(m_videoFormatEx.raw_seqhdr_data, m_videoFormatEx.format.seqhdr_data_length, AV_NOPTS_VALUE, CUVID_NATIVE_TIMEBASE))) {
            AddMessage(NV_LOG_ERROR, _T("Failed to decode header %d (%s).\n"), curesult, char_to_tstring(_cudaGetErrorEnum(curesult)).c_str());
            return curesult;
        }
    }
    AddMessage(NV_LOG_DEBUG, _T("DecodePacket: success\n"));

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
        AddMessage(NV_LOG_ERROR, _T("cuvidParseVideoData error\n"));
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
        pCuvidPacket.timestamp  = av_rescale_q(timestamp, streamtimebase, CUVID_NATIVE_TIMEBASE);
    }

    //cuvidCtxLock(m_ctxLock, 0);
    __try {
        result = cuvidParseVideoData(m_videoParser, &pCuvidPacket);
    } __except (1) {
        AddMessage(NV_LOG_ERROR, _T("cuvidParseVideoData error\n"));
        result = CUDA_ERROR_UNKNOWN;
    }
    //cuvidCtxUnlock(m_ctxLock, 0);
    return result;
}

FrameInfo CuvidDecode::GetDecFrameInfo() {
    FrameInfo frame;
    frame.ptr = nullptr;
    frame.csp = NV_ENC_CSP_NV12;
    frame.width = m_videoDecodeCreateInfo.ulWidth;
    frame.height = m_videoDecodeCreateInfo.ulHeight;
    frame.pitch = 0; //この段階では取得できない、cuvidMapVideoFrameで取得
    frame.timestamp = (uint64_t)AV_NOPTS_VALUE;
    return frame;
}

#endif // #if ENABLE_AVCUVID_READER