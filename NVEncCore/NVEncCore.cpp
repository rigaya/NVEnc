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

#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <Windows.h>
#include <vector>
#include <array>
#include <map>
#include <string>
#include <algorithm>
#include <thread>
#include <tchar.h>
#include <cuda.h>
#include "helper_cuda.h"
#include "process.h"
#pragma comment(lib, "winmm.lib")
#include "nvEncodeAPI.h"
#include "NVEncCore.h"
#include "NVEncVersion.h"
#include "NVEncStatus.h"
#include "NVEncParam.h"
#include "NVEncUtil.h"
#include "NVEncInput.h"
#include "NVEncInputRaw.h"
#include "NVEncInputAvs.h"
#include "NVEncInputVpy.h"
#include "avcodec_reader.h"
#include "avcodec_writer.h"
#include "helper_nvenc.h"
#include "chapter_rw.h"
#include "shlwapi.h"
#pragma comment(lib, "shlwapi.lib")

bool check_if_nvcuda_dll_available() {
    //check for nvcuda.dll
    HMODULE hModule = LoadLibrary(_T("nvcuda.dll"));
    if (hModule == NULL)
        return false;
    FreeLibrary(hModule);
    return true;
}

NVEncoderGPUInfo::NVEncoderGPUInfo() {
    CUresult cuResult = CUDA_SUCCESS;

    if (!check_if_nvcuda_dll_available())
        return;

    if (CUDA_SUCCESS != (cuResult = cuInit(0)))
        return;

    int deviceCount = 0;
    if (CUDA_SUCCESS != (cuDeviceGetCount(&deviceCount)) || 0 == deviceCount)
        return;

    for (int currentDevice = 0; currentDevice < deviceCount; currentDevice++) {
        char gpu_name[1024] = { 0 };
        int SMminor = 0, SMmajor = 0;
        CUdevice cuDevice = 0;
        if (   CUDA_SUCCESS == cuDeviceGet(&cuDevice, currentDevice)
            && CUDA_SUCCESS == cuDeviceGetName(gpu_name, _countof(gpu_name), cuDevice)
            && CUDA_SUCCESS == cuDeviceComputeCapability(&SMmajor, &SMminor, currentDevice)
            && (((SMmajor << 4) + SMminor) >= 0x30)) {
            GPUList.push_back(std::make_pair(currentDevice, char_to_tstring(gpu_name)));
        }
    }
};

InEncodeVideoParam::InEncodeVideoParam() :
    input(),
    outputFilename(),
    sAVMuxOutputFormat(),
    preset(0),
    deviceID(0),
    inputBuffer(16),
    par(),
    picStruct(NV_ENC_PIC_STRUCT_FRAME),
    encConfig(),
    codec(0),
    bluray(0),                   //bluray出力
    yuv444(0),                   //YUV444出力
    lossless(0),                 //ロスレス出力
    logfile(),              //ログ出力先
    loglevel(NV_LOG_INFO),                 //ログ出力レベル
    nOutputBufSizeMB(DEFAULT_OUTPUT_BUF),         //出力バッファサイズ
    sFramePosListLog(),     //framePosList出力先
    fSeekSec(0.0f),               //指定された秒数分先頭を飛ばす
    nSubtitleSelectCount(0),
    pSubtitleSelect(nullptr),
    nAudioSourceCount(0),
    ppAudioSourceList(nullptr),
    nAudioSelectCount(0), //pAudioSelectの数
    ppAudioSelectList(nullptr),
    nAudioResampler(NV_RESAMPLER_SWR),
    nAVDemuxAnalyzeSec(0),
    nAVMux(NVENC_MUX_NONE),                       //NVENC_MUX_xxx
    nVideoTrack(0),
    nVideoStreamId(0),
    nTrimCount(0),
    pTrimList(nullptr),
    bCopyChapter(false),
    nOutputThread(NV_OUTPUT_THREAD_AUTO),
    nAudioThread(NV_INPUT_THREAD_AUTO),
    nInputThread(NV_AUDIO_THREAD_AUTO),
    bAudioIgnoreNoTrackError(false),
    nAudioIgnoreDecodeError(DEFAULT_IGNORE_DECODE_ERROR),
    pMuxOpt(nullptr),
    sChapterFile(),
    pMuxVidTsLogFile(nullptr),
    pAVInputFormat(nullptr),
    nAVSyncMode(NV_AVSYNC_THROUGH),     //avsyncの方法 (NV_AVSYNC_xxx)
    nProcSpeedLimit(0),      //処理速度制限 (0で制限なし)
    vpp() {
    encConfig = NVEncCore::DefaultParam();
    memset(&par,       0, sizeof(par));
    memset(&input,     0, sizeof(input));
}

NVEncoderGPUInfo::~NVEncoderGPUInfo() {
};

NVEncCore::NVEncCore() {
    m_pEncodeAPI = nullptr;
    m_ctxLock = NULL;
    m_hinstLib = NULL;
    m_hEncoder = nullptr;
    m_pStatus = nullptr;
    m_pFileReader = nullptr;
    m_uEncodeBufferCount = 16;
    m_pDevice = nullptr;
    m_nDeviceId = 0;
    m_pAbortByUser = nullptr;
    m_pTrimParam = nullptr;

    INIT_CONFIG(m_stCreateEncodeParams, NV_ENC_INITIALIZE_PARAMS);
    INIT_CONFIG(m_stEncConfig, NV_ENC_CONFIG);

    memset(&m_stEOSOutputBfr, 0, sizeof(m_stEOSOutputBfr));
    memset(&m_stEncodeBuffer, 0, sizeof(m_stEncodeBuffer));
}

NVEncCore::~NVEncCore() {
    Deinitialize();

    if (m_pEncodeAPI) {
        delete m_pEncodeAPI;
        m_pEncodeAPI = nullptr;
    }

    if (m_hinstLib) {
        FreeLibrary(m_hinstLib);
        m_hinstLib = NULL;
    }
}

void NVEncCore::SetAbortFlagPointer(bool *abortFlag) {
    m_pAbortByUser = abortFlag;
}

#pragma warning(push)
#pragma warning(disable:4100)
void NVEncCore::PrintMes(int logLevel, const TCHAR *format, ...) {
    if (m_pNVLog.get() == nullptr) {
        if (logLevel <= NV_LOG_INFO) {
            return;
        }
    } else if (logLevel < m_pNVLog->getLogLevel()) {
        return;
    }

    va_list args;
    va_start(args, format);

    int len = _vsctprintf(format, args) + 1; // _vscprintf doesn't count terminating '\0'
    vector<TCHAR> buffer(len, 0);
    _vstprintf_s(buffer.data(), len, format, args);
    va_end(args);

    if (m_pNVLog.get() != nullptr) {
        m_pNVLog->write(logLevel, buffer.data());
    } else {
        _ftprintf(stderr, _T("%s"), buffer.data());
    }
}

void NVEncCore::NVPrintFuncError(const TCHAR *funcName, NVENCSTATUS nvStatus) {
    PrintMes(NV_LOG_ERROR, (FOR_AUO) ? _T("%s() がエラーを返しました。: %d (%s)\n") : _T("Error on %s: %d (%s)\n"), funcName, nvStatus, char_to_tstring(_nvencGetErrorEnum(nvStatus)).c_str());
}

void NVEncCore::NVPrintFuncError(const TCHAR *funcName, CUresult code) {
    PrintMes(NV_LOG_ERROR, (FOR_AUO) ? _T("%s() がエラーを返しました。: %d (%s)\n") : _T("Error on %s: %d (%s)\n"), funcName, (int)code, char_to_tstring(_cudaGetErrorEnum(code)).c_str());
}

//ログを初期化
NVENCSTATUS NVEncCore::InitLog(const InEncodeVideoParam *inputParam) {
    //ログの初期化
    m_pNVLog.reset(new CNVEncLog(inputParam->logfile.c_str(), inputParam->loglevel));
    if (inputParam->logfile.length()) {
        m_pNVLog->writeFileHeader(inputParam->outputFilename.c_str());
    }
    return NV_ENC_SUCCESS;
}

NVENCSTATUS NVEncCore::readChapterFile(const tstring& chapfile) {
#if ENABLE_AVCUVID_READER
    ChapterRW chapter;
    auto err = chapter.read_file(chapfile.c_str(), CODE_PAGE_UNSET, 0.0);
    if (err != AUO_CHAP_ERR_NONE) {
        PrintMes(NV_LOG_ERROR, _T("failed to %s chapter file: \"%s\".\n"), (err == AUO_CHAP_ERR_FILE_OPEN) ? _T("open") : _T("read"), chapfile.c_str());
        return NV_ENC_ERR_GENERIC;
    }
    if (chapter.chapterlist().size() == 0) {
        PrintMes(NV_LOG_ERROR, _T("no chapter found from chapter file: \"%s\".\n"), chapfile.c_str());
        return NV_ENC_ERR_GENERIC;
    }
    m_AVChapterFromFile.clear();
    const auto& chapter_list = chapter.chapterlist();
    tstring chap_log;
    for (size_t i = 0; i < chapter_list.size(); i++) {
        unique_ptr<AVChapter> avchap(new AVChapter);
        avchap->time_base = av_make_q(1, 1000);
        avchap->start = chapter_list[i]->get_ms();
        avchap->end = (i < chapter_list.size()-1) ? chapter_list[i+1]->get_ms() : avchap->start + 1;
        avchap->id = (int)m_AVChapterFromFile.size();
        avchap->metadata = nullptr;
        av_dict_set(&avchap->metadata, "title", wstring_to_string(chapter_list[i]->name, CP_UTF8).c_str(), 0);
        chap_log += strsprintf(_T("chapter #%02d [%d.%02d.%02d.%03d]: %s.\n"),
            avchap->id, chapter_list[i]->h, chapter_list[i]->m, chapter_list[i]->s, chapter_list[i]->ms,
            wstring_to_tstring(chapter_list[i]->name).c_str());
        m_AVChapterFromFile.push_back(std::move(avchap));
    }
    PrintMes(NV_LOG_ERROR, _T("%s"), chap_log.c_str());
    return NV_ENC_SUCCESS;
#else
    PrintMes(NV_LOG_ERROR, _T("chater reading unsupportted in this build"));
    return NV_ENC_ERR_UNIMPLEMENTED;
#endif //#if ENABLE_AVCODEC_QSV_READER
}

NVENCSTATUS NVEncCore::InitInput(InEncodeVideoParam *inputParam) {
    int sourceAudioTrackIdStart = 1;    //トラック番号は1スタート
    int sourceSubtitleTrackIdStart = 1; //トラック番号は1スタート
#if RAW_READER
    if (inputParam->input.type == NV_ENC_INPUT_UNKNWON) {
        if (check_ext(inputParam->input.filename, { ".y4m" })) {
            inputParam->input.type = NV_ENC_INPUT_Y4M;
        } else if (check_ext(inputParam->input.filename, { ".yuv" })) {
            inputParam->input.type = NV_ENC_INPUT_RAW;
#if AVI_READER
        } else if (check_ext(inputParam->input.filename, { ".avi" })) {
            inputParam->input.type = NV_ENC_INPUT_AVI;
#endif
#if AVS_READER
        } else if (check_ext(inputParam->input.filename, { ".avs" })) {
            inputParam->input.type = NV_ENC_INPUT_AVS;
#endif
#if VPY_READER
        } else if (check_ext(inputParam->input.filename, { ".vpy" })) {
            inputParam->input.type = NV_ENC_INPUT_VPY_MT;
#endif
        } else {
#if ENABLE_AVCUVID_READER
            inputParam->input.type = NV_ENC_INPUT_AVANY;
#else
            inputParam->input.type = NV_ENC_INPUT_RAW;
#endif
        }
    }

    //Check if selected format is enabled
    if (inputParam->input.type == NV_ENC_INPUT_AVS && !AVS_READER) {
        PrintMes(NV_LOG_ERROR, _T("avs reader not compiled in this binary.\n"));
        return NV_ENC_ERR_UNSUPPORTED_PARAM;
    }
    if (inputParam->input.type == NV_ENC_INPUT_VPY_MT && !VPY_READER) {
        PrintMes(NV_LOG_ERROR, _T("vpy reader not compiled in this binary.\n"));
        return NV_ENC_ERR_UNSUPPORTED_PARAM;
    }
    if (inputParam->input.type == NV_ENC_INPUT_AVI && !AVI_READER) {
        PrintMes(NV_LOG_ERROR, _T("avi reader not compiled in this binary.\n"));
        return NV_ENC_ERR_UNSUPPORTED_PARAM;
    }
    if (inputParam->input.type == NV_ENC_INPUT_AVCUVID && !ENABLE_AVCUVID_READER) {
        PrintMes(NV_LOG_ERROR, _T("avcodec + cuvid reader not compiled in this binary.\n"));
        return NV_ENC_ERR_UNSUPPORTED_PARAM;
    }
    if (inputParam->input.type == NV_ENC_INPUT_AVSW && !ENABLE_AVCUVID_READER) {
        PrintMes(NV_LOG_ERROR, _T("avsw reader not compiled in this binary.\n"));
        return NV_ENC_ERR_UNSUPPORTED_PARAM;
    }
    
#if AVS_READER
    InputInfoAvs inputInfoAvs = { 0 };
#endif
#if VPY_READER
    InputInfoVpy inputInfoVpy = { 0 };
#endif
#if ENABLE_AVCUVID_READER
    AvcodecReaderPrm inputInfoAVCuvid = { 0 };
#endif

    switch (inputParam->input.type) {
#if AVS_READER
    case NV_ENC_INPUT_AVS:
        inputInfoAvs.interlaced = is_interlaced(inputParam->picStruct);
        inputParam->input.otherPrm = &inputInfoAvs;
        PrintMes(NV_LOG_DEBUG, _T("avs reader selected.\n"));
        m_pFileReader.reset(new NVEncInputAvs());
        break;
#endif //AVS_READER
#if VPY_READER
    case NV_ENC_INPUT_VPY:
    case NV_ENC_INPUT_VPY_MT:
        inputInfoVpy.interlaced = is_interlaced(inputParam->picStruct);
        inputInfoVpy.mt = (inputParam->input.type == NV_ENC_INPUT_VPY_MT);
        inputParam->input.otherPrm = &inputInfoVpy;
        PrintMes(NV_LOG_DEBUG, _T("vpy reader selected.\n"));
        m_pFileReader.reset(new NVEncInputVpy());
        break;
#endif //VPY_READER
#if ENABLE_AVCUVID_READER
    case NV_ENC_INPUT_AVCUVID:
    case NV_ENC_INPUT_AVSW:
    case NV_ENC_INPUT_AVANY:
        inputInfoAVCuvid.pInputFormat = inputParam->pAVInputFormat;
        inputInfoAVCuvid.bReadVideo = true;
        inputInfoAVCuvid.nVideoDecodeSW = decodeModeFromInputFmtType(inputParam->input.type);
        inputInfoAVCuvid.nVideoTrack = inputParam->nVideoTrack;
        inputInfoAVCuvid.nVideoStreamId = inputParam->nVideoStreamId;
        inputInfoAVCuvid.nReadAudio = inputParam->nAudioSelectCount > 0;
        inputInfoAVCuvid.bReadSubtitle = inputParam->nSubtitleSelectCount > 0;
        inputInfoAVCuvid.bReadChapter = !!inputParam->bCopyChapter;
        inputInfoAVCuvid.nVideoAvgFramerate = std::make_pair(inputParam->input.rate, inputParam->input.scale);
        inputInfoAVCuvid.nAnalyzeSec = inputParam->nAVDemuxAnalyzeSec;
        inputInfoAVCuvid.nTrimCount = inputParam->nTrimCount;
        inputInfoAVCuvid.pTrimList = inputParam->pTrimList;
        inputInfoAVCuvid.nAudioTrackStart = sourceAudioTrackIdStart;
        inputInfoAVCuvid.nSubtitleTrackStart = sourceSubtitleTrackIdStart;
        inputInfoAVCuvid.nAudioSelectCount = inputParam->nAudioSelectCount;
        inputInfoAVCuvid.ppAudioSelect = inputParam->ppAudioSelectList;
        inputInfoAVCuvid.nSubtitleSelectCount = inputParam->nSubtitleSelectCount;
        inputInfoAVCuvid.pSubtitleSelect = inputParam->pSubtitleSelect;
        inputInfoAVCuvid.nProcSpeedLimit = inputParam->nProcSpeedLimit;
        inputInfoAVCuvid.nAVSyncMode = inputParam->nAVSyncMode;
        inputInfoAVCuvid.fSeekSec = inputParam->fSeekSec;
        inputInfoAVCuvid.pFramePosListLog = inputParam->sFramePosListLog.c_str();
        inputInfoAVCuvid.nInputThread = inputParam->nInputThread;
        inputInfoAVCuvid.bAudioIgnoreNoTrackError = inputParam->bAudioIgnoreNoTrackError;
        inputParam->input.otherPrm = &inputInfoAVCuvid;
        PrintMes(NV_LOG_DEBUG, _T("avcuvid reader selected.\n"));
        m_pFileReader.reset(new CAvcodecReader());
        break;
#endif //#if ENABLE_AVCUVID_READER
    case NV_ENC_INPUT_RAW:
    case NV_ENC_INPUT_Y4M:
    default:
        PrintMes(NV_LOG_DEBUG, _T("raw/y4m reader selected.\n"));
        m_pFileReader.reset(new NVEncInputRaw());
        break;
    }
    PrintMes(NV_LOG_DEBUG, _T("InitInput: input selected : %d.\n"), inputParam->input.type);

    m_pStatus.reset(new EncodeStatus());
    m_pStatus->init(m_pNVLog);
    m_pFileReader->SetNVEncLogPtr(m_pNVLog);
    int ret = m_pFileReader->Init(&inputParam->input, m_pStatus);
    if (ret != 0) {
        PrintMes(NV_LOG_ERROR, m_pFileReader->GetInputMessage());
        return NV_ENC_ERR_GENERIC;
    }
    m_pStatus->m_nOutputFPSRate = inputParam->input.rate;
    m_pStatus->m_nOutputFPSScale = inputParam->input.scale;
    sourceAudioTrackIdStart    += m_pFileReader->GetAudioTrackCount();
    sourceSubtitleTrackIdStart += m_pFileReader->GetSubtitleTrackCount();

    m_inputFps.first = inputParam->input.rate;
    m_inputFps.second = inputParam->input.scale;

#if ENABLE_AVCUVID_READER
    if (inputParam->nAudioSourceCount > 0) {

        for (int i = 0; i < (int)inputParam->nAudioSourceCount; i++) {
            InputVideoInfo inputInfo = inputParam->input;
            inputInfo.filename = inputParam->ppAudioSourceList[i];

            AvcodecReaderPrm inputInfoAVAudioReader = { 0 };
            inputInfoAVAudioReader.bReadVideo = false;
            inputInfoAVAudioReader.nReadAudio = inputParam->nAudioSourceCount > 0;
            inputInfoAVAudioReader.bReadSubtitle = false;
            inputInfoAVAudioReader.bReadChapter = false;
            inputInfoAVAudioReader.nVideoAvgFramerate = std::make_pair(m_pStatus->m_nOutputFPSRate, m_pStatus->m_nOutputFPSScale);
            inputInfoAVAudioReader.nAnalyzeSec = inputParam->nAVDemuxAnalyzeSec;
            inputInfoAVAudioReader.nTrimCount = inputParam->nTrimCount;
            inputInfoAVAudioReader.pTrimList = inputParam->pTrimList;
            inputInfoAVAudioReader.nAudioTrackStart = sourceAudioTrackIdStart;
            inputInfoAVAudioReader.nSubtitleTrackStart = sourceSubtitleTrackIdStart;
            inputInfoAVAudioReader.nAudioSelectCount = inputParam->nAudioSelectCount;
            inputInfoAVAudioReader.ppAudioSelect = inputParam->ppAudioSelectList;
            inputInfoAVAudioReader.nProcSpeedLimit = inputParam->nProcSpeedLimit;
            inputInfoAVAudioReader.nAVSyncMode = NV_AVSYNC_THROUGH;
            inputInfoAVAudioReader.fSeekSec = inputParam->fSeekSec;
            inputInfoAVAudioReader.pFramePosListLog = inputParam->sFramePosListLog.c_str();
            inputInfoAVAudioReader.nInputThread = 0;
            inputInfoAVAudioReader.bAudioIgnoreNoTrackError = inputParam->bAudioIgnoreNoTrackError;
            inputInfo.otherPrm = &inputInfoAVAudioReader;

            unique_ptr<CAvcodecReader> audioReader(new CAvcodecReader());
            audioReader->SetNVEncLogPtr(m_pNVLog);
            ret = audioReader->Init(&inputInfo, nullptr);
            if (ret != 0) {
                PrintMes(NV_LOG_ERROR, audioReader->GetInputMessage());
                return NV_ENC_ERR_GENERIC;
            }
            sourceAudioTrackIdStart += audioReader->GetAudioTrackCount();
            sourceSubtitleTrackIdStart += audioReader->GetSubtitleTrackCount();
            m_AudioReaders.push_back(std::move(audioReader));
        }
    }
#endif //#if ENABLE_AVCUVID_READER

    if (!m_pFileReader->inputCodecIsValid()
        && inputParam->nTrimCount > 0) {
        //avqsvリーダー以外は、trimは自分ではセットされないので、ここでセットする
        sTrimParam trimParam;
        trimParam.list = make_vector(inputParam->pTrimList, inputParam->nTrimCount);
        trimParam.offset = 0;
        m_pFileReader->SetTrimParam(trimParam);
    }
    //trim情報をリーダーから取得する
    auto trimParam = m_pFileReader->GetTrimParam();
    m_pTrimParam = (trimParam->list.size()) ? trimParam : nullptr;
    if (m_pTrimParam) {
        PrintMes(NV_LOG_DEBUG, _T("Input: trim options\n"));
        for (int i = 0; i < (int)m_pTrimParam->list.size(); i++) {
            PrintMes(NV_LOG_DEBUG, _T("%d-%d "), m_pTrimParam->list[i].start, m_pTrimParam->list[i].fin);
        }
        PrintMes(NV_LOG_DEBUG, _T(" (offset: %d)\n"), m_pTrimParam->offset);
    }
    return NV_ENC_SUCCESS;
#else
    return NV_ENC_ERR_INVALID_CALL;
#endif //RAW_READER
}
#pragma warning(pop)

NVENCSTATUS NVEncCore::InitOutput(InEncodeVideoParam *inputParams) {
    int sts = 0;
    bool stdoutUsed = false;
#if ENABLE_AVCUVID_READER
    vector<int> streamTrackUsed; //使用した音声/字幕のトラックIDを保存する
    bool useH264ESOutput =
        ((inputParams->sAVMuxOutputFormat.length() > 0 && 0 == _tcscmp(inputParams->sAVMuxOutputFormat.c_str(), _T("raw")))) //--formatにrawが指定されている
        || (PathFindExtension(inputParams->outputFilename.c_str()) == nullptr || PathFindExtension(inputParams->outputFilename.c_str())[0] != '.') //拡張子がしない
        || check_ext(inputParams->outputFilename.c_str(), { ".m2v", ".264", ".h264", ".avc", ".avc1", ".x264", ".265", ".h265", ".hevc" }); //特定の拡張子
    if (!useH264ESOutput) {
        inputParams->nAVMux |= NVENC_MUX_VIDEO;
    }
    //if (inputParams->CodecId == MFX_CODEC_RAW) {
    //    inputParams->nAVMux &= ~NVENC_MUX_VIDEO;
    //}
    if (inputParams->nAVMux & NVENC_MUX_VIDEO) {
        PrintMes(NV_LOG_DEBUG, _T("Output: Using avformat writer.\n"));
        m_pFileWriter = std::make_shared<CAvcodecWriter>();
        AvcodecWriterPrm writerPrm;
        writerPrm.pOutputFormat = inputParams->sAVMuxOutputFormat.c_str();
        if (m_pTrimParam) {
            writerPrm.trimList = m_pTrimParam->list;
        }
        writerPrm.vidPrm.encCodecGUID     = m_stCodecGUID;
        writerPrm.vidPrm.nEncWidth        = m_uEncWidth;
        writerPrm.vidPrm.nEncHeight       = m_uEncHeight;
        writerPrm.vidPrm.pEncConfig       = &m_stEncConfig;
        writerPrm.vidPrm.nPicStruct       = m_stPicStruct;
        writerPrm.vidPrm.sar              = get_sar(m_uEncWidth, m_uEncHeight, m_stCreateEncodeParams.darWidth, m_stCreateEncodeParams.darHeight);
        writerPrm.vidPrm.outFps           = av_make_q(m_stCreateEncodeParams.frameRateNum, m_stCreateEncodeParams.frameRateDen);
        writerPrm.vidPrm.bDtsUnavailable  = false;
        writerPrm.nOutputThread           = inputParams->nOutputThread;
        writerPrm.nAudioThread            = inputParams->nAudioThread;
        writerPrm.nBufSizeMB              = inputParams->nOutputBufSizeMB;
        writerPrm.nAudioResampler         = inputParams->nAudioResampler;
        writerPrm.nAudioIgnoreDecodeError = inputParams->nAudioIgnoreDecodeError;
        writerPrm.pMuxVidTsLogFile        = inputParams->pMuxVidTsLogFile;
        if (inputParams->pMuxOpt > 0) {
            writerPrm.vMuxOpt = *inputParams->pMuxOpt;
        }
        auto pAVCodecReader = std::dynamic_pointer_cast<CAvcodecReader>(m_pFileReader);
        if (pAVCodecReader != nullptr) {
            writerPrm.pInputFormatMetadata = pAVCodecReader->GetInputFormatMetadata();
            if (inputParams->sChapterFile.length() > 0) {
                //チャプターファイルを読み込む
                auto chap_sts = readChapterFile(inputParams->sChapterFile);
                if (chap_sts != NV_ENC_SUCCESS) {
                    return chap_sts;
                }
                writerPrm.chapterList.clear();
                for (uint32_t i = 0; i < m_AVChapterFromFile.size(); i++) {
                    writerPrm.chapterList.push_back(m_AVChapterFromFile[i].get());
                }
            } else {
                //入力ファイルのチャプターをコピーする
                writerPrm.chapterList = pAVCodecReader->GetChapterList();
            }
            writerPrm.vidPrm.nInputFirstKeyPts = pAVCodecReader->GetVideoFirstKeyPts();
            writerPrm.vidPrm.pInputCodecCtx = pAVCodecReader->GetInputVideoCodecCtx();
        }
        if (inputParams->nAVMux & (NVENC_MUX_AUDIO | NVENC_MUX_SUBTITLE)) {
            PrintMes(NV_LOG_DEBUG, _T("Output: Audio/Subtitle muxing enabled.\n"));
            pAVCodecReader = std::dynamic_pointer_cast<CAvcodecReader>(m_pFileReader);
            bool copyAll = false;
            for (int i = 0; !copyAll && i < (int)inputParams->nAudioSelectCount; i++) {
                //トラック"0"が指定されていれば、すべてのトラックをコピーするということ
                copyAll = (inputParams->ppAudioSelectList[i]->nAudioSelect == 0);
            }
            PrintMes(NV_LOG_DEBUG, _T("Output: CopyAll=%s\n"), (copyAll) ? _T("true") : _T("false"));
            vector<AVDemuxStream> streamList;
            if (pAVCodecReader) {
                streamList = pAVCodecReader->GetInputStreamInfo();
            }
            for (const auto& audioReader : m_AudioReaders) {
                if (audioReader->GetAudioTrackCount()) {
                    auto pAVCodecAudioReader = std::dynamic_pointer_cast<CAvcodecReader>(audioReader);
                    if (pAVCodecAudioReader) {
                        vector_cat(streamList, pAVCodecAudioReader->GetInputStreamInfo());
                    }
                    //もしavqsvリーダーでないなら、音声リーダーから情報を取得する必要がある
                    if (pAVCodecReader == nullptr) {
                        writerPrm.vidPrm.nInputFirstKeyPts = pAVCodecAudioReader->GetVideoFirstKeyPts();
                        writerPrm.vidPrm.pInputCodecCtx = pAVCodecAudioReader->GetInputVideoCodecCtx();
                    }
                }
            }

            for (auto& stream : streamList) {
                bool bStreamIsSubtitle = stream.nTrackId < 0;
                const sAudioSelect *pAudioSelect = nullptr;
                for (int i = 0; i < (int)inputParams->nAudioSelectCount; i++) {
                    if (stream.nTrackId == inputParams->ppAudioSelectList[i]->nAudioSelect
                        && inputParams->ppAudioSelectList[i]->pAudioExtractFilename == nullptr) {
                        pAudioSelect = inputParams->ppAudioSelectList[i];
                    }
                }
                if (pAudioSelect != nullptr || copyAll || bStreamIsSubtitle) {
                    streamTrackUsed.push_back(stream.nTrackId);
                    AVOutputStreamPrm prm;
                    prm.src = stream;
                    //pAudioSelect == nullptrは "copyAll" か 字幕ストリーム によるもの
                    prm.nBitrate = (pAudioSelect == nullptr) ? 0 : pAudioSelect->nAVAudioEncodeBitrate;
                    prm.nSamplingRate = (pAudioSelect == nullptr) ? 0 : pAudioSelect->nAudioSamplingRate;
                    prm.pEncodeCodec = (pAudioSelect == nullptr) ? AVQSV_CODEC_COPY : pAudioSelect->pAVAudioEncodeCodec;
                    prm.pFilter = (pAudioSelect == nullptr) ? nullptr : pAudioSelect->pAudioFilter;
                    PrintMes(NV_LOG_DEBUG, _T("Output: Added %s track#%d (stream idx %d) for mux, bitrate %d, codec: %s\n"),
                        (bStreamIsSubtitle) ? _T("sub") : _T("audio"),
                        stream.nTrackId, stream.nIndex, prm.nBitrate, prm.pEncodeCodec);
                    writerPrm.inputStreamList.push_back(std::move(prm));
                }
            }
        }
        m_pFileWriter->SetNVEncLogPtr(m_pNVLog);
        sts = m_pFileWriter->Init(inputParams->outputFilename.c_str(), &writerPrm, m_pStatus);
        if (sts != 0) {
            PrintMes(NV_LOG_ERROR, m_pFileWriter->GetOutputMessage());
            return NV_ENC_ERR_GENERIC;
        } else if (inputParams->nAVMux & (NVENC_MUX_AUDIO | NVENC_MUX_SUBTITLE)) {
            m_pFileWriterListAudio.push_back(m_pFileWriter);
        }
        stdoutUsed = m_pFileWriter->outputStdout();
        PrintMes(NV_LOG_DEBUG, _T("Output: Initialized avformat writer%s.\n"), (stdoutUsed) ? _T("using stdout") : _T(""));

        uint32_t payload_size = 0;
        vector<uint8_t> sequence_prm_buffer(NV_MAX_SEQ_HDR_LEN, 0);
        NV_ENC_SEQUENCE_PARAM_PAYLOAD sequence_prm;
        INIT_CONFIG(sequence_prm, NV_ENC_SEQUENCE_PARAM_PAYLOAD);
        sequence_prm.inBufferSize = (int)sequence_prm_buffer.size();
        sequence_prm.spsppsBuffer = sequence_prm_buffer.data();
        sequence_prm.outSPSPPSPayloadSize = &payload_size;
        NvEncGetSequenceParams(&sequence_prm);
        if (0 != m_pFileWriter->SetVideoParam(&m_stEncConfig, m_stPicStruct, &sequence_prm)) {
            return NV_ENC_ERR_GENERIC;
        }
    } else if (inputParams->nAVMux & (NVENC_MUX_AUDIO | NVENC_MUX_SUBTITLE)) {
        PrintMes(NV_LOG_ERROR, _T("Audio mux cannot be used alone, should be use with video mux.\n"));
        return NV_ENC_ERR_GENERIC;
    } else {
#endif //ENABLE_AVCUVID_READER
        m_pFileWriter = std::make_shared<NVEncOutBitstream>();
        m_pFileWriter->SetNVEncLogPtr(m_pNVLog);
        CQSVOutRawPrm rawPrm = { 0 };
        rawPrm.nBufSizeMB = inputParams->nOutputBufSizeMB;
        sts = m_pFileWriter->Init(inputParams->outputFilename.c_str(), &rawPrm, m_pStatus);
        if (sts != 0) {
            PrintMes(NV_LOG_ERROR, m_pFileWriter->GetOutputMessage());
            return NV_ENC_ERR_GENERIC;
        }
        stdoutUsed = m_pFileWriter->outputStdout();
        PrintMes(NV_LOG_DEBUG, _T("Output: Initialized bitstream writer%s.\n"), (stdoutUsed) ? _T("using stdout") : _T(""));
#if ENABLE_AVCUVID_READER
    }

    //音声の抽出
    if (inputParams->nAudioSelectCount + inputParams->nSubtitleSelectCount > (int)streamTrackUsed.size()) {
        PrintMes(NV_LOG_DEBUG, _T("Output: Audio file output enabled.\n"));
        auto pAVCodecReader = std::dynamic_pointer_cast<CAvcodecReader>(m_pFileReader);
        if (inputParams->input.type != NV_ENC_INPUT_AVCUVID
            || inputParams->input.type != NV_ENC_INPUT_AVSW
            || inputParams->input.type != NV_ENC_INPUT_AVANY
            || pAVCodecReader == nullptr) {
            PrintMes(NV_LOG_ERROR, _T("Audio output is only supported with transcoding (avqsv reader).\n"));
            return NV_ENC_ERR_GENERIC;
        } else {
            auto inutAudioInfoList = pAVCodecReader->GetInputStreamInfo();
            for (auto& audioTrack : inutAudioInfoList) {
                bool bTrackAlreadyUsed = false;
                for (auto usedTrack : streamTrackUsed) {
                    if (usedTrack == audioTrack.nTrackId) {
                        bTrackAlreadyUsed = true;
                        PrintMes(NV_LOG_DEBUG, _T("Audio track #%d is already set to be muxed, so cannot be extracted to file.\n"), audioTrack.nTrackId);
                        break;
                    }
                }
                if (bTrackAlreadyUsed) {
                    continue;
                }
                const sAudioSelect *pAudioSelect = nullptr;
                for (int i = 0; i < (int)inputParams->nAudioSelectCount; i++) {
                    if (audioTrack.nTrackId == inputParams->ppAudioSelectList[i]->nAudioSelect
                        && inputParams->ppAudioSelectList[i]->pAudioExtractFilename != nullptr) {
                        pAudioSelect = inputParams->ppAudioSelectList[i];
                    }
                }
                if (pAudioSelect == nullptr) {
                    PrintMes(NV_LOG_ERROR, _T("Audio track #%d is not used anyware, this should not happen.\n"), audioTrack.nTrackId);
                    return NV_ENC_ERR_GENERIC;
                }
                PrintMes(NV_LOG_DEBUG, _T("Output: Output audio track #%d (stream index %d) to \"%s\", format: %s, codec %s, bitrate %d\n"),
                    audioTrack.nTrackId, audioTrack.nIndex, pAudioSelect->pAudioExtractFilename, pAudioSelect->pAudioExtractFormat, pAudioSelect->pAVAudioEncodeCodec, pAudioSelect->nAVAudioEncodeBitrate);

                AVOutputStreamPrm prm;
                prm.src = audioTrack;
                //pAudioSelect == nullptrは "copyAll" によるもの
                prm.nBitrate = pAudioSelect->nAVAudioEncodeBitrate;
                prm.pFilter = pAudioSelect->pAudioFilter;
                prm.pEncodeCodec = pAudioSelect->pAVAudioEncodeCodec;
                prm.nSamplingRate = pAudioSelect->nAudioSamplingRate;

                AvcodecWriterPrm writerAudioPrm;
                writerAudioPrm.nOutputThread   = inputParams->nOutputThread;
                writerAudioPrm.nAudioThread    = inputParams->nAudioThread;
                writerAudioPrm.nBufSizeMB      = inputParams->nOutputBufSizeMB;
                writerAudioPrm.pOutputFormat   = pAudioSelect->pAudioExtractFormat;
                writerAudioPrm.nAudioIgnoreDecodeError = inputParams->nAudioIgnoreDecodeError;
                writerAudioPrm.nAudioResampler = inputParams->nAudioResampler;
                writerAudioPrm.inputStreamList.push_back(prm);
                if (m_pTrimParam) {
                    writerAudioPrm.trimList = m_pTrimParam->list;
                }
                writerAudioPrm.vidPrm.nInputFirstKeyPts = pAVCodecReader->GetVideoFirstKeyPts();
                writerAudioPrm.vidPrm.pInputCodecCtx = pAVCodecReader->GetInputVideoCodecCtx();

                auto pWriter = std::make_shared<CAvcodecWriter>();
                pWriter->SetNVEncLogPtr(m_pNVLog);
                sts = pWriter->Init(pAudioSelect->pAudioExtractFilename, &writerAudioPrm, m_pStatus);
                if (sts != 0) {
                    PrintMes(NV_LOG_ERROR, pWriter->GetOutputMessage());
                    return NV_ENC_ERR_GENERIC;
                }
                PrintMes(NV_LOG_DEBUG, _T("Output: Intialized audio output for track #%d.\n"), audioTrack.nTrackId);
                bool audioStdout = pWriter->outputStdout();
                if (stdoutUsed && audioStdout) {
                    PrintMes(NV_LOG_ERROR, _T("Multiple stream outputs are set to stdout, please remove conflict.\n"));
                    return NV_ENC_ERR_GENERIC;
                }
                stdoutUsed |= audioStdout;
                m_pFileWriterListAudio.push_back(std::move(pWriter));
            }
        }
    }
#endif //ENABLE_AVCUVID_READER
    return NV_ENC_SUCCESS;
}

NVENCSTATUS NVEncCore::InitCuda(uint32_t deviceID) {
    CUresult cuResult;
    if (CUDA_SUCCESS != (cuResult = cuInit(0))) {
        PrintMes(NV_LOG_ERROR, _T("cuInit error:0x%x\n"), cuResult);
        return NV_ENC_ERR_NO_ENCODE_DEVICE;
    }
    PrintMes(NV_LOG_DEBUG, _T("cuInit: Success.\n"));
    m_nDeviceId = deviceID;
    int deviceCount = 0;
    if (CUDA_SUCCESS != (cuResult = cuDeviceGetCount(&deviceCount))) {
        PrintMes(NV_LOG_ERROR, _T("cuDeviceGetCount error:0x%x\n"), cuResult);
        return NV_ENC_ERR_NO_ENCODE_DEVICE;
    }
    PrintMes(NV_LOG_DEBUG, _T("cuDeviceGetCount: Success.\n"));

    if (deviceID > (unsigned int)deviceCount - 1) {
        PrintMes(NV_LOG_ERROR, _T("Invalid Device Id = %d\n"), deviceID);
        return NV_ENC_ERR_INVALID_ENCODERDEVICE;
    }
    
    if (CUDA_SUCCESS != (cuResult = cuDeviceGet(&m_device, deviceID))) {
        PrintMes(NV_LOG_ERROR, _T("cuDeviceGet error:0x%x\n"), cuResult);
        return NV_ENC_ERR_NO_ENCODE_DEVICE;
    }
    PrintMes(NV_LOG_DEBUG, _T("cuDeviceGet: Success.\n"));
    
    int SMminor = 0, SMmajor = 0;
    if (CUDA_SUCCESS != (cuDeviceComputeCapability(&SMmajor, &SMminor, deviceID))) {
        PrintMes(NV_LOG_ERROR, _T("cuDeviceComputeCapability error:0x%x\n"), cuResult);
        return NV_ENC_ERR_NO_ENCODE_DEVICE;
    }

    if (((SMmajor << 4) + SMminor) < 0x30) {
        PrintMes(NV_LOG_ERROR, _T("GPU %d does not have NVENC capabilities exiting\n"), deviceID);
        return NV_ENC_ERR_NO_ENCODE_DEVICE;
    }
    PrintMes(NV_LOG_DEBUG, _T("NVENC capabilities: OK.\n"));

    if (CUDA_SUCCESS != (cuResult = cuCtxCreate((CUcontext*)(&m_pDevice), CU_CTX_SCHED_AUTO, m_device))) {
        PrintMes(NV_LOG_ERROR, _T("cuCtxCreate error:0x%x\n"), cuResult);
        return NV_ENC_ERR_NO_ENCODE_DEVICE;
    }
    PrintMes(NV_LOG_DEBUG, _T("cuCtxCreate: Success.\n"));
    
    if (CUDA_SUCCESS != (cuResult = cuCtxPopCurrent(&m_cuContextCurr))) {
        PrintMes(NV_LOG_ERROR, _T("cuCtxPopCurrent error:0x%x\n"), cuResult);
        return NV_ENC_ERR_NO_ENCODE_DEVICE;
    }
    PrintMes(NV_LOG_DEBUG, _T("cuCtxPopCurrent: Success.\n"));

#if ENABLE_AVCUVID_READER
    if (CUDA_SUCCESS != (cuResult = cuvidCtxLockCreate(&m_ctxLock, m_cuContextCurr))) {
        PrintMes(NV_LOG_ERROR, _T("Failed cuvidCtxLockCreate %d\n"), cuResult);
        return NV_ENC_ERR_NO_ENCODE_DEVICE;
    }
    PrintMes(NV_LOG_DEBUG, _T("cuvidCtxLockCreate: Success.\n"));
#endif //#if ENABLE_AVCUVID_READER
    return NV_ENC_SUCCESS;
}

NVENCSTATUS NVEncCore::NvEncCreateInputBuffer(uint32_t width, uint32_t height, void **inputBuffer, NV_ENC_BUFFER_FORMAT inputFormat) {
    NV_ENC_CREATE_INPUT_BUFFER createInputBufferParams;
    INIT_CONFIG(createInputBufferParams, NV_ENC_CREATE_INPUT_BUFFER);

    createInputBufferParams.width = width;
    createInputBufferParams.height = height;
    createInputBufferParams.memoryHeap = NV_ENC_MEMORY_HEAP_SYSMEM_CACHED;
    createInputBufferParams.bufferFmt = inputFormat;

    NVENCSTATUS nvStatus = m_pEncodeAPI->nvEncCreateInputBuffer(m_hEncoder, &createInputBufferParams);
    if (nvStatus != NV_ENC_SUCCESS) {
        NVPrintFuncError(_T("nvEncCreateInputBuffer"), nvStatus);
        return nvStatus;
    }

    *inputBuffer = createInputBufferParams.inputBuffer;

    return nvStatus;
}

NVENCSTATUS NVEncCore::NvEncDestroyInputBuffer(NV_ENC_INPUT_PTR inputBuffer) {
    NVENCSTATUS nvStatus = NV_ENC_SUCCESS;
    if (inputBuffer) {
        nvStatus = m_pEncodeAPI->nvEncDestroyInputBuffer(m_hEncoder, inputBuffer);
        if (nvStatus != NV_ENC_SUCCESS) {
            NVPrintFuncError(_T("nvEncDestroyInputBuffer"), nvStatus);
            return nvStatus;
        }
    }
    return nvStatus;
}

NVENCSTATUS NVEncCore::NvEncCreateBitstreamBuffer(uint32_t size, void **bitstreamBuffer) {
    NV_ENC_CREATE_BITSTREAM_BUFFER createBitstreamBufferParams;
    INIT_CONFIG(createBitstreamBufferParams, NV_ENC_CREATE_BITSTREAM_BUFFER);

    createBitstreamBufferParams.size = size;
    createBitstreamBufferParams.memoryHeap = NV_ENC_MEMORY_HEAP_SYSMEM_CACHED;

    NVENCSTATUS nvStatus = m_pEncodeAPI->nvEncCreateBitstreamBuffer(m_hEncoder, &createBitstreamBufferParams);
    if (nvStatus != NV_ENC_SUCCESS) {
        NVPrintFuncError(_T("nvEncCreateBitstreamBuffer"), nvStatus);
        return nvStatus;
    }

    *bitstreamBuffer = createBitstreamBufferParams.bitstreamBuffer;

    return nvStatus;
}

NVENCSTATUS NVEncCore::NvEncDestroyBitstreamBuffer(NV_ENC_OUTPUT_PTR bitstreamBuffer) {
    NVENCSTATUS nvStatus = NV_ENC_SUCCESS;
    if (bitstreamBuffer) {
        nvStatus = m_pEncodeAPI->nvEncDestroyBitstreamBuffer(m_hEncoder, bitstreamBuffer);
        if (nvStatus != NV_ENC_SUCCESS) {
            NVPrintFuncError(_T("nvEncDestroyBitstreamBuffer"), nvStatus);
            return nvStatus;
        }
    }
    return nvStatus;
}

NVENCSTATUS NVEncCore::NvEncLockBitstream(NV_ENC_LOCK_BITSTREAM *lockBitstreamBufferParams) {
    NVENCSTATUS nvStatus = m_pEncodeAPI->nvEncLockBitstream(m_hEncoder, lockBitstreamBufferParams);
    if (nvStatus != NV_ENC_SUCCESS) {
        NVPrintFuncError(_T("nvEncLockBitstream"), nvStatus);
        return nvStatus;
    }
    return nvStatus;
}

NVENCSTATUS NVEncCore::NvEncUnlockBitstream(NV_ENC_OUTPUT_PTR bitstreamBuffer) {
    NVENCSTATUS nvStatus = m_pEncodeAPI->nvEncUnlockBitstream(m_hEncoder, bitstreamBuffer);
    if (nvStatus != NV_ENC_SUCCESS) {
        NVPrintFuncError(_T("nvEncUnlockBitstream"), nvStatus);
        return nvStatus;
    }
    return nvStatus;
}

NVENCSTATUS NVEncCore::NvEncLockInputBuffer(void *inputBuffer, void **bufferDataPtr, uint32_t *pitch) {
    NV_ENC_LOCK_INPUT_BUFFER lockInputBufferParams;
    INIT_CONFIG(lockInputBufferParams, NV_ENC_LOCK_INPUT_BUFFER);

    lockInputBufferParams.inputBuffer = inputBuffer;
    NVENCSTATUS nvStatus = m_pEncodeAPI->nvEncLockInputBuffer(m_hEncoder, &lockInputBufferParams);
    if (nvStatus != NV_ENC_SUCCESS) {
        NVPrintFuncError(_T("nvEncLockInputBuffer"), nvStatus);
        return nvStatus;
    }

    *bufferDataPtr = lockInputBufferParams.bufferDataPtr;
    *pitch = lockInputBufferParams.pitch;

    return nvStatus;
}

NVENCSTATUS NVEncCore::NvEncUnlockInputBuffer(NV_ENC_INPUT_PTR inputBuffer) {
    NVENCSTATUS nvStatus = m_pEncodeAPI->nvEncUnlockInputBuffer(m_hEncoder, inputBuffer);
    if (nvStatus != NV_ENC_SUCCESS) {
        NVPrintFuncError(_T("nvEncUnlockInputBuffer"), nvStatus);
        return nvStatus;
    }
    return nvStatus;
}

NVENCSTATUS NVEncCore::NvEncGetEncodeStats(NV_ENC_STAT *encodeStats) {
    NVENCSTATUS nvStatus = m_pEncodeAPI->nvEncGetEncodeStats(m_hEncoder, encodeStats);
    if (nvStatus != NV_ENC_SUCCESS) {
        NVPrintFuncError(_T("nvEncGetEncodeStats"), nvStatus);
        return nvStatus;
    }

    return nvStatus;
}

NVENCSTATUS NVEncCore::NvEncGetSequenceParams(NV_ENC_SEQUENCE_PARAM_PAYLOAD *sequenceParamPayload) {
    NVENCSTATUS nvStatus = m_pEncodeAPI->nvEncGetSequenceParams(m_hEncoder, sequenceParamPayload);
    if (nvStatus != NV_ENC_SUCCESS) {
        NVPrintFuncError(_T("nvEncGetSequenceParams"), nvStatus);
        return nvStatus;
    }

    return nvStatus;
}

NVENCSTATUS NVEncCore::NvEncRegisterAsyncEvent(void **completionEvent) {
    NV_ENC_EVENT_PARAMS eventParams;
    INIT_CONFIG(eventParams, NV_ENC_EVENT_PARAMS);

    eventParams.completionEvent = CreateEvent(NULL, FALSE, FALSE, NULL);
    NVENCSTATUS nvStatus = m_pEncodeAPI->nvEncRegisterAsyncEvent(m_hEncoder, &eventParams);
    if (nvStatus != NV_ENC_SUCCESS) {
        NVPrintFuncError(_T("nvEncRegisterAsyncEvent"), nvStatus);
        return nvStatus;
    }

    *completionEvent = eventParams.completionEvent;

    return nvStatus;
}

NVENCSTATUS NVEncCore::NvEncUnregisterAsyncEvent(void *completionEvent) {
    if (completionEvent) {
        NV_ENC_EVENT_PARAMS eventParams;
        INIT_CONFIG(eventParams, NV_ENC_EVENT_PARAMS);

        eventParams.completionEvent = completionEvent;

        NVENCSTATUS nvStatus = m_pEncodeAPI->nvEncUnregisterAsyncEvent(m_hEncoder, &eventParams);
        if (nvStatus != NV_ENC_SUCCESS) {
            NVPrintFuncError(_T("nvEncUnregisterAsyncEvent"), nvStatus);
            return nvStatus;
        }
    }

    return NV_ENC_SUCCESS;
}

NVENCSTATUS NVEncCore::NvEncRegisterResource(NV_ENC_INPUT_RESOURCE_TYPE resourceType, void* resourceToRegister, uint32_t width, uint32_t height, uint32_t pitch, void** registeredResource) {
    NVENCSTATUS nvStatus = NV_ENC_SUCCESS;
    NV_ENC_REGISTER_RESOURCE registerResParams;

    INIT_CONFIG(registerResParams, NV_ENC_REGISTER_RESOURCE);

    registerResParams.resourceType = resourceType;
    registerResParams.resourceToRegister = resourceToRegister;
    registerResParams.width = width;
    registerResParams.height = height;
    registerResParams.pitch = pitch;

    nvStatus = m_pEncodeAPI->nvEncRegisterResource(m_hEncoder, &registerResParams);
    if (nvStatus != NV_ENC_SUCCESS) {
        NVPrintFuncError(_T("nvEncRegisterResource"), nvStatus);
    }

    *registeredResource = registerResParams.registeredResource;

    return nvStatus;
}

NVENCSTATUS NVEncCore::NvEncUnregisterResource(NV_ENC_REGISTERED_PTR registeredRes) {
    NVENCSTATUS nvStatus = NV_ENC_SUCCESS;

    nvStatus = m_pEncodeAPI->nvEncUnregisterResource(m_hEncoder, registeredRes);
    if (nvStatus != NV_ENC_SUCCESS) {
        NVPrintFuncError(_T("nvEncUnregisterResource"), nvStatus);
    }
    return nvStatus;
}

NVENCSTATUS NVEncCore::NvEncMapInputResource(void *registeredResource, void **mappedResource) {
    NV_ENC_MAP_INPUT_RESOURCE mapInputResParams;
    INIT_CONFIG(mapInputResParams, NV_ENC_MAP_INPUT_RESOURCE);

    mapInputResParams.registeredResource = registeredResource;

    NVENCSTATUS nvStatus = m_pEncodeAPI->nvEncMapInputResource(m_hEncoder, &mapInputResParams);
    if (nvStatus != NV_ENC_SUCCESS) {
        NVPrintFuncError(_T("nvEncMapInputResource"), nvStatus);
        return nvStatus;
    }

    *mappedResource = mapInputResParams.mappedResource;

    return nvStatus;
}

NVENCSTATUS NVEncCore::NvEncUnmapInputResource(NV_ENC_INPUT_PTR mappedInputBuffer) {
    if (mappedInputBuffer) {
        NVENCSTATUS nvStatus = m_pEncodeAPI->nvEncUnmapInputResource(m_hEncoder, mappedInputBuffer);
        if (nvStatus != NV_ENC_SUCCESS) {
            NVPrintFuncError(_T("nvEncUnmapInputResource"), nvStatus);
            return nvStatus;
        }
    }

    return NV_ENC_SUCCESS;
}

NVENCSTATUS NVEncCore::NvEncDestroyEncoder() {
    NVENCSTATUS nvStatus = NV_ENC_SUCCESS;

    if (m_hEncoder && m_pEncodeAPI) {
        nvStatus = m_pEncodeAPI->nvEncDestroyEncoder(m_hEncoder);
        m_hEncoder = NULL;
        m_pEncodeAPI = nullptr;
    }

    return nvStatus;
}

NVENCSTATUS NVEncCore::NvEncFlushEncoderQueue(void *hEOSEvent) {
    NV_ENC_PIC_PARAMS encPicParams;
    INIT_CONFIG(encPicParams, NV_ENC_PIC_PARAMS);
    encPicParams.encodePicFlags = NV_ENC_PIC_FLAG_EOS;
    encPicParams.completionEvent = hEOSEvent;

    NVENCSTATUS nvStatus = m_pEncodeAPI->nvEncEncodePicture(m_hEncoder, &encPicParams);
    if (nvStatus != NV_ENC_SUCCESS) {
        NVPrintFuncError(_T("nvEncEncodePicture"), nvStatus);
        return nvStatus;
    }
    return nvStatus;
}

NVENCSTATUS NVEncCore::ProcessOutput(const EncodeBuffer *pEncodeBuffer) {
    if (pEncodeBuffer->stOutputBfr.hBitstreamBuffer == NULL && pEncodeBuffer->stOutputBfr.bEOSFlag == FALSE) {
        return NV_ENC_ERR_INVALID_PARAM;
    }

    if (pEncodeBuffer->stOutputBfr.bWaitOnEvent == TRUE) {
        if (!pEncodeBuffer->stOutputBfr.hOutputEvent) {
            return NV_ENC_ERR_INVALID_PARAM;
        }
        WaitForSingleObject(pEncodeBuffer->stOutputBfr.hOutputEvent, INFINITE);
    }

    if (pEncodeBuffer->stOutputBfr.bEOSFlag)
        return NV_ENC_SUCCESS;

    NV_ENC_LOCK_BITSTREAM lockBitstreamData;
    INIT_CONFIG(lockBitstreamData, NV_ENC_LOCK_BITSTREAM);
    lockBitstreamData.outputBitstream = pEncodeBuffer->stOutputBfr.hBitstreamBuffer;
    lockBitstreamData.doNotWait = false;

    NVENCSTATUS nvStatus = m_pEncodeAPI->nvEncLockBitstream(m_hEncoder, &lockBitstreamData);
    if (nvStatus == NV_ENC_SUCCESS) {
        m_pFileWriter->WriteNextFrame(&lockBitstreamData);
        nvStatus = m_pEncodeAPI->nvEncUnlockBitstream(m_hEncoder, pEncodeBuffer->stOutputBfr.hBitstreamBuffer);
    } else {
        NVPrintFuncError(_T("nvEncLockBitstream"), nvStatus);
        return nvStatus;
    }

    return nvStatus;
}

NVENCSTATUS NVEncCore::FlushEncoder() {
    NVENCSTATUS nvStatus = NvEncFlushEncoderQueue(m_stEOSOutputBfr.hOutputEvent);
    if (nvStatus != NV_ENC_SUCCESS) {
        NVPrintFuncError(_T("NvEncFlushEncoderQueue"), nvStatus);
        return nvStatus;
    }

    EncodeBuffer *pEncodeBufer = m_EncodeBufferQueue.GetPending();
    while (pEncodeBufer) {
        ProcessOutput(pEncodeBufer);
        pEncodeBufer = m_EncodeBufferQueue.GetPending();
    }

    if (WaitForSingleObject(m_stEOSOutputBfr.hOutputEvent, 500) != WAIT_OBJECT_0) {
        PrintMes(NV_LOG_ERROR, _T("m_stEOSOutputBfr.hOutputEvent%s"), (FOR_AUO) ? _T("が終了しません。") : _T(" does not finish within proper time."));
        nvStatus = NV_ENC_ERR_GENERIC;
    }

    return nvStatus;
}

NVENCSTATUS NVEncCore::Deinitialize() {
    NVENCSTATUS nvStatus = NV_ENC_SUCCESS;

    m_AudioReaders.clear();
    m_pFileReader.reset();
    m_pFileWriter.reset();
    m_pFileWriterListAudio.clear();

    ReleaseIOBuffers();

    nvStatus = NvEncDestroyEncoder();

#if ENABLE_AVCUVID_READER
    m_cuvidDec.reset();

    if (m_ctxLock) {
        cuvidCtxLockDestroy(m_ctxLock);
        m_ctxLock = nullptr;
    }
#endif //#if ENABLE_AVCUVID_READER

    m_pStatus.reset();

    if (m_pDevice) {
        CUresult cuResult = CUDA_SUCCESS;
        cuResult = cuCtxDestroy((CUcontext)m_pDevice);
        if (cuResult != CUDA_SUCCESS)
            PrintMes(NV_LOG_ERROR, _T("cuCtxDestroy error:0x%x\n"), cuResult);

        m_pDevice = NULL;
    }

    m_pNVLog.reset();
    m_pAbortByUser = nullptr;
    m_pTrimParam = nullptr;

    return nvStatus;
}

NVENCSTATUS NVEncCore::AllocateIOBuffers(uint32_t uInputWidth, uint32_t uInputHeight, NV_ENC_BUFFER_FORMAT inputFormat) {
    NVENCSTATUS nvStatus = NV_ENC_SUCCESS;

    m_EncodeBufferQueue.Initialize(m_stEncodeBuffer, m_uEncodeBufferCount);
    for (uint32_t i = 0; i < m_uEncodeBufferCount; i++) {
#if ENABLE_AVCUVID_READER
        if (m_cuvidDec) {
            cuvidCtxLock(m_ctxLock, 0);
            auto curesult = cuMemAllocPitch(&m_stEncodeBuffer[i].stInputBfr.pNV12devPtr,
                (size_t *)&m_stEncodeBuffer[i].stInputBfr.uNV12Stride, uInputWidth, uInputHeight * 3 / 2, 16);
            cuvidCtxUnlock(m_ctxLock, 0);
            if (curesult != CUDA_SUCCESS) {
                PrintMes(NV_LOG_ERROR, _T("Failed to cuMemAllocPitch, %d (%s)\n"), curesult, char_to_tstring(_cudaGetErrorEnum(curesult)).c_str());
            }

            nvStatus = NvEncRegisterResource(NV_ENC_INPUT_RESOURCE_TYPE_CUDADEVICEPTR,
                (void*)m_stEncodeBuffer[i].stInputBfr.pNV12devPtr,
                uInputWidth, uInputHeight, m_stEncodeBuffer[i].stInputBfr.uNV12Stride,
                &m_stEncodeBuffer[i].stInputBfr.nvRegisteredResource);
        } else
#endif //#if ENABLE_AVCUVID_READER
        {
            nvStatus = NvEncCreateInputBuffer(uInputWidth, uInputHeight, &m_stEncodeBuffer[i].stInputBfr.hInputSurface, inputFormat);
            if (nvStatus != NV_ENC_SUCCESS) {
                PrintMes(NV_LOG_ERROR, _T("Failed to allocate Input Buffer, Please reduce MAX_FRAMES_TO_PRELOAD\n"));
                return nvStatus;
            }
        }

        m_stEncodeBuffer[i].stInputBfr.bufferFmt = inputFormat;
        m_stEncodeBuffer[i].stInputBfr.dwWidth = uInputWidth;
        m_stEncodeBuffer[i].stInputBfr.dwHeight = uInputHeight;

        nvStatus = NvEncCreateBitstreamBuffer(BITSTREAM_BUFFER_SIZE, &m_stEncodeBuffer[i].stOutputBfr.hBitstreamBuffer);
        if (nvStatus != NV_ENC_SUCCESS) {
            PrintMes(NV_LOG_ERROR, _T("Failed to allocate Output Buffer, Please reduce MAX_FRAMES_TO_PRELOAD\n"));
            return nvStatus;
        }
        m_stEncodeBuffer[i].stOutputBfr.dwBitstreamBufferSize = BITSTREAM_BUFFER_SIZE;

        nvStatus = NvEncRegisterAsyncEvent(&m_stEncodeBuffer[i].stOutputBfr.hOutputEvent);
        if (nvStatus != NV_ENC_SUCCESS)
            return nvStatus;
        m_stEncodeBuffer[i].stOutputBfr.bWaitOnEvent = true;
    }

    m_stEOSOutputBfr.bEOSFlag = TRUE;

    nvStatus = NvEncRegisterAsyncEvent(&m_stEOSOutputBfr.hOutputEvent);
    if (nvStatus != NV_ENC_SUCCESS)
        return nvStatus;

    return NV_ENC_SUCCESS;
}

NVENCSTATUS NVEncCore::ReleaseIOBuffers() {
    for (uint32_t i = 0; i < m_uEncodeBufferCount; i++) {
#if ENABLE_AVCUVID_READER
        if (m_cuvidDec) {
            if (m_stEncodeBuffer[i].stInputBfr.pNV12devPtr) {
                cuvidCtxLock(m_ctxLock, 0);
                cuMemFree(m_stEncodeBuffer[i].stInputBfr.pNV12devPtr);
                cuvidCtxUnlock(m_ctxLock, 0);
                m_stEncodeBuffer[i].stInputBfr.pNV12devPtr = NULL;
            }
        } else
#endif //#if ENABLE_AVCUVID_READER
        {
            if (m_stEncodeBuffer[i].stInputBfr.hInputSurface) {
                NvEncDestroyInputBuffer(m_stEncodeBuffer[i].stInputBfr.hInputSurface);
                m_stEncodeBuffer[i].stInputBfr.hInputSurface = NULL;
            }
        }

        if (m_stEncodeBuffer[i].stOutputBfr.hBitstreamBuffer) {
            NvEncDestroyBitstreamBuffer(m_stEncodeBuffer[i].stOutputBfr.hBitstreamBuffer);
            m_stEncodeBuffer[i].stOutputBfr.hBitstreamBuffer = NULL;
        }
        if (m_stEncodeBuffer[i].stOutputBfr.hOutputEvent) {
            NvEncUnregisterAsyncEvent(m_stEncodeBuffer[i].stOutputBfr.hOutputEvent);
            nvCloseFile(m_stEncodeBuffer[i].stOutputBfr.hOutputEvent);
            m_stEncodeBuffer[i].stOutputBfr.hOutputEvent = NULL;
        }
    }

    if (m_stEOSOutputBfr.hOutputEvent) {
        NvEncUnregisterAsyncEvent(m_stEOSOutputBfr.hOutputEvent);
        nvCloseFile(m_stEOSOutputBfr.hOutputEvent);
        m_stEOSOutputBfr.hOutputEvent = NULL;
    }

    return NV_ENC_SUCCESS;
}

NVENCSTATUS NVEncCore::InitDevice(const InEncodeVideoParam *inputParam) {
    if (!check_if_nvcuda_dll_available()) {
        PrintMes(NV_LOG_ERROR,
            FOR_AUO ? _T("CUDAが使用できないため、NVEncによるエンコードが行えません。(check_if_nvcuda_dll_available)\n") : _T("CUDA not available.\n"));
        return NV_ENC_ERR_UNSUPPORTED_DEVICE;
    }

    NVEncoderGPUInfo gpuInfo;
    if (0 == gpuInfo.getGPUList().size()) {
        PrintMes(NV_LOG_ERROR, FOR_AUO ? _T("NVEncが使用可能なGPUが見つかりませんでした。\n") : _T("No GPU found suitable for NVEnc Encoding.\n"));
        return NV_ENC_ERR_NO_ENCODE_DEVICE;
    }

    NVENCSTATUS nvStatus;
    if (NV_ENC_SUCCESS != (nvStatus = InitCuda(inputParam->deviceID))) {
        PrintMes(NV_LOG_ERROR, FOR_AUO ? _T("Cudaの初期化に失敗しました。\n") : _T("Failed to initialize CUDA.\n"));
        return nvStatus;
    }
    PrintMes(NV_LOG_DEBUG, _T("InitCuda: Success.\n"));
    return NV_ENC_SUCCESS;
}

NVENCSTATUS NVEncCore::NvEncOpenEncodeSessionEx(void *device, NV_ENC_DEVICE_TYPE deviceType) {
    NV_ENC_OPEN_ENCODE_SESSION_EX_PARAMS openSessionExParams;
    INIT_CONFIG(openSessionExParams, NV_ENC_OPEN_ENCODE_SESSION_EX_PARAMS);

    openSessionExParams.device = device;
    openSessionExParams.deviceType = deviceType;
    openSessionExParams.reserved = NULL;
    openSessionExParams.apiVersion = NVENCAPI_VERSION;
    
    NVENCSTATUS nvStatus = NV_ENC_SUCCESS;
    if (NV_ENC_SUCCESS != (nvStatus = m_pEncodeAPI->nvEncOpenEncodeSessionEx(&openSessionExParams, &m_hEncoder))) {
        NVPrintFuncError(_T("nvEncOpenEncodeSessionEx"), nvStatus);
        if (nvStatus == NV_ENC_ERR_OUT_OF_MEMORY) {
            PrintMes(NV_LOG_ERROR, 
                FOR_AUO ? _T("このエラーはメモリが不足しているか、同時にNVEncで3ストリーム以上エンコードしようとすると発生することがあります。")
                        : _T("This error might occur when shortage of memory, or when trying to encode more than 2 streams by NVEnc."));
        }
        return nvStatus;
    }

    return nvStatus;
}

NVENCSTATUS NVEncCore::SetEncodeCodecList(void *hEncoder) {
    NVENCSTATUS nvStatus = NV_ENC_SUCCESS;
    uint32_t dwEncodeGUIDCount = 0;
    if (NV_ENC_SUCCESS != (nvStatus = m_pEncodeAPI->nvEncGetEncodeGUIDCount(hEncoder, &dwEncodeGUIDCount))) {
        NVPrintFuncError(_T("nvEncGetEncodeGUIDCount"), nvStatus);
        return nvStatus;
    }
    uint32_t uArraysize = 0;
    GUID guid_init = { 0 };
    std::vector<GUID> list_codecs;
    list_codecs.resize(dwEncodeGUIDCount, guid_init);
    if (NV_ENC_SUCCESS != (nvStatus = m_pEncodeAPI->nvEncGetEncodeGUIDs(hEncoder, &list_codecs[0], dwEncodeGUIDCount, &uArraysize))) {
        NVPrintFuncError(_T("nvEncGetEncodeGUIDs"), nvStatus);
        return nvStatus;
    }
    for (auto codec : list_codecs) {
        m_EncodeFeatures.push_back(NVEncCodecFeature(codec));
    }
    return nvStatus;
}

NVENCSTATUS NVEncCore::setCodecProfileList(void *hEncoder, NVEncCodecFeature& codecFeature) {
    NVENCSTATUS nvStatus = NV_ENC_SUCCESS;
    uint32_t dwCodecProfileGUIDCount = 0;
    if (NV_ENC_SUCCESS != (nvStatus = m_pEncodeAPI->nvEncGetEncodeProfileGUIDCount(hEncoder, codecFeature.codec, &dwCodecProfileGUIDCount))) {
        NVPrintFuncError(_T("nvEncGetEncodeProfileGUIDCount"), nvStatus);
        return nvStatus;
    }
    uint32_t uArraysize = 0;
    GUID guid_init = { 0 };
    codecFeature.profiles.resize(dwCodecProfileGUIDCount, guid_init);
    if (NV_ENC_SUCCESS != (nvStatus = m_pEncodeAPI->nvEncGetEncodeProfileGUIDs(hEncoder, codecFeature.codec, &codecFeature.profiles[0], dwCodecProfileGUIDCount, &uArraysize))) {
        NVPrintFuncError(_T("nvEncGetEncodeProfileGUIDs"), nvStatus);
        return nvStatus;
    }
    return nvStatus;
}

NVENCSTATUS NVEncCore::setCodecPresetList(void *hEncoder, NVEncCodecFeature& codecFeature, bool getPresetConfig) {
    NVENCSTATUS nvStatus = NV_ENC_SUCCESS;
    uint32_t dwCodecProfileGUIDCount = 0;
    if (NV_ENC_SUCCESS != (nvStatus = m_pEncodeAPI->nvEncGetEncodePresetCount(hEncoder, codecFeature.codec, &dwCodecProfileGUIDCount))) {
        NVPrintFuncError(_T("nvEncGetEncodePresetCount"), nvStatus);
        return nvStatus;
    }
    uint32_t uArraysize = 0;
    GUID guid_init = { 0 };
    NV_ENC_PRESET_CONFIG config_init = { 0 };
    codecFeature.presets.resize(dwCodecProfileGUIDCount, guid_init);
    codecFeature.presetConfigs.resize(dwCodecProfileGUIDCount, config_init);
    if (NV_ENC_SUCCESS != (nvStatus = m_pEncodeAPI->nvEncGetEncodePresetGUIDs(hEncoder, codecFeature.codec, &codecFeature.presets[0], dwCodecProfileGUIDCount, &uArraysize))) {
        NVPrintFuncError(_T("nvEncGetEncodePresetGUIDs"), nvStatus);
        return nvStatus;
    }
    if (getPresetConfig) {
        for (uint32_t i = 0; i < codecFeature.presets.size(); i++) {
            INIT_CONFIG(codecFeature.presetConfigs[i], NV_ENC_PRESET_CONFIG);
            SET_VER(codecFeature.presetConfigs[i].presetCfg, NV_ENC_CONFIG);
            if (NV_ENC_SUCCESS != (nvStatus = m_pEncodeAPI->nvEncGetEncodePresetConfig(hEncoder, codecFeature.codec, codecFeature.presets[i], &codecFeature.presetConfigs[i]))) {
        NVPrintFuncError(_T("nvEncGetEncodePresetConfig"), nvStatus);
                return nvStatus;
            }
        }
    }
    return nvStatus;
}

NVENCSTATUS NVEncCore::setInputFormatList(void *hEncoder, NVEncCodecFeature& codecFeature) {
    NVENCSTATUS nvStatus = NV_ENC_SUCCESS;
    uint32_t dwInputFmtCount = 0;
    if (NV_ENC_SUCCESS != (nvStatus = m_pEncodeAPI->nvEncGetInputFormatCount(hEncoder, codecFeature.codec, &dwInputFmtCount))) {
        NVPrintFuncError(_T("nvEncGetInputFormatCount"), nvStatus);
        return nvStatus;
    }
    uint32_t uArraysize = 0;
    codecFeature.surfaceFmt.resize(dwInputFmtCount);
    if (NV_ENC_SUCCESS != (nvStatus = m_pEncodeAPI->nvEncGetInputFormats(hEncoder, codecFeature.codec, &codecFeature.surfaceFmt[0], dwInputFmtCount, &uArraysize))) {
        NVPrintFuncError(_T("nvEncGetInputFormats"), nvStatus);
        return nvStatus;
    }

    return nvStatus;
}

NVENCSTATUS NVEncCore::GetCurrentDeviceNVEncCapability(void *hEncoder, NVEncCodecFeature& codecFeature) {
    NVENCSTATUS nvStatus = NV_ENC_SUCCESS;
    bool check_h264 = get_value_from_guid(codecFeature.codec, list_nvenc_codecs) == NV_ENC_H264;
    auto add_cap_info = [&](NV_ENC_CAPS cap_id, bool for_h264_only, const TCHAR *cap_name) {
        if (!(!check_h264 && for_h264_only)) {
            NV_ENC_CAPS_PARAM param;
            INIT_CONFIG(param, NV_ENC_CAPS_PARAM);
            param.capsToQuery = cap_id;
            int value = 0;
            NVENCSTATUS result = m_pEncodeAPI->nvEncGetEncodeCaps(hEncoder, codecFeature.codec, &param, &value);
            if (NV_ENC_SUCCESS == result) {
                NVEncCap cap ={ 0 };
                cap.id = cap_id;
                cap.name = cap_name;
                cap.value = value;
                codecFeature.caps.push_back(cap);
            } else {
                nvStatus = result;
            }
        }
    };

    add_cap_info(NV_ENC_CAPS_NUM_MAX_BFRAMES,              false, _T("Max Bframes"));
    add_cap_info(NV_ENC_CAPS_SUPPORTED_RATECONTROL_MODES,  false, _T("RC Modes"));
    add_cap_info(NV_ENC_CAPS_SUPPORT_FIELD_ENCODING,       false, _T("Field Encoding"));
    add_cap_info(NV_ENC_CAPS_SUPPORT_MONOCHROME,           false, _T("MonoChrome"));
    add_cap_info(NV_ENC_CAPS_SUPPORT_FMO,                  true,  _T("FMO"));
    add_cap_info(NV_ENC_CAPS_SUPPORT_QPELMV,               false, _T("Quater-Pel MV"));
    add_cap_info(NV_ENC_CAPS_SUPPORT_BDIRECT_MODE,         false, _T("B Direct Mode"));
    add_cap_info(NV_ENC_CAPS_SUPPORT_CABAC,                true,  _T("CABAC"));
    add_cap_info(NV_ENC_CAPS_SUPPORT_ADAPTIVE_TRANSFORM,   true,  _T("Adaptive Transform"));
    add_cap_info(NV_ENC_CAPS_NUM_MAX_TEMPORAL_LAYERS,      false, _T("Max Temporal Layers"));
    add_cap_info(NV_ENC_CAPS_SUPPORT_HIERARCHICAL_PFRAMES, false, _T("Hierarchial P Frames"));
    add_cap_info(NV_ENC_CAPS_SUPPORT_HIERARCHICAL_BFRAMES, false, _T("Hierarchial B Frames"));
    add_cap_info(NV_ENC_CAPS_LEVEL_MAX,                    false, _T("Max Level"));
    add_cap_info(NV_ENC_CAPS_LEVEL_MIN,                    false, _T("Min Level"));
    add_cap_info(NV_ENC_CAPS_SEPARATE_COLOUR_PLANE,        false, _T("4:4:4"));
    add_cap_info(NV_ENC_CAPS_WIDTH_MAX,                    false, _T("Max Width"));
    add_cap_info(NV_ENC_CAPS_HEIGHT_MAX,                   false, _T("Max Height"));
    add_cap_info(NV_ENC_CAPS_SUPPORT_DYN_RES_CHANGE,       false, _T("Dynamic Resolution Change"));
    add_cap_info(NV_ENC_CAPS_SUPPORT_DYN_BITRATE_CHANGE,   false, _T("Dynamic Bitrate Change"));
    add_cap_info(NV_ENC_CAPS_SUPPORT_DYN_FORCE_CONSTQP,    false, _T("Forced constant QP"));
    add_cap_info(NV_ENC_CAPS_SUPPORT_DYN_RCMODE_CHANGE,    false, _T("Dynamic RC Mode Change"));
    add_cap_info(NV_ENC_CAPS_SUPPORT_SUBFRAME_READBACK,    false, _T("Subframe Readback"));
    add_cap_info(NV_ENC_CAPS_SUPPORT_CONSTRAINED_ENCODING, false, _T("Constrained Encoding"));
    add_cap_info(NV_ENC_CAPS_SUPPORT_INTRA_REFRESH,        false, _T("Intra Refresh"));
    add_cap_info(NV_ENC_CAPS_SUPPORT_CUSTOM_VBV_BUF_SIZE,  false, _T("Custom VBV Bufsize"));
    add_cap_info(NV_ENC_CAPS_SUPPORT_DYNAMIC_SLICE_MODE,   false, _T("Dynamic Slice Mode"));
    add_cap_info(NV_ENC_CAPS_SUPPORT_REF_PIC_INVALIDATION, false, _T("Ref Pic Invalidiation"));
    add_cap_info(NV_ENC_CAPS_PREPROC_SUPPORT,              false, _T("PreProcess"));
    add_cap_info(NV_ENC_CAPS_ASYNC_ENCODE_SUPPORT,         false, _T("Async Encoding"));
    add_cap_info(NV_ENC_CAPS_MB_NUM_MAX,                   false, _T("Max MBs"));
    add_cap_info(NV_ENC_CAPS_MB_PER_SEC_MAX,               false, _T("MAX MB per sec"));
    add_cap_info(NV_ENC_CAPS_SUPPORT_LOSSLESS_ENCODE,      false, _T("Lossless"));
    add_cap_info(NV_ENC_CAPS_SUPPORT_SAO,                  false, _T("SAO"));
    add_cap_info(NV_ENC_CAPS_SUPPORT_MEONLY_MODE,          false, _T("Me Only Mode"));
    add_cap_info(NV_ENC_CAPS_SUPPORT_LOOKAHEAD,            false, _T("Lookahead"));
    add_cap_info(NV_ENC_CAPS_SUPPORT_TEMPORAL_AQ,          false, _T("AQ (temporal)"));
    add_cap_info(NV_ENC_CAPS_SUPPORT_10BIT_ENCODE,         false, _T("10bit depth"));
    return nvStatus;
}

NVENCSTATUS NVEncCore::createDeviceCodecList() {
    return SetEncodeCodecList(m_hEncoder);
}

NVENCSTATUS NVEncCore::createDeviceFeatureList(bool getPresetConfig) {
    NVENCSTATUS nvStatus = NV_ENC_SUCCESS;
    //m_EncodeFeaturesが作成されていなければ、自動的に作成
    if (m_EncodeFeatures.size() == 0)
        nvStatus = SetEncodeCodecList(m_hEncoder);

    if (NV_ENC_SUCCESS == nvStatus) {
        for (uint32_t i = 0; i < m_EncodeFeatures.size(); i++) {
            setCodecProfileList(m_hEncoder, m_EncodeFeatures[i]);
            setCodecPresetList(m_hEncoder, m_EncodeFeatures[i], getPresetConfig);
            setInputFormatList(m_hEncoder, m_EncodeFeatures[i]);
            GetCurrentDeviceNVEncCapability(m_hEncoder, m_EncodeFeatures[i]);
        }
    }
    return nvStatus;
}

const std::vector<NVEncCodecFeature>& NVEncCore::GetNVEncCapability() {
    if (m_EncodeFeatures.size() == 0) {
        createDeviceFeatureList();
    }
    return m_EncodeFeatures;
}

const NVEncCodecFeature *NVEncCore::getCodecFeature(const GUID& codec) {
    for (uint32_t i = 0; i < m_EncodeFeatures.size(); i++) {
        if (0 == memcmp(&m_EncodeFeatures[i].codec, &codec, sizeof(m_stCodecGUID))) {
            return &m_EncodeFeatures[i];
        }
    }
    return nullptr;
}

int NVEncCore::getCapLimit(NV_ENC_CAPS flag, const NVEncCodecFeature *codecFeature) {
    if (nullptr == codecFeature) {
        if (nullptr == (codecFeature = getCodecFeature(m_stCodecGUID))) {
            return 0;
        }
    }
    return get_value(flag, codecFeature->caps);
}

bool NVEncCore::checkProfileSupported(GUID profile, const NVEncCodecFeature *codecFeature) {
    if (nullptr == codecFeature) {
        if (nullptr == (codecFeature = getCodecFeature(m_stCodecGUID))) {
            return false;
        }
    }
    for (auto codecProf : codecFeature->profiles) {
        if (0 == memcmp(&profile, &codecProf, sizeof(codecProf))) {
            return true;
        }
    }
    return false;
}

bool NVEncCore::checkPresetSupported(GUID preset, const NVEncCodecFeature *codecFeature) {
    if (nullptr == codecFeature) {
        if (nullptr == (codecFeature = getCodecFeature(m_stCodecGUID))) {
            return false;
        }
    }
    for (auto codecPreset : codecFeature->presets) {
        if (0 == memcmp(&preset, &codecPreset, sizeof(codecPreset))) {
            return true;
        }
    }
    return false;
}

bool NVEncCore::checkSurfaceFmtSupported(NV_ENC_BUFFER_FORMAT surfaceFormat, const NVEncCodecFeature *codecFeature) {
    if (nullptr == codecFeature) {
        if (nullptr == (codecFeature = getCodecFeature(m_stCodecGUID))) {
            return false;
        }
    }
    for (auto codecFmt : codecFeature->surfaceFmt) {
        if (0 == memcmp(&surfaceFormat, &codecFmt, sizeof(surfaceFormat))) {
            return true;
        }
    }
    return false;
}

#pragma warning(push)
#pragma warning(disable: 4100)
NVENCSTATUS NVEncCore::CreateDecoder(const InEncodeVideoParam *inputParam) {
#if ENABLE_AVCUVID_READER
    if (m_pFileReader->inputCodecIsValid()) {
        m_cuvidDec.reset(new CuvidDecode());

        auto result = m_cuvidDec->InitDecode(m_ctxLock, &inputParam->input, &inputParam->vpp, m_pNVLog);
        if (result != CUDA_SUCCESS) {
            PrintMes(NV_LOG_ERROR, _T("failed to init decoder.\n"));
            return NV_ENC_ERR_UNSUPPORTED_PARAM;
        }
    }
#endif //#if ENABLE_AVCUVID_READER
    return NV_ENC_SUCCESS;
}
#pragma warning(pop)

NVENCSTATUS NVEncCore::SetInputParam(const InEncodeVideoParam *inputParam) {
    memcpy(&m_stEncConfig, &inputParam->encConfig, sizeof(m_stEncConfig));
    memcpy(&m_stPicStruct, &inputParam->picStruct, sizeof(m_stPicStruct));
    
    //コーデックの決定とチェックNV_ENC_PIC_PARAMS
    m_stCodecGUID = inputParam->codec == NV_ENC_H264 ? NV_ENC_CODEC_H264_GUID : NV_ENC_CODEC_HEVC_GUID;
    if (nullptr == getCodecFeature(m_stCodecGUID)) {
        PrintMes(NV_LOG_ERROR, FOR_AUO ? _T("指定されたコーデックはサポートされていません。\n") : _T("Selected codec is not supported.\n"));
        return NV_ENC_ERR_UNSUPPORTED_PARAM;
    }

    //プロファイルのチェック
    if (inputParam->codec == NV_ENC_HEVC) {
        //m_stEncConfig.profileGUIDはデフォルトではH.264のプロファイル情報
        //HEVCのプロファイル情報は、m_stEncConfig.encodeCodecConfig.hevcConfig.tierに保存されている
        m_stEncConfig.profileGUID = get_guid_from_value(m_stEncConfig.encodeCodecConfig.hevcConfig.tier, h265_profile_names);
        //NV_ENC_TIER_HEVC_MAIN10, NV_ENC_TIER_HEVC_MAIN444は独自拡張なので、エンコーダにはNV_ENC_TIER_HEVC_MAINとして渡す
        static const uint32_t CHECK_TIER[] = {
            NV_ENC_TIER_HEVC_MAIN, NV_ENC_TIER_HEVC_MAIN10, NV_ENC_TIER_HEVC_MAIN444
        };
        for (int i = 0; i < _countof(CHECK_TIER); i++) {
            if (m_stEncConfig.encodeCodecConfig.hevcConfig.tier == CHECK_TIER[i]) {
                m_stEncConfig.encodeCodecConfig.hevcConfig.tier = NV_ENC_TIER_HEVC_MAIN;
                break;
            }
        }
    }
    if (!checkProfileSupported(m_stEncConfig.profileGUID)) {
        PrintMes(NV_LOG_ERROR, FOR_AUO ? _T("指定されたプロファイルはサポートされていません。\n") : _T("Selected profile is not supported.\n"));
        return NV_ENC_ERR_UNSUPPORTED_PARAM;
    }

    //プリセットのチェック
    if (!checkPresetSupported(get_guid_from_value(inputParam->preset, preset_names))) {
        PrintMes(NV_LOG_ERROR, FOR_AUO ? _T("指定されたプリセットはサポートされていません。\n") : _T("Selected preset is not supported.\n"));
        return NV_ENC_ERR_UNSUPPORTED_PARAM;
    }

    //入力フォーマットはここでは気にしない
    //NV_ENC_BUFFER_FORMAT_NV12_TILED64x16
    //if (!checkSurfaceFmtSupported(NV_ENC_BUFFER_FORMAT_NV12_TILED64x16)) {
    //    PrintMes(NV_LOG_ERROR, FOR_AUO ? _T("入力フォーマットが決定できません。\n") : _T("Input format is not supported.\n"));
    //    return NV_ENC_ERR_UNSUPPORTED_PARAM;
    //}

    //バッファサイズ (固定で32として与える)
    m_uEncodeBufferCount = 32; // inputParam->inputBuffer;
    if (m_uEncodeBufferCount > MAX_ENCODE_QUEUE) {
#if FOR_AUO
        PrintMes(NV_LOG_ERROR, _T("入力バッファは多すぎます。: %d フレーム\n"), m_uEncodeBufferCount);
        PrintMes(NV_LOG_ERROR, _T("%d フレームまでに設定して下さい。\n"), MAX_ENCODE_QUEUE);
#else
        PrintMes(NV_LOG_ERROR, _T("Input frame of %d exceeds the maximum size allowed (%d).\n"), m_uEncodeBufferCount, MAX_ENCODE_QUEUE);
#endif
        return NV_ENC_ERR_UNSUPPORTED_PARAM;
    }

    //解像度の決定
    m_uEncWidth   = inputParam->input.width  - inputParam->input.crop.e.left - inputParam->input.crop.e.right;
    m_uEncHeight  = inputParam->input.height - inputParam->input.crop.e.bottom - inputParam->input.crop.e.up;

    if (inputParam->input.dstWidth && inputParam->input.dstHeight) {
#if ENABLE_AVCUVID_READER
        if (m_pFileReader->inputCodecIsValid()) {
            m_uEncWidth  = inputParam->input.dstWidth;
            m_uEncHeight = inputParam->input.dstHeight;
        } else
#endif
        if (m_uEncWidth != inputParam->input.width || m_uEncHeight != inputParam->input.height) {
            PrintMes(NV_LOG_ERROR, _T("resizing requires to be used with avcuvid reader.\n"));
            PrintMes(NV_LOG_ERROR, _T(" input %dx%d -> output %dx%d.\n"), m_uEncWidth, m_uEncHeight, inputParam->input.dstWidth, inputParam->input.dstHeight);
            return NV_ENC_ERR_UNSUPPORTED_PARAM;
        }
    }

    if (inputParam->input.rate <= 0 || inputParam->input.scale <= 0) {
        if (inputParam->input.type == NV_ENC_INPUT_RAW) {
            PrintMes(NV_LOG_ERROR, _T("Please set fps when using raw input.\n"));
        } else {
            PrintMes(NV_LOG_ERROR, _T("Invalid fps: %d/%d.\n"), inputParam->input.rate, inputParam->input.scale);
        }
        return NV_ENC_ERR_UNSUPPORTED_PARAM;
    }

    //picStructの設定
    m_stPicStruct = (inputParam->picStruct == 0) ? NV_ENC_PIC_STRUCT_FRAME : inputParam->picStruct;

    if (inputParam->vpp.deinterlace != cudaVideoDeinterlaceMode_Weave) {
#if ENABLE_AVCUVID_READER
        if (m_pFileReader->inputCodecIsValid()) {
            PrintMes(NV_LOG_ERROR, _T("vpp-deinterlace requires to be used with avcuvid reader.\n"));
            return NV_ENC_ERR_UNSUPPORTED_PARAM;
        }
#endif
        m_stPicStruct = NV_ENC_PIC_STRUCT_FRAME;
    }
    
    //制限事項チェック
    if (inputParam->input.width < 0 && inputParam->input.height < 0) {
        PrintMes(NV_LOG_ERROR, _T("%s: %dx%d\n"), FOR_AUO ? _T("解像度が無効です。") : _T("Invalid resolution."), inputParam->input.width, inputParam->input.height);
        return NV_ENC_ERR_UNSUPPORTED_PARAM;
    }
#if ENABLE_AVCUVID_READER
    if (inputParam->input.crop.e.left > 0 && m_pFileReader->inputCodecIsValid()) {
        PrintMes(NV_LOG_ERROR, _T("left crop is unsupported with avcuvid reader.\n"));
        return NV_ENC_ERR_UNSUPPORTED_PARAM;
    }
#endif
    if (   (int)inputParam->input.width  <= inputParam->input.crop.e.left + inputParam->input.crop.e.right
        && (int)inputParam->input.height <= inputParam->input.crop.e.up   + inputParam->input.crop.e.bottom) {
        PrintMes(NV_LOG_ERROR, _T("%s: %dx%d, Crop [%d,%d,%d,%d]\n"),
             FOR_AUO ? _T("Crop値が無効です。") : _T("Invalid crop value."), 
            inputParam->input.width, inputParam->input.height,
            inputParam->input.crop.c[0], inputParam->input.crop.c[1], inputParam->input.crop.c[2], inputParam->input.crop.c[3]);
        return NV_ENC_ERR_UNSUPPORTED_PARAM;
    }

    const int height_check_mask = 1 + 2 * !!is_interlaced(m_stPicStruct);
    if ((m_uEncWidth & 1) || (m_uEncHeight & height_check_mask)) {
        PrintMes(NV_LOG_ERROR, _T("%s: %dx%d\n"), FOR_AUO ? _T("解像度が無効です。") : _T("Invalid resolution."), m_uEncWidth, m_uEncHeight);
        PrintMes(NV_LOG_ERROR, FOR_AUO ? _T("縦横の解像度は2の倍数である必要があります。\n") : _T("Relosution of mod2 required.\n"));
        if (is_interlaced(m_stPicStruct)) {
            PrintMes(NV_LOG_ERROR, FOR_AUO ? _T("さらに、インタレ保持エンコードでは縦解像度は4の倍数である必要があります。\n") : _T("For interlaced encoding, mod4 is required for height.\n"));
        }
        return NV_ENC_ERR_UNSUPPORTED_PARAM;
    }
    if ((inputParam->input.crop.e.left & 1) || (inputParam->input.crop.e.right & 1)
        || (inputParam->input.crop.e.up & height_check_mask) || (inputParam->input.crop.e.bottom & height_check_mask)) {
        PrintMes(NV_LOG_ERROR, _T("%s: %dx%d, Crop [%d,%d,%d,%d]\n"),
             FOR_AUO ? _T("Crop値が無効です。") : _T("Invalid crop value."), 
            inputParam->input.width, inputParam->input.height,
            inputParam->input.crop.c[0], inputParam->input.crop.c[1], inputParam->input.crop.c[2], inputParam->input.crop.c[3]);
        PrintMes(NV_LOG_ERROR, FOR_AUO ? _T("Crop値は2の倍数である必要があります。\n") : _T("Crop value of mod2 required.\n"));
        if (is_interlaced(m_stPicStruct)) {
            PrintMes(NV_LOG_ERROR, FOR_AUO ? _T("さらに、インタレ保持エンコードでは縦Crop値は4の倍数である必要があります。\n") : _T("For interlaced encoding, mod4 is required for height.\n"));
        }
        return NV_ENC_ERR_UNSUPPORTED_PARAM;
    }
    if (inputParam->nTrimCount > 0 && !m_pFileReader->inputCodecIsValid()) {
        PrintMes(NV_LOG_ERROR, _T("trim is supported only with avcuvid reader.\n"));
        return NV_ENC_ERR_UNSUPPORTED_PARAM;
    }
    if (inputParam->nAVSyncMode && inputParam->nTrimCount > 0) {
        PrintMes(NV_LOG_ERROR, _T("avsync forcecfr + trim is not supported.\n"));
        return NV_ENC_ERR_UNSUPPORTED_PARAM;
    }
    //環境による制限
    auto error_resolution_over_limit = [&](const TCHAR *feature, uint32_t featureValue, NV_ENC_CAPS featureID) {
        const TCHAR *error_mes = FOR_AUO ? _T("解像度が上限を超えています。") : _T("Resolution is over limit.");
        if (nullptr == feature)
            PrintMes(NV_LOG_ERROR, _T("%s: %dx%d [上限: %dx%d]\n"), error_mes, m_uEncWidth, m_uEncHeight, getCapLimit(NV_ENC_CAPS_WIDTH_MAX), getCapLimit(NV_ENC_CAPS_HEIGHT_MAX));
        else
            PrintMes(NV_LOG_ERROR, _T("%s: %dx%d, [%s]: %d [上限: %d]\n"), error_mes, m_uEncWidth, m_uEncHeight, feature, featureValue, getCapLimit(featureID));
    };

    if (m_uEncWidth > (uint32_t)getCapLimit(NV_ENC_CAPS_WIDTH_MAX) || m_uEncHeight > (uint32_t)getCapLimit(NV_ENC_CAPS_HEIGHT_MAX)) {
        error_resolution_over_limit(nullptr, 0, (NV_ENC_CAPS)0);
        return NV_ENC_ERR_UNSUPPORTED_PARAM;
    }
    uint32_t heightMod = 16 * (1 + !!is_interlaced(m_stPicStruct));
    uint32_t targetMB = ((m_uEncWidth + 15) / 16) * ((m_uEncHeight + (heightMod - 1)) / heightMod);
    if (targetMB > (uint32_t)getCapLimit(NV_ENC_CAPS_MB_NUM_MAX)) {
        error_resolution_over_limit(_T("MB"), targetMB, NV_ENC_CAPS_MB_NUM_MAX);
        return NV_ENC_ERR_UNSUPPORTED_PARAM;
    }
    //MB/sの制限は特にチェックする必要がなさそう
    //uint32_t targetMBperSec = (targetMB * inputParam->input.rate + inputParam->input.scale - 1) / inputParam->input.scale;
    //if (targetMBperSec > (uint32_t)getCapLimit(NV_ENC_CAPS_MB_PER_SEC_MAX)) {
    //    error_resolution_over_limit(_T("MB/s"), targetMBperSec, NV_ENC_CAPS_MB_PER_SEC_MAX);
    //    return NV_ENC_ERR_UNSUPPORTED_PARAM;
    //}

    auto error_feature_unsupported = [&](int log_level, const TCHAR *feature_name) {
        PrintMes(log_level, FOR_AUO ? _T("%sはサポートされていません。\n") : _T("%s unsupported.\n"), feature_name);
    };

    if (is_interlaced(m_stPicStruct) && !getCapLimit(NV_ENC_CAPS_SUPPORT_FIELD_ENCODING)) {
        return NV_ENC_ERR_UNSUPPORTED_PARAM;
    }
    if (m_stEncConfig.rcParams.rateControlMode != (m_stEncConfig.rcParams.rateControlMode & getCapLimit(NV_ENC_CAPS_SUPPORTED_RATECONTROL_MODES))) {
        error_feature_unsupported(NV_LOG_ERROR, FOR_AUO ? _T("選択されたレート制御モード") : _T("Selected encode mode"));
        return NV_ENC_ERR_UNSUPPORTED_PARAM;
    }
    if (m_stEncConfig.frameIntervalP < 0) {
        PrintMes(NV_LOG_ERROR, _T("%s: %d\n"),
            FOR_AUO ? _T("Bフレーム設定が無効です。正の値を使用してください。\n") : _T("B frame settings are invalid. Please use a number > 0.\n"),
            m_stEncConfig.frameIntervalP - 1);
        return NV_ENC_ERR_UNSUPPORTED_PARAM;
    }
    if (m_stEncConfig.rcParams.enableLookahead && !getCapLimit(NV_ENC_CAPS_SUPPORT_LOOKAHEAD)) {
        error_feature_unsupported(NV_LOG_WARN, _T("Lookahead"));
        m_stEncConfig.rcParams.enableLookahead = 0;
        m_stEncConfig.rcParams.lookaheadDepth = 0;
        m_stEncConfig.rcParams.disableBadapt = 0;
        m_stEncConfig.rcParams.disableIadapt = 0;
    }
    if (m_stEncConfig.rcParams.enableTemporalAQ && !getCapLimit(NV_ENC_CAPS_SUPPORT_TEMPORAL_AQ)) {
        error_feature_unsupported(NV_LOG_WARN, _T("Temporal AQ"));
        m_stEncConfig.rcParams.enableTemporalAQ = 0;
    }
    if (inputParam->bluray) {
        if (inputParam->codec == NV_ENC_HEVC) {
            PrintMes(NV_LOG_ERROR, FOR_AUO ? _T("HEVCではBluray用出力はサポートされていません。\n") : _T("Bluray output is not supported for HEVC codec.\n"));
            return NV_ENC_ERR_UNSUPPORTED_PARAM;
        }
        const auto VBR_RC_LIST = make_array<NV_ENC_PARAMS_RC_MODE>(NV_ENC_PARAMS_RC_VBR, NV_ENC_PARAMS_RC_VBR_MINQP, NV_ENC_PARAMS_RC_2_PASS_VBR, NV_ENC_PARAMS_RC_CBR, NV_ENC_PARAMS_RC_CBR2);
        if (std::find(VBR_RC_LIST.begin(), VBR_RC_LIST.end(), inputParam->encConfig.rcParams.rateControlMode) == VBR_RC_LIST.end()) {
            PrintMes(NV_LOG_ERROR, FOR_AUO ? _T("Bluray用出力では、VBRモードを使用してください。\n") :  _T("Please use VBR mode for bluray output.\n"));
            return NV_ENC_ERR_UNSUPPORTED_PARAM;
        }
        if (!getCapLimit(NV_ENC_CAPS_SUPPORT_CUSTOM_VBV_BUF_SIZE)) {
            error_feature_unsupported(NV_LOG_ERROR, FOR_AUO ? _T("VBVバッファサイズの指定") : _T("Custom VBV Bufsize"));
            PrintMes(NV_LOG_ERROR, FOR_AUO ? _T("Bluray用出力を行えません。\n") :  _T("Therfore you cannot output for bluray.\n"));
            return NV_ENC_ERR_UNSUPPORTED_PARAM;
        }
    }
    if (m_stEncConfig.frameIntervalP - 1 > getCapLimit(NV_ENC_CAPS_NUM_MAX_BFRAMES)) {
        m_stEncConfig.frameIntervalP = getCapLimit(NV_ENC_CAPS_NUM_MAX_BFRAMES) + 1;
        PrintMes(NV_LOG_WARN, FOR_AUO ? _T("Bフレームの最大数は%dです。\n") : _T("Max B frames are %d frames.\n"), getCapLimit(NV_ENC_CAPS_NUM_MAX_BFRAMES));
    }
    if (inputParam->codec == NV_ENC_H264) {
        if (NV_ENC_H264_ENTROPY_CODING_MODE_CABAC == m_stEncConfig.encodeCodecConfig.h264Config.entropyCodingMode && !getCapLimit(NV_ENC_CAPS_SUPPORT_CABAC)) {
            m_stEncConfig.encodeCodecConfig.h264Config.entropyCodingMode = NV_ENC_H264_ENTROPY_CODING_MODE_CAVLC;
            error_feature_unsupported(NV_LOG_WARN, _T("CABAC"));
        }
        if (NV_ENC_H264_FMO_ENABLE == m_stEncConfig.encodeCodecConfig.h264Config.fmoMode && !getCapLimit(NV_ENC_CAPS_SUPPORT_FMO)) {
            m_stEncConfig.encodeCodecConfig.h264Config.fmoMode = NV_ENC_H264_FMO_DISABLE;
            error_feature_unsupported(NV_LOG_WARN, _T("FMO"));
        }
        if (NV_ENC_H264_BDIRECT_MODE_TEMPORAL & m_stEncConfig.encodeCodecConfig.h264Config.bdirectMode && !getCapLimit(NV_ENC_CAPS_SUPPORT_BDIRECT_MODE)) {
            m_stEncConfig.encodeCodecConfig.h264Config.bdirectMode = NV_ENC_H264_BDIRECT_MODE_DISABLE;
            error_feature_unsupported(NV_LOG_WARN, _T("B Direct mode"));
        }
        if (NV_ENC_H264_ADAPTIVE_TRANSFORM_ENABLE != m_stEncConfig.encodeCodecConfig.h264Config.adaptiveTransformMode && !getCapLimit(NV_ENC_CAPS_SUPPORT_ADAPTIVE_TRANSFORM)) {
            m_stEncConfig.encodeCodecConfig.h264Config.adaptiveTransformMode = NV_ENC_H264_ADAPTIVE_TRANSFORM_DISABLE;
            error_feature_unsupported(NV_LOG_WARN, _T("Adaptive Tranform"));
        }
    }
    if ((NV_ENC_MV_PRECISION_QUARTER_PEL == m_stEncConfig.mvPrecision) && !getCapLimit(NV_ENC_CAPS_SUPPORT_QPELMV)) {
        m_stEncConfig.mvPrecision = NV_ENC_MV_PRECISION_HALF_PEL;
        error_feature_unsupported(NV_LOG_WARN, FOR_AUO ? _T("1/4画素精度MV探索") : _T("Q-Pel MV"));
    }
    if (0 != m_stEncConfig.rcParams.vbvBufferSize && !getCapLimit(NV_ENC_CAPS_SUPPORT_CUSTOM_VBV_BUF_SIZE)) {
        m_stEncConfig.rcParams.vbvBufferSize = 0;
        error_feature_unsupported(NV_LOG_WARN, FOR_AUO ? _T("VBVバッファサイズの指定") : _T("Custom VBV Bufsize"));
    }
    if (inputParam->yuv444 || inputParam->lossless) {
#if ENABLE_AVCUVID_READER
        if (m_pFileReader->inputCodecIsValid()) {
            PrintMes(NV_LOG_ERROR, _T("high444 not supported avcuvid reader.\n"));
            return NV_ENC_ERR_UNSUPPORTED_PARAM;
        }
#endif
    }
    if (inputParam->lossless) {
        if (inputParam->codec != NV_ENC_H264) {
            PrintMes(NV_LOG_ERROR, FOR_AUO ? _T("lossless出力はH.264エンコード時のみ使用できます。\n") : _T("lossless output is only for H.264 codec.\n"));
            return NV_ENC_ERR_UNSUPPORTED_PARAM;
        }
        if (!getCapLimit(NV_ENC_CAPS_SUPPORT_LOSSLESS_ENCODE)) {
            error_feature_unsupported(NV_LOG_ERROR, _T("lossless"));
            return NV_ENC_ERR_UNSUPPORTED_PARAM;
        }
    }
    if (inputParam->codec == NV_ENC_HEVC) {
        if (   ( m_stEncConfig.encodeCodecConfig.hevcConfig.maxCUSize != NV_ENC_HEVC_CUSIZE_AUTOSELECT
              && m_stEncConfig.encodeCodecConfig.hevcConfig.maxCUSize != NV_ENC_HEVC_CUSIZE_32x32)
            || ( m_stEncConfig.encodeCodecConfig.hevcConfig.minCUSize != NV_ENC_HEVC_CUSIZE_AUTOSELECT
              && m_stEncConfig.encodeCodecConfig.hevcConfig.minCUSize != NV_ENC_HEVC_CUSIZE_8x8)) {
            PrintMes(NV_LOG_WARN, _T("it is not recommended to use --cu-max or --cu-min, leaving it auto will enhance video quality.\n"));
        }
    }
    //自動決定パラメータ
    if (0 == m_stEncConfig.gopLength) {
        m_stEncConfig.gopLength = (int)(inputParam->input.rate / (double)inputParam->input.scale + 0.5) * 10;
    }
    if (m_stEncConfig.encodeCodecConfig.h264Config.enableLTR && m_stEncConfig.encodeCodecConfig.h264Config.ltrNumFrames == 0) {
        m_stEncConfig.encodeCodecConfig.h264Config.ltrNumFrames = m_stEncConfig.encodeCodecConfig.h264Config.maxNumRefFrames;
    }
    if (m_stEncConfig.encodeCodecConfig.hevcConfig.enableLTR && m_stEncConfig.encodeCodecConfig.hevcConfig.ltrNumFrames == 0) {
        m_stEncConfig.encodeCodecConfig.hevcConfig.ltrNumFrames = m_stEncConfig.encodeCodecConfig.hevcConfig.maxNumRefFramesInDPB;
    }
    //SAR自動設定
    auto par = std::make_pair(inputParam->par[0], inputParam->par[1]);
    if ((!inputParam->par[0] || !inputParam->par[1]) //SAR比の指定がない
        && inputParam->input.sar[0] && inputParam->input.sar[1] //入力側からSAR比を取得ずみ
        && (m_uEncWidth == inputParam->input.width && m_uEncHeight == inputParam->input.height)) {//リサイズは行われない
        par = std::make_pair(inputParam->input.sar[0], inputParam->input.sar[1]);
    }
    adjust_sar(&par.first, &par.second, m_uEncWidth, m_uEncHeight);

    //色空間設定自動
    int frame_height = m_uEncHeight;
    auto apply_auto_colormatrix = [frame_height](uint32_t& value, const CX_DESC *list) {
        if (COLOR_VALUE_AUTO == value)
            value = list[(frame_height >= HD_HEIGHT_THRESHOLD) ? HD_INDEX : SD_INDEX].value;
    };

    apply_auto_colormatrix(m_stEncConfig.encodeCodecConfig.h264Config.h264VUIParameters.colourPrimaries,         list_colorprim);
    apply_auto_colormatrix(m_stEncConfig.encodeCodecConfig.h264Config.h264VUIParameters.transferCharacteristics, list_transfer);
    apply_auto_colormatrix(m_stEncConfig.encodeCodecConfig.h264Config.h264VUIParameters.colourMatrix,            list_colormatrix);

    INIT_CONFIG(m_stCreateEncodeParams, NV_ENC_INITIALIZE_PARAMS);
    m_stCreateEncodeParams.encodeConfig        = &m_stEncConfig;
    m_stCreateEncodeParams.encodeHeight        = m_uEncHeight;
    m_stCreateEncodeParams.encodeWidth         = m_uEncWidth;
    m_stCreateEncodeParams.darHeight           = m_uEncHeight;
    m_stCreateEncodeParams.darWidth            = m_uEncWidth;
    get_dar_pixels(&m_stCreateEncodeParams.darWidth, &m_stCreateEncodeParams.darHeight, par.first, par.second);

    m_stCreateEncodeParams.maxEncodeHeight     = m_uEncHeight;
    m_stCreateEncodeParams.maxEncodeWidth      = m_uEncWidth;

    m_stCreateEncodeParams.frameRateNum        = inputParam->input.rate;
    m_stCreateEncodeParams.frameRateDen        = inputParam->input.scale;
    if (inputParam->vpp.deinterlace == cudaVideoDeinterlaceMode_Bob) {
        m_stCreateEncodeParams.frameRateNum *= 2;
    }
    //Fix me add theading model
    m_stCreateEncodeParams.enableEncodeAsync   = true;
    m_stCreateEncodeParams.enablePTD           = true;
    m_stCreateEncodeParams.encodeGUID          = m_stCodecGUID;
    m_stCreateEncodeParams.presetGUID          = preset_names[inputParam->preset].id;

    if (inputParam->codec == NV_ENC_HEVC) {
        //整合性チェック (一般, H.265/HEVC)
        m_stCreateEncodeParams.encodeConfig->encodeCodecConfig.hevcConfig.idrPeriod = m_stCreateEncodeParams.encodeConfig->gopLength;
        //YUV444出力
        if (inputParam->yuv444 || inputParam->lossless) {
            m_stCreateEncodeParams.encodeConfig->encodeCodecConfig.hevcConfig.chromaFormatIDC = 3;
            //m_stCreateEncodeParams.encodeConfig->encodeCodecConfig.h264Config.separateColourPlaneFlag = 1;
            m_stCreateEncodeParams.encodeConfig->profileGUID = NV_ENC_HEVC_PROFILE_FREXT_GUID;
        }
        if (m_stCreateEncodeParams.encodeConfig->encodeCodecConfig.hevcConfig.pixelBitDepthMinus8 > 0) {
            m_stCreateEncodeParams.encodeConfig->profileGUID = (inputParam->yuv444) ? NV_ENC_HEVC_PROFILE_FREXT_GUID : NV_ENC_HEVC_PROFILE_MAIN10_GUID;
        }
        //整合性チェック (HEVC VUI)
        m_stCreateEncodeParams.encodeConfig->encodeCodecConfig.hevcConfig.hevcVUIParameters.overscanInfoPresentFlag =
            (m_stCreateEncodeParams.encodeConfig->encodeCodecConfig.hevcConfig.hevcVUIParameters.overscanInfo) ? 1 : 0;

        m_stCreateEncodeParams.encodeConfig->encodeCodecConfig.hevcConfig.hevcVUIParameters.videoSignalTypePresentFlag =
            (get_cx_value(list_videoformat, _T("undef")) != (int)m_stCreateEncodeParams.encodeConfig->encodeCodecConfig.hevcConfig.hevcVUIParameters.videoFormat
                || m_stCreateEncodeParams.encodeConfig->encodeCodecConfig.hevcConfig.hevcVUIParameters.videoFullRangeFlag) ? 1 : 0;
        if (!m_stCreateEncodeParams.encodeConfig->encodeCodecConfig.hevcConfig.hevcVUIParameters.videoSignalTypePresentFlag) {
            m_stCreateEncodeParams.encodeConfig->encodeCodecConfig.hevcConfig.hevcVUIParameters.videoFormat = 0;
        }

        m_stCreateEncodeParams.encodeConfig->encodeCodecConfig.hevcConfig.hevcVUIParameters.colourDescriptionPresentFlag =
            (      get_cx_value(list_colorprim,   _T("undef")) != (int)m_stCreateEncodeParams.encodeConfig->encodeCodecConfig.hevcConfig.hevcVUIParameters.colourPrimaries
                || get_cx_value(list_transfer,    _T("undef")) != (int)m_stCreateEncodeParams.encodeConfig->encodeCodecConfig.hevcConfig.hevcVUIParameters.transferCharacteristics
                || get_cx_value(list_colormatrix, _T("undef")) != (int)m_stCreateEncodeParams.encodeConfig->encodeCodecConfig.hevcConfig.hevcVUIParameters.colourMatrix) ? 1 : 0;
        if (!m_stCreateEncodeParams.encodeConfig->encodeCodecConfig.hevcConfig.hevcVUIParameters.colourDescriptionPresentFlag) {
            m_stCreateEncodeParams.encodeConfig->encodeCodecConfig.hevcConfig.hevcVUIParameters.colourPrimaries = 0;
            m_stCreateEncodeParams.encodeConfig->encodeCodecConfig.hevcConfig.hevcVUIParameters.transferCharacteristics = 0;
            m_stCreateEncodeParams.encodeConfig->encodeCodecConfig.hevcConfig.hevcVUIParameters.colourMatrix = 0;
        }
    } else if (inputParam->codec == NV_ENC_H264) {
        //Bluray 互換出力
        if (inputParam->bluray) {
            m_stCreateEncodeParams.encodeConfig->encodeCodecConfig.h264Config.outputPictureTimingSEI = 1;
            m_stCreateEncodeParams.encodeConfig->encodeCodecConfig.h264Config.outputRecoveryPointSEI = 1;
            m_stCreateEncodeParams.encodeConfig->encodeCodecConfig.h264Config.outputAUD = 1;
            m_stCreateEncodeParams.encodeConfig->encodeCodecConfig.h264Config.sliceMode = 3;
            m_stCreateEncodeParams.encodeConfig->encodeCodecConfig.h264Config.sliceModeData = 4;
            m_stCreateEncodeParams.encodeConfig->encodeCodecConfig.h264Config.level = std::min<uint32_t>(m_stCreateEncodeParams.encodeConfig->encodeCodecConfig.h264Config.level, NV_ENC_LEVEL_H264_41);
            m_stCreateEncodeParams.encodeConfig->rcParams.maxBitRate = std::min<uint32_t>(m_stCreateEncodeParams.encodeConfig->rcParams.maxBitRate, 40000 * 1000);
            if (m_stCreateEncodeParams.encodeConfig->rcParams.vbvBufferSize == 0) {
                m_stCreateEncodeParams.encodeConfig->rcParams.vbvBufferSize = m_stCreateEncodeParams.encodeConfig->rcParams.maxBitRate;
            }
            m_stCreateEncodeParams.encodeConfig->rcParams.vbvInitialDelay = m_stCreateEncodeParams.encodeConfig->rcParams.vbvBufferSize / 2;
            m_stCreateEncodeParams.encodeConfig->rcParams.averageBitRate = std::min(m_stCreateEncodeParams.encodeConfig->rcParams.averageBitRate, m_stCreateEncodeParams.encodeConfig->rcParams.maxBitRate);
            m_stCreateEncodeParams.encodeConfig->frameIntervalP = std::min(m_stCreateEncodeParams.encodeConfig->frameIntervalP, 3+1);
            m_stCreateEncodeParams.encodeConfig->gopLength = (std::min(m_stCreateEncodeParams.encodeConfig->gopLength, 30u) / m_stCreateEncodeParams.encodeConfig->frameIntervalP) * m_stCreateEncodeParams.encodeConfig->frameIntervalP;
        }

        //ロスレス出力
        if (inputParam->lossless) {
            m_stCreateEncodeParams.encodeConfig->encodeCodecConfig.h264Config.qpPrimeYZeroTransformBypassFlag = 1;
            m_stCreateEncodeParams.encodeConfig->rcParams.rateControlMode = NV_ENC_PARAMS_RC_CONSTQP;
            m_stCreateEncodeParams.encodeConfig->rcParams.maxQP.qpInterB = 0;
            m_stCreateEncodeParams.encodeConfig->rcParams.constQP.qpIntra = 0;
            m_stCreateEncodeParams.encodeConfig->rcParams.constQP.qpInterP = 0;
            m_stCreateEncodeParams.encodeConfig->rcParams.constQP.qpInterB = 0;
            m_stCreateEncodeParams.encodeConfig->rcParams.enableMinQP = 0;
            m_stCreateEncodeParams.encodeConfig->rcParams.enableMaxQP = 0;
            m_stCreateEncodeParams.encodeConfig->rcParams.minQP.qpIntra = 0;
            m_stCreateEncodeParams.encodeConfig->rcParams.minQP.qpInterP = 0;
            m_stCreateEncodeParams.encodeConfig->rcParams.minQP.qpInterB = 0;
            m_stCreateEncodeParams.encodeConfig->rcParams.maxQP.qpIntra = 0;
            m_stCreateEncodeParams.encodeConfig->rcParams.maxQP.qpInterP = 0;
            m_stCreateEncodeParams.encodeConfig->rcParams.maxQP.qpInterB = 0;
        }
        //YUV444出力
        if (inputParam->yuv444 || inputParam->lossless) {
            m_stCreateEncodeParams.encodeConfig->encodeCodecConfig.h264Config.chromaFormatIDC = 3;
            //m_stCreateEncodeParams.encodeConfig->encodeCodecConfig.h264Config.separateColourPlaneFlag = 1;
            m_stCreateEncodeParams.encodeConfig->profileGUID = NV_ENC_H264_PROFILE_HIGH_444_GUID;
        }

        //整合性チェック (一般, H.264/AVC)
        m_stCreateEncodeParams.encodeConfig->frameFieldMode = (m_stPicStruct == NV_ENC_PIC_STRUCT_FRAME) ? NV_ENC_PARAMS_FRAME_FIELD_MODE_FRAME : NV_ENC_PARAMS_FRAME_FIELD_MODE_FIELD;
        //m_stCreateEncodeParams.encodeConfig->encodeCodecConfig.h264Config.entropyCodingMode = (m_stEncoderInput[0].profile > 66) ? NV_ENC_H264_ENTROPY_CODING_MODE_CABAC : NV_ENC_H264_ENTROPY_CODING_MODE_CAVLC;
        m_stCreateEncodeParams.encodeConfig->encodeCodecConfig.h264Config.idrPeriod = m_stCreateEncodeParams.encodeConfig->gopLength;
        if (m_stCreateEncodeParams.encodeConfig->frameIntervalP - 1 <= 0) {
            m_stCreateEncodeParams.encodeConfig->encodeCodecConfig.h264Config.bdirectMode = NV_ENC_H264_BDIRECT_MODE_DISABLE;
        }

        //整合性チェック (H.264 VUI)
        m_stCreateEncodeParams.encodeConfig->encodeCodecConfig.h264Config.h264VUIParameters.overscanInfoPresentFlag =
            (m_stCreateEncodeParams.encodeConfig->encodeCodecConfig.h264Config.h264VUIParameters.overscanInfo) ? 1 : 0;

        m_stCreateEncodeParams.encodeConfig->encodeCodecConfig.h264Config.h264VUIParameters.videoSignalTypePresentFlag =
            (get_cx_value(list_videoformat, _T("undef")) != (int)m_stCreateEncodeParams.encodeConfig->encodeCodecConfig.h264Config.h264VUIParameters.videoFormat
            || m_stCreateEncodeParams.encodeConfig->encodeCodecConfig.h264Config.h264VUIParameters.videoFullRangeFlag) ? 1 : 0;

        m_stCreateEncodeParams.encodeConfig->encodeCodecConfig.h264Config.h264VUIParameters.colourDescriptionPresentFlag =
            (  get_cx_value(list_colorprim,   _T("undef")) != (int)m_stCreateEncodeParams.encodeConfig->encodeCodecConfig.h264Config.h264VUIParameters.colourPrimaries
            || get_cx_value(list_transfer,    _T("undef")) != (int)m_stCreateEncodeParams.encodeConfig->encodeCodecConfig.h264Config.h264VUIParameters.transferCharacteristics
            || get_cx_value(list_colormatrix, _T("undef")) != (int)m_stCreateEncodeParams.encodeConfig->encodeCodecConfig.h264Config.h264VUIParameters.colourMatrix) ? 1 : 0;
    }

    return NV_ENC_SUCCESS;
}

NVENCSTATUS NVEncCore::CreateEncoder(const InEncodeVideoParam *inputParam) {
    NVENCSTATUS nvStatus;

    if (NV_ENC_SUCCESS != (nvStatus = SetInputParam(inputParam)))
        return nvStatus;
    PrintMes(NV_LOG_DEBUG, _T("SetInputParam: Success.\n"));

    if (NV_ENC_SUCCESS != (nvStatus = m_pEncodeAPI->nvEncInitializeEncoder(m_hEncoder, &m_stCreateEncodeParams))) {
        PrintMes(NV_LOG_ERROR,
            _T("%s: %d (%s)\n"), FOR_AUO ? _T("エンコーダの初期化に失敗しました。\n") : _T("Failed to Initialize the encoder\n."),
            nvStatus, char_to_tstring(_nvencGetErrorEnum(nvStatus)).c_str());
        return nvStatus;
    }
    PrintMes(NV_LOG_DEBUG, _T("m_pEncodeAPI->nvEncInitializeEncoder: Success.\n"));

    return nvStatus;
}

NVENCSTATUS NVEncCore::InitEncode(InEncodeVideoParam *inputParam) {
    NVENCSTATUS nvStatus = NV_ENC_SUCCESS;

    if (inputParam->lossless) {
        inputParam->yuv444 = TRUE;
    }
    const bool bOutputHighBitDepth = inputParam->codec == NV_ENC_HEVC && inputParam->encConfig.encodeCodecConfig.hevcConfig.pixelBitDepthMinus8 > 0;
    if (bOutputHighBitDepth) {
        inputParam->input.csp = (inputParam->yuv444) ? NV_ENC_CSP_YUV444_10 : NV_ENC_CSP_P010;
    } else {
        inputParam->input.csp = (inputParam->yuv444) ? NV_ENC_CSP_YUV444 : NV_ENC_CSP_NV12;
    }
    m_nAVSyncMode = inputParam->nAVSyncMode;
    m_nProcSpeedLimit = inputParam->nProcSpeedLimit;

    //入力ファイルを開き、入力情報も取得
    if (NV_ENC_SUCCESS != (nvStatus = InitInput(inputParam))) {
        PrintMes(NV_LOG_ERROR, FOR_AUO ? _T("入力ファイルを開けませんでした。\n") : _T("Failed to open input file.\n"));
        return nvStatus;
    }
    PrintMes(NV_LOG_DEBUG, _T("InitInput: Success.\n"));
    
    //作成したデバイスの情報をfeature取得
    if (NV_ENC_SUCCESS != (nvStatus = createDeviceFeatureList(false))) {
        return nvStatus;
    }
    PrintMes(NV_LOG_DEBUG, _T("createDeviceFeatureList: Success.\n"));

    //必要ならデコーダを作成
    if (NV_ENC_SUCCESS != (nvStatus = CreateDecoder(inputParam))) {
        return nvStatus;
    }
    PrintMes(NV_LOG_DEBUG, _T("CreateDecoder: Success.\n"));

    //エンコーダにパラメータを渡し、初期化
    if (NV_ENC_SUCCESS != (nvStatus = CreateEncoder(inputParam))) {
        return nvStatus;
    }
    PrintMes(NV_LOG_DEBUG, _T("CreateEncoder: Success.\n"));
    
    //入出力用メモリ確保
    NV_ENC_BUFFER_FORMAT encBufferFormat;
    if (bOutputHighBitDepth) {
        encBufferFormat = (inputParam->yuv444) ? NV_ENC_BUFFER_FORMAT_YUV444_10BIT : NV_ENC_BUFFER_FORMAT_YUV420_10BIT;
    } else {
        encBufferFormat = (inputParam->yuv444) ? NV_ENC_BUFFER_FORMAT_YUV444_PL : NV_ENC_BUFFER_FORMAT_NV12_PL;
    }
    m_nAVSyncMode = inputParam->nAVSyncMode;
    if (NV_ENC_SUCCESS != (nvStatus = AllocateIOBuffers(m_uEncWidth, m_uEncHeight, encBufferFormat))) {
        return nvStatus;
    }
    PrintMes(NV_LOG_DEBUG, _T("AllocateIOBuffers: Success.\n"));

    //出力ファイルを開く
    if (NV_ENC_SUCCESS != (nvStatus = InitOutput(inputParam))) {
        PrintMes(NV_LOG_ERROR, FOR_AUO ? _T("出力ファイルのオープンに失敗しました。: \"%s\"\n") : _T("Failed to open output file: \"%s\"\n"), inputParam->outputFilename.c_str());
        return nvStatus;
    }
    PrintMes(NV_LOG_DEBUG, _T("InitOutput: Success.\n"), inputParam->outputFilename.c_str());
    return nvStatus;
}

NVENCSTATUS NVEncCore::Initialize(InEncodeVideoParam *inputParam) {
    NVENCSTATUS nvStatus = NV_ENC_SUCCESS;

    InitLog(inputParam);

    if (NULL == m_hinstLib) {
        if (NULL == (m_hinstLib = LoadLibrary(NVENCODE_API_DLL))) {
#if FOR_AUO
            PrintMes(NV_LOG_ERROR, _T("%sがシステムに存在しません。\n"), NVENCODE_API_DLL);
            PrintMes(NV_LOG_ERROR, _T("NVIDIAのドライバが動作条件を満たしているか確認して下さい。"));
#else
            PrintMes(NV_LOG_ERROR, _T("%s does not exists in your system.\n"), NVENCODE_API_DLL);
            PrintMes(NV_LOG_ERROR, _T("Please check if the GPU driver is propery installed."));
#endif
            return NV_ENC_ERR_OUT_OF_MEMORY;
        }
    }
    PrintMes(NV_LOG_DEBUG, _T("Loaded %s.\n"), NVENCODE_API_DLL);
    
    MYPROC nvEncodeAPICreateInstance; // function pointer to create instance in nvEncodeAPI
    if (NULL == (nvEncodeAPICreateInstance = (MYPROC)GetProcAddress(m_hinstLib, "NvEncodeAPICreateInstance"))) {
        PrintMes(NV_LOG_ERROR, FOR_AUO ? _T("NvEncodeAPICreateInstanceのアドレス取得に失敗しました。\n") : _T("Failed to get address of NvEncodeAPICreateInstance.\n"));
        return NV_ENC_ERR_OUT_OF_MEMORY;
    }

    if (NULL == (m_pEncodeAPI = new NV_ENCODE_API_FUNCTION_LIST)) {
        PrintMes(NV_LOG_ERROR, FOR_AUO ? _T("NV_ENCODE_API_FUNCTION_LIST用のメモリ確保に失敗しました。\n") : _T("Failed to allocate memory for NV_ENCODE_API_FUNCTION_LIST.\n"));
        return NV_ENC_ERR_OUT_OF_MEMORY;
    }

    memset(m_pEncodeAPI, 0, sizeof(NV_ENCODE_API_FUNCTION_LIST));
    m_pEncodeAPI->version = NV_ENCODE_API_FUNCTION_LIST_VER;

    if (NV_ENC_SUCCESS != (nvStatus = nvEncodeAPICreateInstance(m_pEncodeAPI))) {
        if (nvStatus == NV_ENC_ERR_INVALID_VERSION) {
#if FOR_AUO
            PrintMes(NV_LOG_ERROR, _T("nvEncodeAPIのインスタンス作成に失敗しました。ドライバのバージョンが古い可能性があります。\n"));
            PrintMes(NV_LOG_ERROR, _T("最新のドライバに更新して試してみてください。\n"));
#else
            PrintMes(NV_LOG_ERROR, _T("Failed to create instance of nvEncodeAPI, please consider updating your GPU driver.\n"));
#endif
        } else {
            NVPrintFuncError(_T("nvEncodeAPICreateInstance"), nvStatus);
        }
        return nvStatus;
    }
    PrintMes(NV_LOG_DEBUG, _T("nvEncodeAPICreateInstance: Success.\n"));

    //m_pDeviceを初期化
    if (NV_ENC_SUCCESS != (nvStatus = InitDevice(inputParam))) {
        return nvStatus;
    }
    PrintMes(NV_LOG_DEBUG, _T("InitDevice: Success.\n"));

    if (NV_ENC_SUCCESS != (nvStatus = NvEncOpenEncodeSessionEx(m_pDevice, NV_ENC_DEVICE_TYPE_CUDA))) {
        return nvStatus;
    }
    PrintMes(NV_LOG_DEBUG, _T("NvEncOpenEncodeSessionEx: Success.\n"));
    return nvStatus;
}

NVENCSTATUS NVEncCore::NvEncEncodeFrame(EncodeBuffer *pEncodeBuffer, uint64_t timestamp) {
    
    NV_ENC_PIC_PARAMS encPicParams;
    INIT_CONFIG(encPicParams, NV_ENC_PIC_PARAMS);

    encPicParams.inputBuffer = pEncodeBuffer->stInputBfr.hInputSurface;
    encPicParams.bufferFmt = pEncodeBuffer->stInputBfr.bufferFmt;
    encPicParams.inputWidth = m_uEncWidth;
    encPicParams.inputHeight = m_uEncHeight;
    encPicParams.outputBitstream = pEncodeBuffer->stOutputBfr.hBitstreamBuffer;
    encPicParams.completionEvent = pEncodeBuffer->stOutputBfr.hOutputEvent;
    encPicParams.inputTimeStamp = timestamp;
    encPicParams.pictureStruct = m_stPicStruct;
    //encPicParams.qpDeltaMap = qpDeltaMapArray;
    //encPicParams.qpDeltaMapSize = qpDeltaMapArraySize;


    //if (encPicCommand)
    //{
    //    if (encPicCommand->bForceIDR)
    //    {
    //        encPicParams.encodePicFlags |= NV_ENC_PIC_FLAG_FORCEIDR;
    //    }

    //    if (encPicCommand->bForceIntraRefresh)
    //    {
    //        if (codecGUID == NV_ENC_CODEC_HEVC_GUID)
    //        {
    //            encPicParams.codecPicParams.hevcPicParams.forceIntraRefreshWithFrameCnt = encPicCommand->intraRefreshDuration;
    //        }
    //        else
    //        {
    //            encPicParams.codecPicParams.h264PicParams.forceIntraRefreshWithFrameCnt = encPicCommand->intraRefreshDuration;
    //        }
    //    }
    //}

    NVENCSTATUS nvStatus = m_pEncodeAPI->nvEncEncodePicture(m_hEncoder, &encPicParams);
    if (nvStatus != NV_ENC_SUCCESS && nvStatus != NV_ENC_ERR_NEED_MORE_INPUT) {
        PrintMes(NV_LOG_ERROR, FOR_AUO ? _T("フレームの投入に失敗しました。\n") : _T("Failed to add frame into the encoder.\n"));
        return nvStatus;
    }

    return NV_ENC_SUCCESS;
}

NVENCSTATUS NVEncCore::EncodeFrame(uint64_t timestamp) {
    EncodeBuffer *pEncodeBuffer = m_EncodeBufferQueue.GetAvailable();
    if (!pEncodeBuffer) {
        ProcessOutput(m_EncodeBufferQueue.GetPending());
        pEncodeBuffer = m_EncodeBufferQueue.GetAvailable();
    }

    NvEncEncodeFrame(pEncodeBuffer, timestamp);

    return NV_ENC_SUCCESS;
}

#pragma warning(push)
#pragma warning(disable: 4100)
NVENCSTATUS NVEncCore::EncodeFrame(EncodeFrameConfig *pEncodeFrame, uint64_t timestamp) {
    NVENCSTATUS nvStatus = NV_ENC_SUCCESS;
#if ENABLE_AVCUVID_READER
    EncodeBuffer *pEncodeBuffer = m_EncodeBufferQueue.GetAvailable();
    if (!pEncodeBuffer) {
        pEncodeBuffer = m_EncodeBufferQueue.GetPending();
        ProcessOutput(pEncodeBuffer);
        PrintMes(NV_LOG_TRACE, _T("Output frame %d\n"), m_pStatus->m_sData.frameOut);
        if (pEncodeBuffer->stInputBfr.hInputSurface) {
            nvStatus = NvEncUnmapInputResource(pEncodeBuffer->stInputBfr.hInputSurface);
            pEncodeBuffer->stInputBfr.hInputSurface = NULL;
        }
        pEncodeBuffer = m_EncodeBufferQueue.GetAvailable();
    }

    // encode width and height
    unsigned int dwWidth  = pEncodeBuffer->stInputBfr.dwWidth;
    unsigned int dwHeight = pEncodeBuffer->stInputBfr.dwHeight;

    cuvidCtxLock(m_ctxLock, 0);
    assert(pEncodeFrame->width == dwWidth && pEncodeFrame->height == dwHeight);

    CUDA_MEMCPY2D memcpy2D  = {0};
    memcpy2D.srcMemoryType  = CU_MEMORYTYPE_DEVICE;
    memcpy2D.srcDevice      = pEncodeFrame->dptr;
    memcpy2D.srcPitch       = pEncodeFrame->pitch;
    memcpy2D.dstMemoryType  = CU_MEMORYTYPE_DEVICE;
    memcpy2D.dstDevice      = (CUdeviceptr)pEncodeBuffer->stInputBfr.pNV12devPtr;
    memcpy2D.dstPitch       = pEncodeBuffer->stInputBfr.uNV12Stride;
    memcpy2D.WidthInBytes   = dwWidth;
    memcpy2D.Height         = dwHeight*3/2;
    cuMemcpy2D(&memcpy2D);

    cuvidCtxUnlock(m_ctxLock, 0);

    nvStatus = NvEncMapInputResource(pEncodeBuffer->stInputBfr.nvRegisteredResource, &pEncodeBuffer->stInputBfr.hInputSurface);
    if (nvStatus != NV_ENC_SUCCESS) {
        PrintMes(NV_LOG_ERROR, _T("Failed to Map input buffer %p\n"), pEncodeBuffer->stInputBfr.hInputSurface);
        return nvStatus;
    }
    NvEncEncodeFrame(pEncodeBuffer, timestamp);
#endif //#if ENABLE_AVCUVID_READER
    return nvStatus;
}
#pragma warning(pop)

NVENCSTATUS NVEncCore::Encode() {
    NVENCSTATUS nvStatus = NV_ENC_SUCCESS;

    m_pStatus->SetStart();

    int ret = 0;
    const int bufferCount = m_uEncodeBufferCount;
#if ENABLE_AVCUVID_READER
    const AVCodecContext *pVideoCtx = nullptr;
    CAvcodecReader *pReader = dynamic_cast<CAvcodecReader *>(m_pFileReader.get());
    if (pReader != nullptr) {
        pVideoCtx = pReader->GetInputVideoCodecCtx();
    }

    //streamのindexから必要なwriteへのポインタを返すテーブルを作成
    std::map<int, shared_ptr<CAvcodecWriter>> pWriterForAudioStreams;
    for (auto pWriter : m_pFileWriterListAudio) {
        auto pAVCodecWriter = std::dynamic_pointer_cast<CAvcodecWriter>(pWriter);
        if (pAVCodecWriter) {
            auto trackIdList = pAVCodecWriter->GetStreamTrackIdList();
            for (auto trackID : trackIdList) {
                pWriterForAudioStreams[trackID] = pAVCodecWriter;
            }
        }
    }

    auto extract_audio =[&]() {
        int sts = 0;
        if (m_pFileWriterListAudio.size()) {
            auto pAVCodecReader = std::dynamic_pointer_cast<CAvcodecReader>(m_pFileReader);
            vector<AVPacket> packetList;
            if (pAVCodecReader != nullptr) {
                packetList = pAVCodecReader->GetStreamDataPackets();
            }
            //音声ファイルリーダーからのトラックを結合する
            for (const auto& reader : m_AudioReaders) {
                auto pReader = std::dynamic_pointer_cast<CAvcodecReader>(reader);
                if (pReader != nullptr) {
                    vector_cat(packetList, pReader->GetStreamDataPackets());
                }
            }
            //パケットを各Writerに分配する
            for (uint32_t i = 0; i < packetList.size(); i++) {
                const int nTrackId = (int16_t)(packetList[i].flags >> 16);
                if (pWriterForAudioStreams.count(nTrackId)) {
                    auto pWriter = pWriterForAudioStreams[nTrackId];
                    if (pWriter == nullptr) {
                        PrintMes(NV_LOG_ERROR, _T("Invalid writer found for audio track %d\n"), nTrackId);
                        return 1;
                    }
                    if (0 != (sts = pWriter->WriteNextPacket(&packetList[i]))) {
                        return 1;
                    }
                } else {
                    PrintMes(NV_LOG_ERROR, _T("Failed to find writer for audio track %d\n"), nTrackId);
                    return 1;
                }
            }
        }
        return 0;
    };

    if (m_cuvidDec) {
        auto th_input = std::thread([this, pVideoCtx, &nvStatus](){
            CUresult curesult = CUDA_SUCCESS;
            vector<uint8_t> bitstream;
            int sts = NVENC_THREAD_RUNNING;
            for (int i = 0; sts == NVENC_THREAD_RUNNING && nvStatus == NV_ENC_SUCCESS && !m_cuvidDec->GetError(); i++) {
                sts = m_pFileReader->LoadNextFrame(nullptr, 0);
                int64_t pts;
                m_pFileReader->GetNextBitstream(bitstream, &pts);
                PrintMes(NV_LOG_TRACE, _T("Set packet %d\n"), i);
                if (CUDA_SUCCESS != (curesult = m_cuvidDec->DecodePacket(bitstream.data(), bitstream.size(), pts, pVideoCtx->pkt_timebase))) {
                    PrintMes(NV_LOG_ERROR, _T("Error in DecodePacket: %d (%s).\n"), curesult, char_to_tstring(_cudaGetErrorEnum(curesult)).c_str());
                    return curesult;
                }
            }
            if (CUDA_SUCCESS != (curesult = m_cuvidDec->DecodePacket(nullptr, 0, AV_NOPTS_VALUE, pVideoCtx->pkt_timebase))) {
                PrintMes(NV_LOG_ERROR, _T("Error in DecodePacketFin: %d (%s).\n"), curesult, char_to_tstring(_cudaGetErrorEnum(curesult)).c_str());
            }
            return curesult;
        });
        PrintMes(NV_LOG_DEBUG, _T("Started Encode thread\n"));

        CProcSpeedControl speedCtrl(m_nProcSpeedLimit);
        int64_t nEstimatedPts = AV_NOPTS_VALUE;
        const int nFrameDuration = (int)av_rescale_q(1, av_make_q(m_inputFps.second, m_inputFps.first), pVideoCtx->pkt_timebase);
        int decodedFrame = 0;
        int encodedFrame = 0;
        while (!m_cuvidDec->GetError()
            && !(m_cuvidDec->frameQueue()->isEndOfDecode() && m_cuvidDec->frameQueue()->isEmpty())) {
            if (m_pAbortByUser && *m_pAbortByUser) {
                nvStatus = NV_ENC_ERR_ABORT;
                break;
            }

            CUVIDPARSERDISPINFO pInfo;
            if (m_cuvidDec->frameQueue()->dequeue(&pInfo)) {
                int64_t pts = av_rescale_q(pInfo.timestamp, CUVID_NATIVE_TIMEBASE, pVideoCtx->pkt_timebase);
                if (m_pTrimParam && !frame_inside_range(decodedFrame++, m_pTrimParam->list)) {
                    m_cuvidDec->frameQueue()->releaseFrame(&pInfo);
                    continue;
                }

                auto encode = [&](CUVIDPROCPARAMS oVPP) {
                    speedCtrl.wait();
                    CUresult curesult = CUDA_SUCCESS;
                    PrintMes(NV_LOG_TRACE, _T("Get decoded frame %d\n"), decodedFrame);
                    CUdeviceptr dMappedFrame = 0;
                    unsigned int pitch;
                    if (CUDA_SUCCESS != (curesult = cuvidMapVideoFrame(m_cuvidDec->GetDecoder(), pInfo.picture_index, &dMappedFrame, &pitch, &oVPP))) {
                        PrintMes(NV_LOG_ERROR, _T("Error cuvidMapVideoFrame: %d (%s).\n"), curesult, char_to_tstring(_cudaGetErrorEnum(curesult)).c_str());
                        return NV_ENC_ERR_GENERIC;
                    }

                    EncodeFrameConfig stEncodeConfig = { 0 };
                    stEncodeConfig.dptr = dMappedFrame;
                    stEncodeConfig.pitch = pitch;
                    stEncodeConfig.width = m_uEncWidth;
                    stEncodeConfig.height = m_uEncHeight;
                    PrintMes(NV_LOG_TRACE, _T("Set frame to encode %d\n"), decodedFrame);
                    pInfo.timestamp = encodedFrame;
                    auto encstatus = EncodeFrame(&stEncodeConfig, pInfo.timestamp);
                    encodedFrame++;
                    if (NV_ENC_SUCCESS != encstatus) {
                        PrintMes(NV_LOG_ERROR, _T("Error EncodeFrame: %d.\n"), encstatus);
                        return encstatus;
                    }

                    if (CUDA_SUCCESS != (curesult = cuvidUnmapVideoFrame(m_cuvidDec->GetDecoder(), dMappedFrame))) {
                        PrintMes(NV_LOG_ERROR, _T("Error cuvidMapVideoFrame: %d (%s).\n"), curesult, char_to_tstring(_cudaGetErrorEnum(curesult)).c_str());
                        return NV_ENC_ERR_GENERIC;
                    }
                    return NV_ENC_SUCCESS;
                };
                CUVIDPROCPARAMS oVPP = { 0 };
                oVPP.top_field_first = m_stPicStruct != NV_ENC_PIC_STRUCT_FIELD_BOTTOM_TOP;

                auto encode_frame = [&]() {
                    NVENCSTATUS status = NV_ENC_SUCCESS;
                    auto deint = m_cuvidDec->getDeinterlaceMode();
                    switch (deint) {
                    case cudaVideoDeinterlaceMode_Weave:
                        oVPP.unpaired_field = 1;
                        oVPP.progressive_frame = (m_stPicStruct == NV_ENC_PIC_STRUCT_FRAME);
                        status = encode(oVPP);
                        break;
                    case cudaVideoDeinterlaceMode_Bob:
                        oVPP.progressive_frame = 0;
                        oVPP.second_field = 0;
                        status = encode(oVPP);
                        if (NV_ENC_SUCCESS != status) return status;
                        oVPP.second_field = 1;
                        status = encode(oVPP);
                        break;
                    case cudaVideoDeinterlaceMode_Adaptive:
                        oVPP.progressive_frame = 0;
                        status = encode(oVPP);
                        break;
                    default:
                        PrintMes(NV_LOG_ERROR, _T("Unknown Deinterlace mode\n"));
                        status = NV_ENC_ERR_GENERIC;
                        break;
                    }
                    return status;
                };

                if ((m_nAVSyncMode & NV_AVSYNC_FORCE_CFR) == NV_AVSYNC_FORCE_CFR) {
                    if (nEstimatedPts == AV_NOPTS_VALUE) {
                        nEstimatedPts = pts;
                    }
                    auto ptsDiff = pts - nEstimatedPts;
                    nEstimatedPts += nFrameDuration;
                    if (ptsDiff >= std::max(1, nFrameDuration * 3 / 4)) {
                        //水増しが必要
                        if (NV_ENC_SUCCESS != (nvStatus = encode_frame())) break;
                        nEstimatedPts += nFrameDuration;
                        if (NV_ENC_SUCCESS != (nvStatus = encode_frame())) break;
                    } else {
                        if (ptsDiff <= std::min(-1, -1 * nFrameDuration * 3 / 4)) {
                            //間引きが必要
                            continue;
                        } else {
                            if (NV_ENC_SUCCESS != (nvStatus = encode_frame())) break;
                        }
                    }
                } else {
                    if (NV_ENC_SUCCESS != (nvStatus = encode_frame())) break;
                }

                if (0 != extract_audio()) {
                    nvStatus = NV_ENC_ERR_GENERIC;
                    break;
                }
                m_cuvidDec->frameQueue()->releaseFrame(&pInfo);
            } else {
                m_cuvidDec->frameQueue()->waitForQueueUpdate();
            }
        }
        if (th_input.joinable()) {
            //ここでフレームをすべて吐き出し切らないと、中断時にデコードスレッドが終了しない
            while (!m_cuvidDec->GetError()
                && !(m_cuvidDec->frameQueue()->isEndOfDecode() && m_cuvidDec->frameQueue()->isEmpty())) {
                CUVIDPARSERDISPINFO pInfo;
                if (m_cuvidDec->frameQueue()->dequeue(&pInfo)) {
                    m_cuvidDec->frameQueue()->releaseFrame(&pInfo);
                }
            }
            th_input.join();
        }
        PrintMes(NV_LOG_DEBUG, _T("Joined Encode thread\n"));
        if (m_cuvidDec->GetError()) {
            nvStatus = NV_ENC_ERR_GENERIC;
        }
    } else
#endif //#if ENABLE_AVCUVID_READER
    {
        CProcSpeedControl speedCtrl(m_nProcSpeedLimit);
        for (int iFrame = 0; nvStatus == NV_ENC_SUCCESS; iFrame++) {
            if (m_pAbortByUser && *m_pAbortByUser) {
                nvStatus = NV_ENC_ERR_ABORT;
                break;
            }
            speedCtrl.wait();
#if ENABLE_AVCUVID_READER
            if (0 != extract_audio()) {
                nvStatus = NV_ENC_ERR_GENERIC;
                break;
            }
#endif //#if ENABLE_AVCUVID_READER
            uint32_t lockedPitch = 0;
            unsigned char *pInputSurface = nullptr;
            const int index = iFrame % bufferCount;
            NvEncLockInputBuffer(m_stEncodeBuffer[index].stInputBfr.hInputSurface, (void**)&pInputSurface, &lockedPitch);
            ret = m_pFileReader->LoadNextFrame(pInputSurface, lockedPitch);
            NvEncUnlockInputBuffer(m_stEncodeBuffer[index].stInputBfr.hInputSurface);
            if (ret)
                break;
            nvStatus = EncodeFrame(iFrame);
        }
    }
#if ENABLE_AVCUVID_READER
    for (const auto& writer : m_pFileWriterListAudio) {
        auto pAVCodecWriter = std::dynamic_pointer_cast<CAvcodecWriter>(writer);
        if (pAVCodecWriter != nullptr) {
            //エンコーダなどにキャッシュされたパケットを書き出す
            pAVCodecWriter->WriteNextPacket(nullptr);
        }
    }
#endif //#if ENABLE_AVCUVID_READER
    PrintMes(NV_LOG_INFO, _T("                                                                         \n"));
    //FlushEncoderはかならず行わないと、NvEncDestroyEncoderで異常終了する
    auto encstatus = FlushEncoder();
    if (nvStatus == NV_ENC_SUCCESS && encstatus != NV_ENC_SUCCESS) {
        PrintMes(NV_LOG_ERROR, _T("Error FlushEncoder: %d.\n"), encstatus);
        nvStatus = encstatus;
    } else {
        PrintMes(NV_LOG_DEBUG, _T("Flushed Encoder\n"));
    }
    m_pFileReader->Close();
    m_pFileWriter->Close();
    m_pStatus->writeResult();
    return nvStatus;
}

NV_ENC_CODEC_CONFIG NVEncCore::DefaultParamH264() {
    NV_ENC_CODEC_CONFIG config = { 0 };

    config.h264Config.level     = NV_ENC_LEVEL_AUTOSELECT;
    config.h264Config.idrPeriod = DEFAULT_GOP_LENGTH;

    config.h264Config.chromaFormatIDC            = 1;
    config.h264Config.disableDeblockingFilterIDC = 0;
    config.h264Config.disableSPSPPS              = 0;
    config.h264Config.sliceMode                  = 3;
    config.h264Config.sliceModeData              = DEFAULT_NUM_SLICES;
    config.h264Config.maxNumRefFrames            = DEFAULT_REF_FRAMES;
    config.h264Config.bdirectMode                = (DEFAULT_B_FRAMES > 0) ? NV_ENC_H264_BDIRECT_MODE_SPATIAL : NV_ENC_H264_BDIRECT_MODE_DISABLE;
    config.h264Config.adaptiveTransformMode      = NV_ENC_H264_ADAPTIVE_TRANSFORM_ENABLE;
    config.h264Config.entropyCodingMode          = NV_ENC_H264_ENTROPY_CODING_MODE_CABAC;

    config.h264Config.h264VUIParameters.overscanInfo = 0;
    config.h264Config.h264VUIParameters.colourMatrix            = get_cx_value(list_colormatrix, _T("undef"));
    config.h264Config.h264VUIParameters.colourPrimaries         = get_cx_value(list_colorprim,   _T("undef"));
    config.h264Config.h264VUIParameters.transferCharacteristics = get_cx_value(list_transfer,    _T("undef"));
    config.h264Config.h264VUIParameters.videoFormat             = get_cx_value(list_videoformat, _T("undef"));

    return config;
}

NV_ENC_CODEC_CONFIG NVEncCore::DefaultParamHEVC() {
    NV_ENC_CODEC_CONFIG config = { 0 };

    config.hevcConfig.level = NV_ENC_LEVEL_AUTOSELECT;
    config.hevcConfig.tier  = NV_ENC_TIER_HEVC_MAIN;
    config.hevcConfig.minCUSize = NV_ENC_HEVC_CUSIZE_8x8;
    config.hevcConfig.maxCUSize = NV_ENC_HEVC_CUSIZE_32x32;
    config.hevcConfig.sliceMode = 0;
    config.hevcConfig.sliceModeData = 0;
    config.hevcConfig.maxNumRefFramesInDPB = DEFAULT_REF_FRAMES;
    config.hevcConfig.chromaFormatIDC = 1;

    config.hevcConfig.hevcVUIParameters.overscanInfo = 0;
    config.hevcConfig.hevcVUIParameters.colourMatrix            = get_cx_value(list_colormatrix, _T("undef"));
    config.hevcConfig.hevcVUIParameters.colourPrimaries         = get_cx_value(list_colorprim,   _T("undef"));
    config.hevcConfig.hevcVUIParameters.transferCharacteristics = get_cx_value(list_transfer,    _T("undef"));
    config.hevcConfig.hevcVUIParameters.videoFormat             = get_cx_value(list_videoformat, _T("undef"));

    return config;
}

NV_ENC_CONFIG NVEncCore::DefaultParam() {

    NV_ENC_CONFIG config = { 0 };
    SET_VER(config, NV_ENC_CONFIG);
    config.frameFieldMode                 = NV_ENC_PARAMS_FRAME_FIELD_MODE_FRAME;
    config.profileGUID                    = NV_ENC_H264_PROFILE_HIGH_GUID;
    config.gopLength                      = DEFAULT_GOP_LENGTH;
    config.rcParams.rateControlMode       = NV_ENC_PARAMS_RC_CONSTQP;
    //config.encodeCodecConfig.h264Config.level;
    config.frameIntervalP                 = DEFAULT_B_FRAMES + 1;
    config.mvPrecision                    = NV_ENC_MV_PRECISION_QUARTER_PEL;
    config.monoChromeEncoding             = 0;
    config.rcParams.version               = NV_ENC_RC_PARAMS_VER;
    config.rcParams.averageBitRate        = DEFAULT_AVG_BITRATE;
    config.rcParams.maxBitRate            = DEFAULT_MAX_BITRATE;
    config.rcParams.enableInitialRCQP     = 1;
    config.rcParams.initialRCQP.qpInterB  = DEFAULT_QP_B;
    config.rcParams.initialRCQP.qpInterP  = DEFAULT_QP_P;
    config.rcParams.initialRCQP.qpIntra   = DEFAUTL_QP_I;
    config.rcParams.constQP.qpInterB      = DEFAULT_QP_B;
    config.rcParams.constQP.qpInterP      = DEFAULT_QP_P;
    config.rcParams.constQP.qpIntra       = DEFAUTL_QP_I;
    config.rcParams.lookaheadDepth        = DEFAULT_LOOKAHEAD;

    config.rcParams.vbvBufferSize         = 0;
    config.rcParams.vbvInitialDelay       = 0;
    config.encodeCodecConfig              = DefaultParamH264();

    return config;
}

tstring NVEncCore::GetEncodingParamsInfo(int output_level) {
    tstring str;
    auto add_str =[output_level, &str](int info_level, const TCHAR *fmt, ...) {
        if (info_level >= output_level) {
            va_list args;
            va_start(args, fmt);
            const size_t append_len = _vsctprintf(fmt, args) + 1;
            size_t current_len = _tcslen(str.c_str());
            str.resize(current_len + append_len, 0);
            _vstprintf_s(&str[current_len], append_len, fmt, args);
        }
    };

    auto value_or_auto =[](int value, int value_auto, const TCHAR *unit) {
        tstring str;
        if (value == value_auto) {
            str = _T("auto");
        } else {
            TCHAR buf[256];
            _stprintf_s(buf, _countof(buf), _T("%d %s"), value, unit);
            str = buf;
        }
        return str;
    };

    auto on_off =[](int value) {
        return (value) ? _T("on") : _T("off");
    };

    TCHAR cpu_info[1024] = { 0 };
    getCPUInfo(cpu_info, _countof(cpu_info));

    TCHAR gpu_info[1024] = { 0 };
    {
        NVEncoderGPUInfo nvencGPUInfo;
        const auto len = _stprintf_s(gpu_info, _T("#%d: "), m_nDeviceId);
        if (m_nDeviceId || 0 != getGPUInfo("NVIDIA", gpu_info + len, _countof(gpu_info) - len)) {
            for (const auto& gpuInfo : nvencGPUInfo.getGPUList()) {
                if (m_nDeviceId == gpuInfo.first) {
                    _stprintf_s(gpu_info, _T("#%d: %s"), gpuInfo.first, gpuInfo.second.c_str());
                }
            }
        }
    }

    int codec = get_value_from_guid(m_stCodecGUID, list_nvenc_codecs);
    auto sar = get_sar(m_uEncWidth, m_uEncHeight, m_stCreateEncodeParams.darWidth, m_stCreateEncodeParams.darHeight);
    add_str(NV_LOG_ERROR, _T("NVEnc %s (%s), using NVENC API v%d.%d\n"), VER_STR_FILEVERSION_TCHAR, BUILD_ARCH_STR, NVENCAPI_MAJOR_VERSION, NVENCAPI_MINOR_VERSION);
    add_str(NV_LOG_INFO,  _T("OS Version     %s (%s)\n"), getOSVersion().c_str(), nv_is_64bit_os() ? _T("x64") : _T("x86"));
    add_str(NV_LOG_INFO,  _T("CPU            %s\n"), cpu_info);
    add_str(NV_LOG_INFO,  _T("GPU            %s\n"), gpu_info);
    add_str(NV_LOG_ERROR, _T("Input Buffers  %s, %d frames\n"), _T("CUDA"), m_uEncodeBufferCount);
    tstring inputMes = m_pFileReader->GetInputMessage();
    for (const auto& reader : m_AudioReaders) {
        inputMes += _T("\n") + tstring(reader->GetInputMessage());
    }
    auto inputMesSplitted = split(inputMes, _T("\n"));
    for (uint32_t i = 0; i < (uint32_t)inputMesSplitted.size(); i++) {
        add_str(NV_LOG_ERROR, _T("%s%s\n"), (i == 0) ? _T("Input Info     ") : _T("               "), inputMesSplitted[i].c_str());
    }
#if ENABLE_AVCUVID_READER
    if (m_cuvidDec && m_cuvidDec->getDeinterlaceMode() != cudaVideoDeinterlaceMode_Weave) {
        add_str(NV_LOG_ERROR, _T("Deinterlace    %s\n"), get_chr_from_value(list_deinterlace, m_cuvidDec->getDeinterlaceMode()));
    }
#endif //#if ENABLE_AVCUVID_READER
    if (m_pTrimParam != NULL && m_pTrimParam->list.size()
        && !(m_pTrimParam->list[0].start == 0 && m_pTrimParam->list[0].fin == TRIM_MAX)) {
        add_str(NV_LOG_ERROR, _T("%s"), _T("Trim           "));
        for (auto trim : m_pTrimParam->list) {
            if (trim.fin == TRIM_MAX) {
                add_str(NV_LOG_ERROR, _T("%d-fin "), trim.start + m_pTrimParam->offset);
            } else {
                add_str(NV_LOG_ERROR, _T("%d-%d "), trim.start + m_pTrimParam->offset, trim.fin + m_pTrimParam->offset);
            }
        }
        add_str(NV_LOG_ERROR, _T("[offset: %d]\n"), m_pTrimParam->offset);
    }
    if (m_nAVSyncMode != NV_AVSYNC_THROUGH) {
        add_str(NV_LOG_ERROR, _T("AVSync         %s\n"), get_chr_from_value(list_avsync, m_nAVSyncMode));
    }
    add_str(NV_LOG_ERROR, _T("Output Info    %s %s\n"), get_name_from_guid(m_stCodecGUID, list_nvenc_codecs),
        (codec == NV_ENC_H264) ? get_name_from_guid(m_stEncConfig.profileGUID, h264_profile_names) : get_name_from_guid(m_stEncConfig.profileGUID, h265_profile_names));
    add_str(NV_LOG_ERROR, _T("               %dx%d%s %d:%d %.3ffps (%d/%dfps)\n"), m_uEncWidth, m_uEncHeight, (m_stEncConfig.frameFieldMode != NV_ENC_PARAMS_FRAME_FIELD_MODE_FRAME) ? _T("i") : _T("p"), sar.first, sar.second, m_stCreateEncodeParams.frameRateNum / (double)m_stCreateEncodeParams.frameRateDen, m_stCreateEncodeParams.frameRateNum, m_stCreateEncodeParams.frameRateDen);
    if (m_pFileWriter) {
        inputMesSplitted = split(m_pFileWriter->GetOutputMessage(), _T("\n"));
        for (auto mes : inputMesSplitted) {
            if (mes.length()) {
                add_str(NV_LOG_ERROR,_T("%s%s\n"), _T("               "), mes.c_str());
            }
        }
    }
    for (auto pWriter : m_pFileWriterListAudio) {
        if (pWriter && pWriter != m_pFileWriter) {
            inputMesSplitted = split(pWriter->GetOutputMessage(), _T("\n"));
            for (auto mes : inputMesSplitted) {
                if (mes.length()) {
                    add_str(NV_LOG_ERROR,_T("%s%s\n"), _T("               "), mes.c_str());
                }
            }
        }
    }
    add_str(NV_LOG_DEBUG, _T("Encoder Preset %s\n"), get_name_from_guid(m_stCreateEncodeParams.presetGUID, preset_names));
    add_str(NV_LOG_ERROR, _T("Rate Control   %s"), get_chr_from_value(list_nvenc_rc_method_en, m_stEncConfig.rcParams.rateControlMode));
    if (NV_ENC_PARAMS_RC_CONSTQP == m_stEncConfig.rcParams.rateControlMode) {
        add_str(NV_LOG_ERROR, _T("  I:%d  P:%d  B:%d%s\n"), m_stEncConfig.rcParams.constQP.qpIntra, m_stEncConfig.rcParams.constQP.qpInterP, m_stEncConfig.rcParams.constQP.qpInterB,
            m_stCreateEncodeParams.encodeConfig->encodeCodecConfig.h264Config.qpPrimeYZeroTransformBypassFlag ? _T(" (lossless)") : _T(""));
    } else {
        add_str(NV_LOG_ERROR, _T("\n"));
        add_str(NV_LOG_ERROR, _T("Bitrate        %d kbps (Max: %d kbps)\n"), m_stEncConfig.rcParams.averageBitRate / 1000, m_stEncConfig.rcParams.maxBitRate / 1000);
        if (m_stEncConfig.rcParams.enableInitialRCQP) {
            add_str(NV_LOG_INFO,  _T("Initial QP     I:%d  P:%d  B:%d\n"), m_stEncConfig.rcParams.initialRCQP.qpIntra, m_stEncConfig.rcParams.initialRCQP.qpInterP, m_stEncConfig.rcParams.initialRCQP.qpInterB);
        }
        if (m_stEncConfig.rcParams.enableMaxQP || m_stEncConfig.rcParams.enableMinQP) {
            int minQPI = (m_stEncConfig.rcParams.enableMinQP) ? m_stEncConfig.rcParams.minQP.qpIntra  :  0;
            int maxQPI = (m_stEncConfig.rcParams.enableMaxQP) ? m_stEncConfig.rcParams.maxQP.qpIntra  : 51;
            int minQPP = (m_stEncConfig.rcParams.enableMinQP) ? m_stEncConfig.rcParams.minQP.qpInterP :  0;
            int maxQPP = (m_stEncConfig.rcParams.enableMaxQP) ? m_stEncConfig.rcParams.maxQP.qpInterP : 51;
            int minQPB = (m_stEncConfig.rcParams.enableMinQP) ? m_stEncConfig.rcParams.minQP.qpInterB :  0;
            int maxQPB = (m_stEncConfig.rcParams.enableMaxQP) ? m_stEncConfig.rcParams.maxQP.qpInterB : 51;
            add_str(NV_LOG_INFO,  _T("QP range       I:%d-%d  P:%d-%d  B:%d-%d\n"), minQPI, maxQPI, minQPP, maxQPP, minQPB, maxQPB);
        }
        add_str(NV_LOG_INFO,  _T("VBV buf size   %s\n"), value_or_auto(m_stEncConfig.rcParams.vbvBufferSize / 1000,   0, _T("kbit")).c_str());
        add_str(NV_LOG_DEBUG, _T("VBV init delay %s\n"), value_or_auto(m_stEncConfig.rcParams.vbvInitialDelay / 1000, 0, _T("kbit")).c_str());
    }
    tstring strLookahead = _T("Lookahead      ");
    if (m_stEncConfig.rcParams.enableLookahead) {
        strLookahead += strsprintf(_T("on, %d frames"), m_stEncConfig.rcParams.lookaheadDepth);
        if (!m_stEncConfig.rcParams.disableBadapt || !m_stEncConfig.rcParams.disableIadapt) {
            strLookahead += _T(", Adaptive ");
            if (!m_stEncConfig.rcParams.disableIadapt) strLookahead += _T("I");
            if (!m_stEncConfig.rcParams.disableBadapt && !m_stEncConfig.rcParams.disableIadapt) strLookahead += _T(", ");
            if (!m_stEncConfig.rcParams.disableIadapt) strLookahead += _T("B");
            strLookahead += _T(" Insert");
        }
    } else {
        strLookahead += _T("off");
    }
    add_str(NV_LOG_INFO,  _T("%s\n"), strLookahead.c_str());
    add_str(NV_LOG_INFO,  _T("GOP length     %d frames\n"), m_stEncConfig.gopLength);
    add_str(NV_LOG_INFO,  _T("B frames       %d frames\n"), m_stEncConfig.frameIntervalP - 1);
    if (codec == NV_ENC_H264) {
        add_str(NV_LOG_DEBUG, _T("Output         "));
        TCHAR bitstream_info[256] ={ 0 };
        if (m_stEncConfig.encodeCodecConfig.h264Config.outputBufferingPeriodSEI) _tcscat_s(bitstream_info, _countof(bitstream_info), _T("BufferingPeriodSEI,"));
        if (m_stEncConfig.encodeCodecConfig.h264Config.outputPictureTimingSEI)   _tcscat_s(bitstream_info, _countof(bitstream_info), _T("PicTimingSEI,"));
        if (m_stEncConfig.encodeCodecConfig.h264Config.outputAUD)                _tcscat_s(bitstream_info, _countof(bitstream_info), _T("AUD,"));
        if (m_stEncConfig.encodeCodecConfig.h264Config.outputFramePackingSEI)    _tcscat_s(bitstream_info, _countof(bitstream_info), _T("FramePackingSEI,"));
        if (m_stEncConfig.encodeCodecConfig.h264Config.outputRecoveryPointSEI)   _tcscat_s(bitstream_info, _countof(bitstream_info), _T("RecoveryPointSEI,"));
        if (m_stEncConfig.encodeCodecConfig.h264Config.repeatSPSPPS)             _tcscat_s(bitstream_info, _countof(bitstream_info), _T("repeatSPSPPS,"));
        if (_tcslen(bitstream_info)) {
            bitstream_info[_tcslen(bitstream_info)-1] = _T('\0');
        } else {
            _tcscpy_s(bitstream_info, _countof(bitstream_info), _T("-"));
        }
        add_str(NV_LOG_DEBUG, _T("%s\n"), bitstream_info);
    }

    const bool bEnableLTR = (codec == NV_ENC_H264) ? m_stEncConfig.encodeCodecConfig.h264Config.enableLTR : m_stEncConfig.encodeCodecConfig.hevcConfig.enableLTR;
    tstring strRef = strsprintf(_T("%d frames, LTR: %s"),
        (codec == NV_ENC_H264) ? m_stEncConfig.encodeCodecConfig.h264Config.maxNumRefFrames : m_stEncConfig.encodeCodecConfig.hevcConfig.maxNumRefFramesInDPB,
        (bEnableLTR) ? _T("on") : _T("off"));
    add_str(NV_LOG_INFO,  _T("Ref frames     %s\n"), strRef.c_str());
    
    tstring strAQ;
    if (m_stEncConfig.rcParams.enableAQ || m_stEncConfig.rcParams.enableTemporalAQ) {
        strAQ = _T("on");
        if (codec == NV_ENC_H264) {
            strAQ += _T("(");
            if (m_stEncConfig.rcParams.enableAQ)         strAQ += _T("spatial");
            if (m_stEncConfig.rcParams.enableAQ && m_stEncConfig.rcParams.enableTemporalAQ) strAQ += _T(", ");
            if (m_stEncConfig.rcParams.enableTemporalAQ) strAQ += _T("temporal");
            strAQ += _T(", strength ");
            strAQ += (m_stEncConfig.rcParams.aqStrength == 0) ? _T("auto") : strsprintf(_T("%d"), m_stEncConfig.rcParams.aqStrength);
            strAQ += _T(")");
        }
    } else {
        strAQ = _T("off");
    }
    add_str(NV_LOG_INFO,  _T("AQ             %s\n"), strAQ.c_str());
    add_str(NV_LOG_INFO,  _T("MV Quality     %s\n"), get_chr_from_value(list_mv_presicion, m_stEncConfig.mvPrecision));
    if (codec == NV_ENC_H264 && 3 == m_stEncConfig.encodeCodecConfig.h264Config.sliceMode) {
        add_str(NV_LOG_DEBUG, _T("Slice number      %d\n"), m_stEncConfig.encodeCodecConfig.h264Config.sliceModeData);
    } else {
        add_str(NV_LOG_DEBUG, _T("Slice          Mode:%d, ModeData:%d\n"), m_stEncConfig.encodeCodecConfig.h264Config.sliceMode, m_stEncConfig.encodeCodecConfig.h264Config.sliceModeData);
    }
    if (codec == NV_ENC_H264) {
        add_str(NV_LOG_INFO,  _T("CABAC/deblock  %s / %s\n"), get_chr_from_value(list_entropy_coding, m_stEncConfig.encodeCodecConfig.h264Config.entropyCodingMode), on_off(!m_stEncConfig.encodeCodecConfig.h264Config.disableDeblockingFilterIDC));
        add_str(NV_LOG_DEBUG, _T("hierarchyFrame P:%s  B:%s\n"), on_off(m_stEncConfig.encodeCodecConfig.h264Config.hierarchicalPFrames), on_off(m_stEncConfig.encodeCodecConfig.h264Config.hierarchicalBFrames));
        add_str(NV_LOG_DEBUG, _T("VFR            %s\n"), on_off(m_stEncConfig.encodeCodecConfig.h264Config.enableVFR));
        add_str(NV_LOG_DEBUG, _T("LTR            %s"),   on_off(m_stEncConfig.encodeCodecConfig.h264Config.enableLTR));
        if (m_stEncConfig.encodeCodecConfig.h264Config.enableLTR) {
            add_str(NV_LOG_DEBUG, _T(", Mode:%d, NumFrames:%d"), m_stEncConfig.encodeCodecConfig.h264Config.ltrTrustMode, m_stEncConfig.encodeCodecConfig.h264Config.ltrNumFrames);
        }
        add_str(NV_LOG_DEBUG, _T("\n"));
        add_str(NV_LOG_DEBUG, _T("Adpt.Transform %s\n"), get_chr_from_value(list_adapt_transform, m_stEncConfig.encodeCodecConfig.h264Config.adaptiveTransformMode));
        add_str(NV_LOG_DEBUG, _T("FMO            %s\n"), get_chr_from_value(list_fmo, m_stEncConfig.encodeCodecConfig.h264Config.fmoMode));
        add_str(NV_LOG_DEBUG, _T("MV Mode        %s\n"), get_chr_from_value(list_bdirect, m_stEncConfig.encodeCodecConfig.h264Config.bdirectMode));
    }
    if (codec == NV_ENC_HEVC) {
        add_str(NV_LOG_INFO, _T("CU max / min   %s / %s\n"),
            get_chr_from_value(list_hevc_cu_size, m_stEncConfig.encodeCodecConfig.hevcConfig.maxCUSize),
            get_chr_from_value(list_hevc_cu_size, m_stEncConfig.encodeCodecConfig.hevcConfig.minCUSize));
    }
    return str;
}

void NVEncCore::PrintEncodingParamsInfo(int output_level) {
    PrintMes(NV_LOG_INFO, _T("%s"), GetEncodingParamsInfo(output_level).c_str());
}
