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

#include <vector>
#include <array>
#include <map>
#include <numeric>
#include <deque>
#include <string>
#include <algorithm>
#include <thread>
#include "rgy_osdep.h"
#pragma warning(push)
#pragma warning(disable: 4819)
//ファイルは、現在のコード ページ (932) で表示できない文字を含んでいます。
//データの損失を防ぐために、ファイルを Unicode 形式で保存してください。
#include <cuda.h>
#include <cuda_runtime.h>
#include "nvEncodeAPI.h"
#pragma warning(pop)
#include "rgy_tchar.h"
#include "rgy_codepage.h"
#include "cpu_info.h"
#include "gpu_info.h"
#include "rgy_version.h"
#include "rgy_status.h"
#include "rgy_filesystem.h"
#include "rgy_env.h"
#include "rgy_input.h"
#include "rgy_input_raw.h"
#include "rgy_input_avi.h"
#include "rgy_input_avs.h"
#include "rgy_input_vpy.h"
#include "rgy_input_sm.h"
#include "rgy_input_avcodec.h"
#include "rgy_output.h"
#include "rgy_output_avcodec.h"
#include "rgy_chapter.h"
#include "rgy_timecode.h"
#include "rgy_aspect_ratio.h"
#include "rgy_level.h"
#include "rgy_level_hevc.h"
#include "rgy_device_info_cache.h"
#include "rgy_parallel_enc.h"
#include "NVEncPipeline.h"
#include "NVEncCore.h"
#include "NVEncFilterDelogo.h"
#include "NVEncFilterConvolution3d.h"
#include "NVEncFilterDenoiseKnn.h"
#include "NVEncFilterDenoiseNLMeans.h"
#include "NVEncFilterDenoisePmd.h"
#include "NVEncFilterDenoiseDct.h"
#include "NVEncFilterSmooth.h"
#include "NVEncFilterDenoiseFFT3D.h"
#include "NVEncFilterNvvfx.h"
#include "NVEncFilterNGX.h"
#include "NVEncFilterLibplacebo.h"
#include "NVEncFilterDeband.h"
#include "NVEncFilterDecimate.h"
#include "NVEncFilterMpdecimate.h"
#include "NVEncFilterAfs.h"
#include "NVEncFilterNnedi.h"
#include "NVEncFilterYadif.h"
#include "NVEncFilterDecomb.h"
#include "NVEncFilterRff.h"
#include "NVEncFilterUnsharp.h"
#include "NVEncFilterEdgelevel.h"
#include "NVEncFilterWarpsharp.h"
#include "NVEncFilterCurves.h"
#include "NVEncFilterTweak.h"
#include "NVEncFilterTransform.h"
#include "NVEncFilterColorspace.h"
#include "NVEncFilterSubburn.h"
#include "NVEncFilterSelectEvery.h"
#include "NVEncFilterOverlay.h"
#include "NVEncFilterNVOFFRUC.h"
#include "helper_cuda.h"
#include "helper_nvenc.h"

#pragma warning(push)
#pragma warning(disable: 4244)
#pragma warning(disable: 4834)
RGY_DISABLE_WARNING_PUSH
RGY_DISABLE_WARNING_STR("-Wunused-result")
RGY_DISABLE_WARNING_STR("-Wtautological-compare")
#define TTMATH_NOASM
#include "ttmath/ttmath.h"
#if _M_IX86
typedef ttmath::Int<4> ttint128;
#else
typedef ttmath::Int<2> ttint128;
#endif
RGY_DISABLE_WARNING_POP
#pragma warning(pop)


using std::deque;

#if ENABLE_NVTX
#include "nvToolsExt.h"
#ifdef _M_IX86
#pragma comment(lib, "nvToolsExt32_1.lib")
#else
#pragma comment(lib, "nvToolsExt64_1.lib")
#endif

class NvtxTracer {
public:
    NvtxTracer(const char *name) {
        nvtxRangePushA(name);
    }
    ~NvtxTracer() {
        nvtxRangePop();
    }
};
#define NVTXRANGE(name) NvtxTracer nvtx ## name( #name );
#else
#define NVTXRANGE(name)
#endif


class FrameBufferDataIn {
public:
    FrameBufferDataIn() : m_pInfo(), m_oVPP(), m_frameInfo(), m_bInputHost(), m_heTransferFin(NULL) {

    };
    FrameBufferDataIn(shared_ptr<CUVIDPARSERDISPINFO> pInfo, const CUVIDPROCPARAMS& oVPP, const RGYFrameInfo& frameInfo) : m_pInfo(), m_oVPP(), m_frameInfo(), m_bInputHost(false), m_heTransferFin(NULL) {
        setInfo(pInfo, oVPP, frameInfo);
    };
    FrameBufferDataIn(const FrameBufferDataIn& obj) :
        m_pInfo(obj.m_pInfo),
        m_oVPP(obj.m_oVPP),
        m_frameInfo(obj.m_frameInfo),
        m_bInputHost(obj.m_bInputHost),
        m_heTransferFin(obj.m_heTransferFin) {
    };
    ~FrameBufferDataIn() {
        m_heTransferFin.reset();
        m_pInfo.reset();
    }
    void setHostFrameInfo(
        const RGYFrameInfo& frameInfo, //入力フレームへのポインタと情報
        shared_ptr<void> heTransferFin //入力フレームに関連付けられたイベント、このフレームが不要になったらSetする
    ) {
        m_pInfo.reset();
        memset(&m_oVPP, 0, sizeof(m_oVPP));
        m_frameInfo = frameInfo;
        m_bInputHost = true;
        m_heTransferFin = heTransferFin;
    }
    void setCuvidInfo(shared_ptr<CUVIDPARSERDISPINFO> pInfo, const RGYFrameInfo& frameInfo) {
        m_pInfo = pInfo;
        memset(&m_oVPP, 0, sizeof(m_oVPP));

        m_frameInfo = frameInfo;
        //frameinfo.picstructの決定
        m_frameInfo.picstruct = (pInfo->progressive_frame) ? RGY_PICSTRUCT_FRAME : ((pInfo->top_field_first) ? RGY_PICSTRUCT_FRAME_TFF : RGY_PICSTRUCT_FRAME_BFF);
        //RFFに関するフラグを念のためクリア
        m_frameInfo.flags = RGY_FRAME_FLAG_NONE;
        m_frameInfo.singleAlloc = true;
        //RFFフラグの有無
        if (pInfo->repeat_first_field == 1) {
            m_frameInfo.flags |= RGY_FRAME_FLAG_RFF;
        }
        //TFFかBFFか
        m_frameInfo.flags |= (pInfo->top_field_first) ? RGY_FRAME_FLAG_RFF_TFF : RGY_FRAME_FLAG_RFF_BFF;
        m_frameInfo.duration = 0;
        m_frameInfo.timestamp = pInfo->timestamp;
        m_bInputHost = false;
    }
    void setInfo(shared_ptr<CUVIDPARSERDISPINFO> pInfo, const CUVIDPROCPARAMS& oVPP, const RGYFrameInfo& frameInfo) {
        m_pInfo = pInfo;
        m_frameInfo = frameInfo;
        m_oVPP = oVPP;
    }
    shared_ptr<CUVIDPARSERDISPINFO> getCuvidInfo() {
        return m_pInfo;
    }
    CUVIDPROCPARAMS getVppInfo() {
        return m_oVPP;
    }
    void resetCuvidInfo() {
        m_pInfo.reset();
    }
    int64_t getTimeStamp() const {
        return m_frameInfo.timestamp;
    }
    void setTimeStamp(int64_t timestamp) {
        m_frameInfo.timestamp = timestamp;
    }
    int64_t getDuration() const {
        return m_frameInfo.duration;
    }
    void setDuration(int64_t duration) {
        m_frameInfo.duration = duration;
    }
    bool inputIsHost() const {
        return m_bInputHost;
    }
    RGYFrameInfo getFrameInfo() const {
        return m_frameInfo;
    }
    void setInterlaceFlag(RGY_PICSTRUCT picstruct) {
        m_frameInfo.picstruct = picstruct;
    }
    void setInputFrameId(int inputFrameId) {
        m_frameInfo.inputFrameId = inputFrameId;
    }
    void addFrameData(std::shared_ptr<RGYFrameData> data) {
        if (data) {
            m_frameInfo.dataList.push_back(data);
        }
    }
    void addFrameData(std::vector<std::shared_ptr<RGYFrameData>> data) {
        for (auto& d : data) {
            addFrameData(d);
        }
    }
private:
    shared_ptr<CUVIDPARSERDISPINFO> m_pInfo;
    CUVIDPROCPARAMS m_oVPP;
    RGYFrameInfo m_frameInfo;
    bool m_bInputHost;      //入力フレームへのポインタと情報
    shared_ptr<void> m_heTransferFin; //入力フレームに関連付けられたイベント、このフレームが不要になったらSetする
};

class FrameBufferDataEnc {
public:
    RGY_CSP m_csp;
    int64_t m_timestamp;
    int64_t m_duration;
    int m_inputFrameId;
    EncodeBuffer *m_pEncodeBuffer;
    cudaEvent_t *m_pEvent;
    std::vector<std::shared_ptr<RGYFrameData>> m_frameDataList;
    FrameBufferDataEnc(RGY_CSP csp, int64_t timestamp, int64_t duration, int inputFrameId, EncodeBuffer *pEncodeBuffer, cudaEvent_t *pEvent, std::vector<std::shared_ptr<RGYFrameData>>& frameDataList) :
        m_csp(csp),
        m_timestamp(timestamp),
        m_duration(duration),
        m_inputFrameId(inputFrameId),
        m_pEncodeBuffer(pEncodeBuffer),
        m_pEvent(pEvent),
        m_frameDataList(frameDataList) {
    };
    ~FrameBufferDataEnc() {
    }
};

NVEncCore::NVEncCore() :
    m_dev(),
    m_pDecoder(),
    m_deviceUsage(),
    m_encRunCtx(),
    m_parallelEnc(),
    m_devNames(),
    m_pAbortByUser(nullptr),
    m_cudaSchedule(CU_CTX_SCHED_AUTO),
    m_stCreateEncodeParams(),
    m_dynamicRC(),
    m_appliedDynamicRC(DYNAMIC_PARAM_NOT_SELECTED),
    m_pipelineDepth(PIPELINE_DEPTH),
    m_inputHostBuffer(),
    m_outputFrameHostRaw(),
    m_trimParam(),
    m_poolPkt(),
    m_poolFrame(),
    m_pFileReader(),
    m_AudioReaders(),
    m_pFileWriter(),
    m_pFileWriterListAudio(),
    m_pStatus(),
    m_pPerfMonitor(),
    m_stPicStruct(),
    m_stEncConfig(),
    m_keyOnChapter(false),
    m_keyFile(),
    m_Chapters(),
    m_hdr10plusMetadataCopy(false),
    m_timecode(),
    m_hdr10plus(),
    m_hdrseiIn(),
    m_hdrseiOut(),
    m_dovirpu(),
    m_dovirpuMetadataCopy(false),
    m_doviProfile(RGY_DOVI_PROFILE_UNSET),
    m_encTimestamp(),
    m_encodeFrameID(0),
    m_videoIgnoreTimestampError(DEFAULT_VIDEO_IGNORE_TIMESTAMP_ERROR),
    m_vpFilters(),
    m_pLastFilterParam(),
#if ENABLE_SSIM
    m_videoQualityMetric(),
#endif //#if ENABLE_SSIM
    m_stCodecGUID(),
    m_uEncWidth(0),
    m_uEncHeight(0),
    m_sar(),
    m_encVUI(),
    m_rgbAsYUV444(),
    m_nProcSpeedLimit(0),
    m_taskPerfMonitor(false),
    m_nAVSyncMode(RGY_AVSYNC_AUTO),
    m_timestampPassThrough(false),
    m_inputFps(),
    m_outputTimebase(),
    m_encFps(),
    m_pipelineTasks(),
    m_encodeBufferCount(16),
    m_EncodeBufferQueue(),
    m_stEOSOutputBfr(),
    m_stEncodeBuffer() {
    m_trimParam.offset = 0;
    m_keyFile.clear();
    m_keyOnChapter = false;
    memset(&m_stCreateEncodeParams, 0, sizeof(m_stCreateEncodeParams));
    memset(&m_stEncConfig, 0, sizeof(m_stEncConfig));
    memset(&m_stCodecGUID,    0, sizeof(m_stCodecGUID));
    memset(&m_stEOSOutputBfr, 0, sizeof(m_stEOSOutputBfr));
    memset(&m_stEncodeBuffer, 0, sizeof(m_stEncodeBuffer));
}

NVEncCore::~NVEncCore() {
    Deinitialize();
}

void NVEncCore::SetAbortFlagPointer(bool *abortFlag) {
    m_pAbortByUser = abortFlag;
}

//エンコーダが出力使用する色空間を入力パラメータをもとに取得
RGY_CSP NVEncCore::GetEncoderCSP(const InEncodeVideoParam *inputParam) const {
    const bool bOutputHighBitDepth = encodeIsHighBitDepth(inputParam);
    if (   RGY_CSP_CHROMA_FORMAT[inputParam->outputCsp] == RGY_CHROMAFMT_RGB
        || RGY_CSP_CHROMA_FORMAT[inputParam->outputCsp] == RGY_CHROMAFMT_RGB_PACKED) {
        return (bOutputHighBitDepth) ? RGY_CSP_GBR_16 : RGY_CSP_GBR;
    }
    const bool yuv422 = RGY_CSP_CHROMA_FORMAT[inputParam->outputCsp] == RGY_CHROMAFMT_YUV422;
    const bool yuv444 = RGY_CSP_CHROMA_FORMAT[inputParam->outputCsp] == RGY_CHROMAFMT_YUV444 || RGY_CSP_CHROMA_FORMAT[inputParam->outputCsp] == RGY_CHROMAFMT_RGB || RGY_CSP_CHROMA_FORMAT[inputParam->outputCsp] == RGY_CHROMAFMT_RGB_PACKED;
    if (rgy_csp_has_alpha(inputParam->outputCsp)) {
        if (bOutputHighBitDepth) {
            return (yuv444) ? RGY_CSP_YUVA444_16 : ((yuv422) ? RGY_CSP_NA : RGY_CSP_P010A);
        } else {
            return (yuv444) ? RGY_CSP_YUVA444 : ((yuv422) ? RGY_CSP_NA : RGY_CSP_NV12A);
        }
    } else {
        if (bOutputHighBitDepth) {
            return (yuv444) ? RGY_CSP_YUV444_16 : ((yuv422) ? RGY_CSP_P210 : RGY_CSP_P010);
        } else {
            return (yuv444) ? RGY_CSP_YUV444 : ((yuv422) ? RGY_CSP_NV16 : RGY_CSP_NV12);
        }
    }
}

RGY_CSP NVEncCore::GetRawOutCSP(const InEncodeVideoParam *inputParam) const {
    if (inputParam->codec_rgy != RGY_CODEC_RAW && inputParam->codec_rgy != RGY_CODEC_AVCODEC) {
        return GetEncoderCSP(inputParam);
    }
    const bool bOutputHighBitDepth = encodeIsHighBitDepth(inputParam);
    if (   RGY_CSP_CHROMA_FORMAT[inputParam->outputCsp] == RGY_CHROMAFMT_RGB
        || RGY_CSP_CHROMA_FORMAT[inputParam->outputCsp] == RGY_CHROMAFMT_RGB_PACKED) {
        return (bOutputHighBitDepth) ? RGY_CSP_GBR_16 : RGY_CSP_GBR;
    }
    const bool yuv422 = RGY_CSP_CHROMA_FORMAT[inputParam->outputCsp] == RGY_CHROMAFMT_YUV422;
    const bool yuv444 = RGY_CSP_CHROMA_FORMAT[inputParam->outputCsp] == RGY_CHROMAFMT_YUV444;
    switch (inputParam->outputDepth) {
    case 10: return (yuv444) ? RGY_CSP_YUV444_10 : ((yuv422) ? RGY_CSP_YUV422_10 : RGY_CSP_YV12_10);
    case 12: return (yuv444) ? RGY_CSP_YUV444_12 : ((yuv422) ? RGY_CSP_YUV422_12 : RGY_CSP_YV12_12);
    case 14: return (yuv444) ? RGY_CSP_YUV444_14 : ((yuv422) ? RGY_CSP_YUV422_14 : RGY_CSP_YV12_14);
    case 16: return (yuv444) ? RGY_CSP_YUV444_16 : ((yuv422) ? RGY_CSP_YUV422_16 : RGY_CSP_YV12_16);
    case 8:
    default: return (yuv444) ? RGY_CSP_YUV444 : ((yuv422) ? RGY_CSP_YUV422 : RGY_CSP_YV12);
    }
}

//ログを初期化
RGY_ERR NVEncCore::InitLog(const InEncodeVideoParam *inputParam) {
    //ログの初期化
    m_pLog.reset(new RGYLog(inputParam->ctrl.logfile.c_str(), inputParam->ctrl.loglevel, inputParam->ctrl.logOpt.addTime, inputParam->ctrl.logOpt.addLogLevel, inputParam->ctrl.logOpt.disableColor));
    if ((inputParam->ctrl.logfile.length() > 0 || inputParam->common.outputFilename.length() > 0) && inputParam->input.type != RGY_INPUT_FMT_SM) {
        m_pLog->writeFileHeader(inputParam->common.outputFilename.c_str());
    }
    return RGY_ERR_NONE;
}

RGY_ERR NVEncCore::readChapterFile(const tstring& chapfile) {
    ChapterRW chapter;
    auto err = chapter.read_file(chapfile.c_str(), CODE_PAGE_UNSET, 0.0);
    if (err != AUO_CHAP_ERR_NONE) {
        PrintMes(RGY_LOG_ERROR, _T("failed to %s chapter file: \"%s\".\n"), (err == AUO_CHAP_ERR_FILE_OPEN) ? _T("open") : _T("read"), chapfile.c_str());
        return RGY_ERR_FILE_OPEN;
    }
    if (chapter.chapterlist().size() == 0) {
        PrintMes(RGY_LOG_ERROR, _T("no chapter found from chapter file: \"%s\".\n"), chapfile.c_str());
        return RGY_ERR_INVALID_PARAM;
    }
    m_Chapters.clear();
    const auto& chapter_list = chapter.chapterlist();
    tstring chap_log;
    for (size_t i = 0; i < chapter_list.size(); i++) {
        unique_ptr<AVChapter> avchap(new AVChapter);
        avchap->time_base = av_make_q(1, 1000);
        avchap->start = chapter_list[i]->get_ms();
        avchap->end = (i < chapter_list.size()-1) ? chapter_list[i+1]->get_ms() : avchap->start + 1;
        avchap->id = (int)m_Chapters.size();
        avchap->metadata = nullptr;
        av_dict_set(&avchap->metadata, "title", chapter_list[i]->name.c_str(), 0); //chapter_list[i]->nameはUTF-8になっている
        chap_log += strsprintf(_T("chapter #%02d [%d.%02d.%02d.%03d]: %s.\n"),
            avchap->id, chapter_list[i]->h, chapter_list[i]->m, chapter_list[i]->s, chapter_list[i]->ms,
            char_to_tstring(chapter_list[i]->name, CODE_PAGE_UTF8).c_str()); //chapter_list[i]->nameはUTF-8になっている
        m_Chapters.push_back(std::move(avchap));
    }
    PrintMes(RGY_LOG_DEBUG, _T("%s"), chap_log.c_str());
    return RGY_ERR_NONE;
}

RGY_ERR NVEncCore::InitChapters(const InEncodeVideoParam *inputParam) {
    m_Chapters.clear();
    if (inputParam->common.chapterFile.length() > 0) {
        //チャプターファイルを読み込む
        auto sts = readChapterFile(inputParam->common.chapterFile);
        if (sts != RGY_ERR_NONE) {
            return sts;
        }
    }
    if (m_Chapters.size() == 0) {
        auto pAVCodecReader = std::dynamic_pointer_cast<RGYInputAvcodec>(m_pFileReader);
        if (pAVCodecReader != nullptr) {
            auto chapterList = pAVCodecReader->GetChapterList();
            //入力ファイルのチャプターをコピーする
            for (uint32_t i = 0; i < chapterList.size(); i++) {
                unique_ptr<AVChapter> avchap(new AVChapter);
                *avchap = *chapterList[i];
                m_Chapters.push_back(std::move(avchap));
            }
        }
    }
    if (m_Chapters.size() > 0) {
        if (inputParam->common.keyOnChapter && m_trimParam.list.size() > 0) {
            PrintMes(RGY_LOG_WARN, _T("--key-on-chap not supported when using --trim.\n"));
        } else {
            m_keyOnChapter = inputParam->common.keyOnChapter;
        }
    }
    return RGY_ERR_NONE;
}

RGY_ERR NVEncCore::InitInput(InEncodeVideoParam *inputParam, DeviceCodecCsp& HWDecCodecCsp) {
    m_pStatus.reset(new EncodeStatus());

    int subburnTrackId = 0;
    for (const auto &subburn : inputParam->vpp.subburn) {
        if (subburn.trackId > 0) {
            subburnTrackId = subburn.trackId;
            break;
        }
    }
    if (inputParam->vpp.smooth.enable && inputParam->vpp.smooth.qp <= 0) {
        m_qpTable = std::make_unique<RGYListRef<RGYFrameDataQP>>();
    }

    //--input-cspの値 (raw読み込み用の入力色空間)
    //この後上書きするので、ここで保存する
    const auto inputCspOfRawReader = inputParam->input.csp;

    //入力モジュールが、エンコーダに返すべき色空間をセット
    const bool bOutputHighBitDepth = encodeIsHighBitDepth(inputParam);
    if (inputParam->lossless
        || (RGY_CSP_CHROMA_FORMAT[inputParam->outputCsp] == RGY_CHROMAFMT_RGB || RGY_CSP_CHROMA_FORMAT[inputParam->outputCsp] == RGY_CHROMAFMT_RGB_PACKED)
        || rgy_csp_has_alpha(inputParam->outputCsp)
        || inputParam->vpp.colorspace.enable) {
        inputParam->input.csp = RGY_CSP_NA; //なるべくそのままの色空間のままGPUへ転送する
    } else {
        if (bOutputHighBitDepth) {
            switch (RGY_CSP_CHROMA_FORMAT[inputParam->outputCsp]) {
            case RGY_CHROMAFMT_YUV444:
            case RGY_CHROMAFMT_RGB:
            case RGY_CHROMAFMT_RGB_PACKED:
                inputParam->input.csp = RGY_CSP_YUV444_16; break;
            case RGY_CHROMAFMT_YUV422: inputParam->input.csp = RGY_CSP_P210; break;
            default: inputParam->input.csp = RGY_CSP_P010; break;
            }
        } else {
            switch (RGY_CSP_CHROMA_FORMAT[inputParam->outputCsp]) {
            case RGY_CHROMAFMT_YUV444:
            case RGY_CHROMAFMT_RGB:
            case RGY_CHROMAFMT_RGB_PACKED:
                inputParam->input.csp = RGY_CSP_YUV444; break;
            case RGY_CHROMAFMT_YUV422: inputParam->input.csp = RGY_CSP_NV16; break;
            default: inputParam->input.csp = RGY_CSP_NV12; break;
            }
        }
    }
    // インタレ解除が指定され、かつインタレの指定がない場合は、自動的にインタレの情報取得を行う
    int deinterlacer = 0;
    if (inputParam->vppnv.deinterlace != cudaVideoDeinterlaceMode_Weave) deinterlacer++;
    if (inputParam->vpp.afs.enable) deinterlacer++;
    if (inputParam->vpp.nnedi.enable) deinterlacer++;
    if (inputParam->vpp.yadif.enable) deinterlacer++;
    if (inputParam->vpp.decomb.enable) deinterlacer++;
    if (deinterlacer > 0 && ((inputParam->input.picstruct & RGY_PICSTRUCT_INTERLACED) == 0)) {
        inputParam->input.picstruct = RGY_PICSTRUCT_AUTO;
    }

    m_poolPkt = std::make_unique<RGYPoolAVPacket>();
    m_poolFrame = std::make_unique<RGYPoolAVFrame>();

    //入力モジュールの初期化
    if (auto sts =initReaders(m_pFileReader, m_AudioReaders, &inputParam->input, &inputParam->inprm, inputCspOfRawReader,
        m_pStatus, &inputParam->common, &inputParam->ctrl, HWDecCodecCsp, subburnTrackId,
        inputParam->vpp.rff.enable, inputParam->vpp.afs.enable, inputParam->vpp.libplacebo_tonemapping.enable,
        m_poolPkt.get(), m_poolFrame.get(), m_qpTable.get(), m_pPerfMonitor.get(), m_pLog); sts != RGY_ERR_NONE) {
        PrintMes(RGY_LOG_ERROR, _T("failed to initialize file reader(s).\n"));
        return sts;
    }

    m_inputFps = rgy_rational<int>(inputParam->input.fpsN, inputParam->input.fpsD);
    m_outputTimebase = (inputParam->common.timebase.is_valid()) ? inputParam->common.timebase : m_inputFps.inv() * rgy_rational<int>(1, 4);
    m_timestampPassThrough = inputParam->common.timestampPassThrough;
    if (inputParam->common.timestampPassThrough) {
        PrintMes(RGY_LOG_DEBUG, _T("Switching to VFR mode as --timestamp-paththrough is used.\n"));
        m_nAVSyncMode = RGY_AVSYNC_VFR;
    }

    auto pAVCodecReader = std::dynamic_pointer_cast<RGYInputAvcodec>(m_pFileReader);
    if (pAVCodecReader && inputParam->vpp.mpdecimate.enable) {
        m_nAVSyncMode |= RGY_AVSYNC_VFR;
        PrintMes(RGY_LOG_DEBUG, _T("Switching to VFR mode as --vpp-mpdecimate is activated.\n"));
    }
    if (inputParam->common.tcfileIn.length() > 0) {
        PrintMes(RGY_LOG_DEBUG, _T("Switching to VFR mode as --tcfile-in is used.\n"));
        m_nAVSyncMode |= RGY_AVSYNC_VFR;
    }
    if (m_nAVSyncMode & RGY_AVSYNC_VFR) {
        //avsync vfr時は、入力streamのtimebaseをそのまま使用する
        m_outputTimebase = m_pFileReader->getInputTimebase();
        if (inputParam->vpp.afs.enable) {
            m_outputTimebase *= rgy_rational<int>(1, 4);
        }
    }

    if (
        pAVCodecReader == nullptr &&
        inputParam->common.pTrimList && inputParam->common.nTrimCount > 0) {
        //avhw/avswリーダー以外は、trimは自分ではセットされないので、ここでセットする
        sTrimParam trimParam;
        trimParam.list = make_vector(inputParam->common.pTrimList, inputParam->common.nTrimCount);
        trimParam.offset = 0;
        m_pFileReader->SetTrimParam(trimParam);
    }
    //trim情報をリーダーから取得する
    m_trimParam = m_pFileReader->GetTrimParam();
    if (m_trimParam.list.size() > 0) {
        PrintMes(RGY_LOG_DEBUG, _T("Input: trim options\n"));
        for (int i = 0; i < (int)m_trimParam.list.size(); i++) {
            PrintMes(RGY_LOG_DEBUG, _T("%d-%d "), m_trimParam.list[i].start, m_trimParam.list[i].fin);
        }
        PrintMes(RGY_LOG_DEBUG, _T(" (offset: %d)\n"), m_trimParam.offset);
    }

    if ((m_nAVSyncMode & (RGY_AVSYNC_VFR | RGY_AVSYNC_FORCE_CFR)) || inputParam->vpp.rff.enable) {
        tstring err_target;
        if (m_nAVSyncMode & RGY_AVSYNC_VFR)       err_target += _T("avsync vfr, ");
        if (m_nAVSyncMode & RGY_AVSYNC_FORCE_CFR) err_target += _T("avsync forcecfr, ");
        if (inputParam->vpp.rff.enable)           err_target += _T("vpp-rff, ");
        err_target = err_target.substr(0, err_target.length()-2);

        if (pAVCodecReader) {
            //timestampになんらかの問題がある場合、vpp-rffとavsync vfrは使用できない
            const auto timestamp_status = pAVCodecReader->GetFramePosList()->getStreamPtsStatus();
            if ((timestamp_status & (~RGY_PTS_NORMAL)) != 0) {

                tstring err_sts;
                if (timestamp_status & RGY_PTS_SOMETIMES_INVALID) err_sts += _T("SOMETIMES_INVALID, "); //時折、無効なptsを得る
                if (timestamp_status & RGY_PTS_HALF_INVALID)      err_sts += _T("HALF_INVALID, "); //PAFFなため、半分のフレームのptsやdtsが無効
                if (timestamp_status & RGY_PTS_ALL_INVALID)       err_sts += _T("ALL_INVALID, "); //すべてのフレームのptsやdtsが無効
                if (timestamp_status & RGY_PTS_NONKEY_INVALID)    err_sts += _T("NONKEY_INVALID, "); //キーフレーム以外のフレームのptsやdtsが無効
                if (timestamp_status & RGY_PTS_DUPLICATE)         err_sts += _T("PTS_DUPLICATE, "); //重複するpts/dtsが存在する
                if (timestamp_status & RGY_DTS_SOMETIMES_INVALID) err_sts += _T("DTS_SOMETIMES_INVALID, "); //時折、無効なdtsを得る
                err_sts = err_sts.substr(0, err_sts.length()-2);

                PrintMes(RGY_LOG_ERROR, _T("timestamp not acquired successfully from input stream, %s cannot be used. \n  [0x%x] %s\n"),
                    err_target.c_str(), (uint32_t)timestamp_status, err_sts.c_str());
                return RGY_ERR_INVALID_PARAM;
            }
            PrintMes(RGY_LOG_DEBUG, _T("timestamp check: 0x%x\n"), timestamp_status);
        } else if (m_outputTimebase.n() == 0 || !m_outputTimebase.is_valid()) {
            PrintMes(RGY_LOG_ERROR, _T("%s cannot be used with current reader.\n"), err_target.c_str());
            return RGY_ERR_INVALID_PARAM;
        }
    } else if (pAVCodecReader && ((pAVCodecReader->GetFramePosList()->getStreamPtsStatus() & (~RGY_PTS_NORMAL)) == 0)) {
        m_nAVSyncMode |= RGY_AVSYNC_VFR;
        const auto timebaseStreamIn = to_rgy(pAVCodecReader->GetInputVideoStream()->time_base);
        if (!inputParam->common.timebase.is_valid()
            && ((timebaseStreamIn.inv() * m_inputFps.inv()).d() == 1 || timebaseStreamIn.n() > 1000)) { //fpsを割り切れるtimebaseなら
            if (!inputParam->vpp.afs.enable && !inputParam->vpp.rff.enable) {
                m_outputTimebase = m_inputFps.inv() * rgy_rational<int>(1, 4);
            }
        }
        PrintMes(RGY_LOG_DEBUG, _T("vfr mode automatically enabled with timebase %d/%d\n"), m_outputTimebase.n(), m_outputTimebase.d());
    }
    if (inputParam->vpp.fruc.enable) {
        rgy_rational<int> frucfps;
        switch (inputParam->vpp.fruc.mode) {
        case VppFrucMode::NVOFFRUCx2:
            frucfps = m_inputFps * 2;
            break;
        case VppFrucMode::NVOFFRUCFps:
            frucfps = inputParam->vpp.fruc.targetFps;
            break;
        default:
            break;
        }
        if (frucfps.is_valid()) {
            const auto timbeaselcm = rgy_lcm(m_outputTimebase.d(), frucfps.n() * 2);
            m_outputTimebase *= rgy_rational<int>(1, timbeaselcm / m_outputTimebase.d());
            PrintMes(RGY_LOG_DEBUG, _T("timebase changed to %d/%d, as vpp-fruc targets %d/%d fps\n"), m_outputTimebase.n(), m_outputTimebase.d(), frucfps.n(), frucfps.d());
        }
    }
#if !FOR_AUO
    if (inputParam->common.dynamicHdr10plusJson.length() > 0) {
        m_hdr10plus = initDynamicHDR10Plus(inputParam->common.dynamicHdr10plusJson, m_pLog);
        if (!m_hdr10plus) {
            PrintMes(RGY_LOG_ERROR, _T("Failed to initialize hdr10plus reader.\n"));
            return RGY_ERR_INVALID_PARAM;
        }
    } else if (inputParam->common.hdr10plusMetadataCopy) {
        m_hdr10plusMetadataCopy = true;
    }

    if (inputParam->common.doviRpuFile.length() > 0) {
        m_dovirpu = std::make_unique<DOVIRpu>();
        if (m_dovirpu->init(inputParam->common.doviRpuFile.c_str()) != 0) {
            PrintMes(RGY_LOG_ERROR, _T("Failed to open dovi rpu \"%s\".\n"), inputParam->common.doviRpuFile.c_str());
            return RGY_ERR_INVALID_PARAM;
        }
    } else if (inputParam->common.doviRpuMetadataCopy) {
        m_dovirpuMetadataCopy = true;
    }
    m_doviProfile = inputParam->common.doviProfile;
#endif

    m_hdrseiIn = createHEVCHDRSei(maxCLLSource, masterDisplaySource, RGY_TRANSFER_UNKNOWN, m_pFileReader.get());
    if (!m_hdrseiIn) {
        PrintMes(RGY_LOG_ERROR, _T("Failed to parse HEVC HDR10 metadata.\n"));
        return RGY_ERR_INVALID_PARAM;
    }

    m_hdrseiOut = createHEVCHDRSei(inputParam->common.maxCll, inputParam->common.masterDisplay, inputParam->common.atcSei, m_pFileReader.get());
    if (!m_hdrseiOut) {
        PrintMes(RGY_LOG_ERROR, _T("Failed to parse HEVC HDR10 metadata.\n"));
        return RGY_ERR_INVALID_PARAM;
    }

    return RGY_ERR_NONE;
}

//Power throttolingは消費電力削減に有効だが、
//fpsが高い場合やvppフィルタを使用する場合は、速度に悪影響がある場合がある
//そのあたりを適当に考慮し、throttolingのauto/onを自動的に切り替え
RGY_ERR NVEncCore::InitPowerThrottoling(InEncodeVideoParam *inputParam) {
    //解像度が低いほど、fpsが出やすい
    int score_resolution = 0;
    const int outputResolution = m_uEncWidth * m_uEncHeight;
    if (outputResolution <= 1024 * 576) {
        score_resolution += 4;
    } else if (outputResolution <= 1280 * 720) {
        score_resolution += 3;
    } else if (outputResolution <= 1920 * 1080) {
        score_resolution += 2;
    } else if (outputResolution <= 2560 * 1440) {
        score_resolution += 1;
    }
    if (inputParam->ctrl.parallelEnc.isEnabled()) {
        score_resolution *= inputParam->ctrl.parallelEnc.parallelCount;
    }
    const bool speedLimit = inputParam->ctrl.procSpeedLimit > 0 && inputParam->ctrl.procSpeedLimit <= 240;
    const int score = (speedLimit) ? 0 : score_resolution;

    //一定以上のスコアなら、throttolingをAuto、それ以外はthrottolingを有効にして消費電力を削減
    const int score_threshold = 3;
    const auto mode = (score >= score_threshold) ? RGYThreadPowerThrottlingMode::Auto : RGYThreadPowerThrottlingMode::Enabled;
    PrintMes(RGY_LOG_DEBUG, _T("selected mode %s : score %d: resolution %d, speed limit %s.\n"),
        rgy_thread_power_throttoling_mode_to_str(mode), score, score_resolution, speedLimit ? _T("on") : _T("off"));

    if (inputParam->ctrl.parallelEnc.isEnabled()) {
        // 並列エンコード時には音声スレッドと出力スレッドが重要なので、throttolingを有効にはならないように
        auto& target = inputParam->ctrl.threadParams.get(RGYThreadType::AUDIO);
        if (target.throttling == RGYThreadPowerThrottlingMode::Unset) {
            target.throttling = RGYThreadPowerThrottlingMode::Auto;
        }
        target = inputParam->ctrl.threadParams.get(RGYThreadType::OUTPUT);
        if (target.throttling == RGYThreadPowerThrottlingMode::Unset) {
            target.throttling = RGYThreadPowerThrottlingMode::Auto;
        }
    }

    for (int i = (int)RGYThreadType::ALL + 1; i < (int)RGYThreadType::END; i++) {
        auto& target = inputParam->ctrl.threadParams.get((RGYThreadType)i);
        if (target.throttling == RGYThreadPowerThrottlingMode::Unset) {
            target.throttling = mode;
        }
    }
    return RGY_ERR_NONE;
}

RGY_ERR NVEncCore::InitPerfMonitor(const InEncodeVideoParam *inputParam) {
    const bool bLogOutput = inputParam->ctrl.perfMonitorSelect || inputParam->ctrl.perfMonitorSelectMatplot;
    tstring perfMonLog;
    if (bLogOutput) {
        perfMonLog = inputParam->common.outputFilename + _T("_perf.csv");
    }
    CPerfMonitorPrm perfMonitorPrm;
#if ENABLE_NVML
    perfMonitorPrm.pciBusId = m_dev->pciBusId();
#endif
    if (m_pPerfMonitor->init(perfMonLog.c_str(), _T(""), (bLogOutput) ? inputParam->ctrl.perfMonitorInterval : 1000,
        (int)inputParam->ctrl.perfMonitorSelect, (int)inputParam->ctrl.perfMonitorSelectMatplot,
#if defined(_WIN32) || defined(_WIN64)
        std::unique_ptr<void, handle_deleter>(OpenThread(SYNCHRONIZE | THREAD_QUERY_INFORMATION, false, GetCurrentThreadId()), handle_deleter()),
#else
        nullptr,
#endif
        inputParam->ctrl.threadParams.get(RGYThreadType::PERF_MONITOR),
        m_pLog, &perfMonitorPrm)) {
        PrintMes(RGY_LOG_WARN, _T("Failed to initialize performance monitor, disabled.\n"));
        m_pPerfMonitor.reset();
    }
    return RGY_ERR_NONE;
}

RGY_ERR NVEncCore::InitParallelEncode(InEncodeVideoParam *inputParam, std::vector<std::unique_ptr<NVGPUInfo>>& gpuList) {
    if (!inputParam->ctrl.parallelEnc.isEnabled()) {
        return RGY_ERR_NONE;
    }
    const bool isChild = inputParam->ctrl.parallelEnc.isChild();
    auto [sts, errmes] = RGYParallelEnc::isParallelEncPossible(inputParam, m_pFileReader.get());
    if (sts != RGY_ERR_NONE) {
        PrintMes(RGY_LOG_WARN, _T("%s"), errmes);
        inputParam->ctrl.parallelEnc.parallelCount = 0;
        inputParam->ctrl.parallelEnc.parallelId = -1;
        return (isChild) ? sts : RGY_ERR_NONE; // 子スレッド側でエラーが起こった場合はエラー、親の場合は正常終了(並列動作を無効化して継続)を返す
    }
    // 並列処理が有効の場合、メインスレッドではエンコードは行わないので、m_deviceUsageは解放する
    if (inputParam->ctrl.parallelEnc.isParent() && m_deviceUsage) {
        m_deviceUsage->close();
    }
    bool delayChildSync = false;
    if (inputParam->ctrl.parallelEnc.isParent()) {
        const int encoderCount = std::max(std::accumulate(gpuList.begin(), gpuList.end(), 0, [&](int sum, const auto& gpu) {
            return sum + ((gpu) ? gpu->encoder_count(codec_guid_rgy_to_enc(inputParam->codec_rgy)) : 0);
            }) + m_dev->encoder_count(codec_guid_rgy_to_enc(inputParam->codec_rgy)), (int)gpuList.size());
        // とんでもなく大きい値にする人がいそうなので、適当に制限する
        const int maxParallelCount = std::max(4, encoderCount * 2);
        // nvvfx, ngx使用時はGPUメモリ使用量の問題があるため、GPUにつき1スレッドに制限する
        const bool limitOnePerGPU = inputParam->vppnv.nvvfxArtifactReduction.enable
            || inputParam->vppnv.nvvfxDenoise.enable
            || inputParam->vppnv.ngxTrueHDR.enable
            || isNvvfxResizeFiter(inputParam->vpp.resize_algo)
            || isNgxResizeFiter(inputParam->vpp.resize_algo);
        if (inputParam->ctrl.parallelEnc.parallelCount < 0) {
            inputParam->ctrl.parallelEnc.parallelCount = (limitOnePerGPU) ? (int)gpuList.size() : encoderCount;
            PrintMes(RGY_LOG_DEBUG, _T("parallelCount set to %d\n"), inputParam->ctrl.parallelEnc.parallelCount);
        } else if (limitOnePerGPU && inputParam->ctrl.parallelEnc.parallelCount > (int)gpuList.size()) {
            inputParam->ctrl.parallelEnc.parallelCount = (int)gpuList.size();
            if (inputParam->ctrl.parallelEnc.parallelCount <= 1) {
                PrintMes(RGY_LOG_WARN, _T("Parallel encoding disabled, as nvvfx/ngx filter is enabled, which has large GPU RAM usage.\n"));
                inputParam->ctrl.parallelEnc.parallelCount = 0;
                inputParam->ctrl.parallelEnc.parallelId = -1;
                return RGY_ERR_NONE;
            }
            PrintMes(RGY_LOG_WARN, _T("Parallel count limited to %d, as nvvfx/ngx filter is enabled, which has large GPU RAM usage.\n"), inputParam->ctrl.parallelEnc.parallelCount);
        } else if (inputParam->ctrl.parallelEnc.parallelCount > maxParallelCount) {
            inputParam->ctrl.parallelEnc.parallelCount = maxParallelCount;
            PrintMes(RGY_LOG_WARN, _T("Parallel count limited to %d.\n"), inputParam->ctrl.parallelEnc.parallelCount);
        }
        if (inputParam->ctrl.parallelEnc.parallelCount <= 1) { // 並列数が1以下ならparallelを無効化
            inputParam->ctrl.parallelEnc.parallelCount = 0;
            inputParam->ctrl.parallelEnc.parallelId = -1;
            PrintMes(RGY_LOG_DEBUG, _T("Parallel encoding disabled, as parallel count id set to %d.\n"), inputParam->ctrl.parallelEnc.parallelCount);
            return RGY_ERR_NONE;
        }
        delayChildSync = inputParam->ctrl.parallelEnc.parallelCount > encoderCount;
    }
    m_parallelEnc = std::make_unique<RGYParallelEnc>(m_pLog);
    if ((sts = m_parallelEnc->parallelRun(inputParam, m_pFileReader.get(), m_outputTimebase, delayChildSync, m_pStatus.get(),
        // 子スレッドの場合はパフォーマンスカウンタは親と共有する
        inputParam->ctrl.parallelEnc.isParent() ? m_pPerfMonitor.get() : nullptr)) != RGY_ERR_NONE) {
        if (inputParam->ctrl.parallelEnc.isChild()) {
            return sts; // 子スレッド側でエラーが起こった場合はエラー
        }
        // うまくいかなかった場合、並列処理を無効化して続行する
        PrintMes(RGY_LOG_WARN, _T("Failed to initialize parallel encoding, disabled.\n"));
        m_parallelEnc.reset();
        // m_deviceUsageはいったん解放したので、登録を再追加
        if (m_deviceUsage) {
            m_deviceUsage = std::make_unique<RGYDeviceUsage>();
            auto devUsageLock = m_deviceUsage->lock(); // ロックは親プロセス側でとる
            // 登録を解除するプロセスを起動
            const auto [err_run_proc, child_pid] = m_deviceUsage->startProcessMonitor(m_dev->id());
            if (err_run_proc == RGY_ERR_NONE) {
                // プロセスが起動できたら、その子プロセスのIDを登録する
                m_deviceUsage->add(m_dev->id(), child_pid, devUsageLock.get());
            }
        }
        return RGY_ERR_NONE; // 親の場合は正常終了(並列動作を無効化して継続)を返す
    }
    return RGY_ERR_NONE;
}

RGY_ERR NVEncCore::InitOutput(InEncodeVideoParam *inputParams, NV_ENC_BUFFER_FORMAT encBufferFormat) {
    const auto outputVideoInfo = videooutputinfo(m_stCodecGUID, encBufferFormat,
        (inputParams->codec_rgy == RGY_CODEC_RAW || inputParams->codec_rgy == RGY_CODEC_AVCODEC) &&
        (  RGY_CSP_CHROMA_FORMAT[inputParams->outputCsp] == RGY_CHROMAFMT_RGB
        || RGY_CSP_CHROMA_FORMAT[inputParams->outputCsp] == RGY_CHROMAFMT_RGB_PACKED),
        m_uEncWidth, m_uEncHeight,
        (inputParams->codec_rgy == RGY_CODEC_RAW || inputParams->codec_rgy == RGY_CODEC_AVCODEC) ? nullptr : &m_stEncConfig, m_stPicStruct,
        std::make_pair(m_sar.n(), m_sar.d()),
        m_encFps);
    if (inputParams->codec_rgy == RGY_CODEC_RAW || inputParams->codec_rgy == RGY_CODEC_AVCODEC) {
        inputParams->common.AVMuxTarget &= ~RGY_MUX_VIDEO;
    }

    if (auto sts = initWriters(m_pFileWriter, m_pFileWriterListAudio, m_pFileReader, m_AudioReaders,
        &inputParams->common, &inputParams->input, &inputParams->ctrl, outputVideoInfo,
        m_trimParam, m_outputTimebase, m_Chapters, m_hdrseiOut.get(), m_hdr10plus.get(), m_dovirpu.get(), m_encTimestamp.get(),
        false, false, false, rgy_csp_has_alpha(inputParams->outputCsp), inputParams->alphaChannelMode,
        m_poolPkt.get(), m_poolFrame.get(), m_pStatus, m_pPerfMonitor, m_pLog); sts != RGY_ERR_NONE) {
        PrintMes(RGY_LOG_ERROR, _T("failed to initialize file reader(s): %s.\n"), get_err_mes(sts));
        return sts;
    }
    if (inputParams->common.timecode) {
        m_timecode = std::make_unique<RGYTimecode>();
        const auto tcfilename = (inputParams->common.timecodeFile.length() > 0) ? inputParams->common.timecodeFile : PathRemoveExtensionS(inputParams->common.outputFilename) + _T(".timecode.txt");
        auto err = m_timecode->init(tcfilename);
        if (err != RGY_ERR_NONE) {
            PrintMes(RGY_LOG_ERROR, _T("failed to open timecode file: \"%s\": %s.\n"), tcfilename.c_str(), get_err_mes(err));
            return err;
        }
    }
    return RGY_ERR_NONE;
}

bool NVEncCore::useNVVFX(const InEncodeVideoParam *inputParam) const {
#if (!defined(_M_IX86))
    const auto& vppnv = inputParam->vppnv;
    if (   vppnv.nvvfxArtifactReduction.enable
        || vppnv.nvvfxDenoise.enable
        || vppnv.nvvfxSuperRes.enable
        || vppnv.nvvfxUpScaler.enable
        || inputParam->vpp.resize_algo == RGY_VPP_RESIZE_NVVFX_SUPER_RES) {
        return true;
    }
#endif
    return false;
}

bool NVEncCore::useNVNGX(const InEncodeVideoParam *inputParam) const {
#if (!defined(_M_IX86))
    const auto& vppnv = inputParam->vppnv;
    if (vppnv.ngxTrueHDR.enable
        || inputParam->vpp.resize_algo == RGY_VPP_RESIZE_NGX_VSR) {
        return true;
    }
#endif
    return false;
}

RGY_ERR NVEncCore::CheckGPUListByEncoder(std::vector<std::unique_ptr<NVGPUInfo>> &gpuList, InEncodeVideoParam *inputParam) {
    if (m_nDeviceId >= 0) {
        //手動で設定されている
        return RGY_ERR_NONE;
    }
    if (inputParam->ctrl.skipHWEncodeCheck) {
        return RGY_ERR_NONE;
    }
    if (inputParam->codec_rgy == RGY_CODEC_RAW || inputParam->codec_rgy == RGY_CODEC_AVCODEC) {
        return RGY_ERR_NONE;
    }
    if (inputParam->codec_rgy == RGY_CODEC_UNKNOWN) {
        PrintMes(RGY_LOG_ERROR, _T("Unknown codec.\n"));
        return RGY_ERR_UNSUPPORTED;
    }
    int highBitDepthSupportCount = 0;
    //エンコーダの対応をチェック
    tstring message; //GPUチェックのメッセージ
    for (auto gpu = gpuList.begin(); gpu != gpuList.end(); ) {
        //コーデックのチェック
        const auto codec = std::find_if((*gpu)->nvenc_codec_features().begin(), (*gpu)->nvenc_codec_features().end(), [codec_rgy = inputParam->codec_rgy](const NVEncCodecFeature& codec) {
            return codec.codec == codec_guid_rgy_to_enc(codec_rgy);
        });
        if (codec == (*gpu)->nvenc_codec_features().end()) {
            message += strsprintf(_T("GPU #%d (%s) does not support %s encoding.\n"),
                (*gpu)->id(), (*gpu)->name().c_str(), CodecToStr(inputParam->codec_rgy).c_str());
            gpu = gpuList.erase(gpu);
            continue;
        }
        // プロファイルのチェック (新フィールドに基づく)
        GUID codecProfileGUID = GUID_EMPTY;
        if (inputParam->codec_rgy == RGY_CODEC_AV1) {
            codecProfileGUID = get_guid_from_value(inputParam->av1.profile, av1_profile_names);
        } else if (inputParam->codec_rgy == RGY_CODEC_HEVC) {
            if (RGY_CSP_CHROMA_FORMAT[inputParam->outputCsp] != RGY_CHROMAFMT_YUV420) {
                codecProfileGUID = NV_ENC_HEVC_PROFILE_FREXT_GUID;
            } else if (inputParam->outputDepth > 8) {
                codecProfileGUID = NV_ENC_HEVC_PROFILE_MAIN10_GUID;
            } else {
                codecProfileGUID = get_guid_from_value(inputParam->hevc.profile, h265_profile_names);
            }
        } else if (inputParam->codec_rgy == RGY_CODEC_H264) {
            if (RGY_CSP_CHROMA_FORMAT[inputParam->outputCsp] == RGY_CHROMAFMT_YUV444 || RGY_CSP_CHROMA_FORMAT[inputParam->outputCsp] == RGY_CHROMAFMT_RGB || RGY_CSP_CHROMA_FORMAT[inputParam->outputCsp] == RGY_CHROMAFMT_RGB_PACKED) {
                codecProfileGUID = NV_ENC_H264_PROFILE_HIGH_444_GUID;
            } else if (RGY_CSP_CHROMA_FORMAT[inputParam->outputCsp] == RGY_CHROMAFMT_YUV422) {
                codecProfileGUID = NV_ENC_H264_PROFILE_HIGH_422_GUID;
            } else if (inputParam->outputDepth > 8) {
                codecProfileGUID = NV_ENC_H264_PROFILE_HIGH_10_GUID;
            } else {
                codecProfileGUID = get_guid_from_value(inputParam->h264.profile, h264_profile_names);
            }
        } else {
            PrintMes(RGY_LOG_ERROR, _T("Unknown codec.\n"));
            return RGY_ERR_UNSUPPORTED;
        }
        if (memcmp(&codecProfileGUID, &GUID_EMPTY, sizeof(GUID_EMPTY)) == 0) {
            PrintMes(RGY_LOG_DEBUG, _T("Selected profile for %s is unfound, profile will be auto selected by NVENC API.\n"), CodecToStr(inputParam->codec_rgy).c_str());
            codecProfileGUID = NV_ENC_CODEC_PROFILE_AUTOSELECT_GUID;
        }
        auto profile = std::find_if(codec->profiles.begin(), codec->profiles.end(), [codecProfileGUID](const GUID& profile_guid) {
            return 0 == memcmp(&codecProfileGUID, &profile_guid, sizeof(profile_guid));
        });
        if (profile == codec->profiles.end()) {
            if (inputParam->ctrl.fallbackBitdepth) {
                if (inputParam->codec_rgy == RGY_CODEC_H264 && memcmp(&codecProfileGUID, &NV_ENC_H264_PROFILE_HIGH_10_GUID, sizeof(codecProfileGUID)) == 0) {
                    profile = std::find_if(codec->profiles.begin(), codec->profiles.end(), [](const GUID& profile_guid) {
                        return 0 == memcmp(&profile_guid, &NV_ENC_H264_PROFILE_HIGH_GUID, sizeof(profile_guid));
                    });
                } else if (inputParam->codec_rgy == RGY_CODEC_HEVC && memcmp(&codecProfileGUID, &NV_ENC_HEVC_PROFILE_MAIN10_GUID, sizeof(codecProfileGUID)) == 0) {
                    profile = std::find_if(codec->profiles.begin(), codec->profiles.end(), [](const GUID& profile_guid) {
                        return 0 == memcmp(&profile_guid, &NV_ENC_HEVC_PROFILE_MAIN_GUID, sizeof(profile_guid));
                    });
                }
            }
            if (profile == codec->profiles.end()) {
                message += strsprintf(_T("GPU #%d (%s) cannot encode %s %s.\n"), (*gpu)->id(), (*gpu)->name().c_str(),
                    CodecToStr(inputParam->codec_rgy).c_str(),
                    get_codec_profile_name_from_guid(inputParam->codec_rgy, codecProfileGUID).c_str());
                gpu = gpuList.erase(gpu);
                continue;
            }
        }
        if (inputParam->lossless && !get_value(NV_ENC_CAPS_SUPPORT_LOSSLESS_ENCODE, codec->caps)) {
            message += strsprintf(_T("GPU #%d (%s) does not support %s lossless encoding.\n"), (*gpu)->id(), (*gpu)->name().c_str(), CodecToStr(inputParam->codec_rgy).c_str());
            gpu = gpuList.erase(gpu);
            continue;
        }
        if ((RGY_CSP_CHROMA_FORMAT[inputParam->outputCsp] == RGY_CHROMAFMT_YUV444 || RGY_CSP_CHROMA_FORMAT[inputParam->outputCsp] == RGY_CHROMAFMT_RGB || RGY_CSP_CHROMA_FORMAT[inputParam->outputCsp] == RGY_CHROMAFMT_RGB_PACKED)
            && !get_value(NV_ENC_CAPS_SUPPORT_YUV444_ENCODE, codec->caps)) {
            message += strsprintf(_T("GPU #%d (%s) does not support %s yuv444 encoding.\n"), (*gpu)->id(), (*gpu)->name().c_str(), CodecToStr(inputParam->codec_rgy).c_str());
            gpu = gpuList.erase(gpu);
            continue;
        }
        if ((RGY_CSP_CHROMA_FORMAT[inputParam->outputCsp] == RGY_CHROMAFMT_YUV422) && !get_value(NV_ENC_CAPS_SUPPORT_YUV422_ENCODE, codec->caps)) {
            message += strsprintf(_T("GPU #%d (%s) does not support %s yuv422 encoding.\n"), (*gpu)->id(), (*gpu)->name().c_str(), CodecToStr(inputParam->codec_rgy).c_str());
            gpu = gpuList.erase(gpu);
            continue;
        }
        if (rgy_csp_has_alpha(inputParam->outputCsp) && !get_value(NV_ENC_CAPS_SUPPORT_ALPHA_LAYER_ENCODING, codec->caps)) {
            message += strsprintf(_T("GPU #%d (%s) does not support %s alpha channel encoding.\n"), (*gpu)->id(), (*gpu)->name().c_str(), CodecToStr(inputParam->codec_rgy).c_str());
            gpu = gpuList.erase(gpu);
            continue;
        }
        const bool highbitdepth = encodeIsHighBitDepth(inputParam);
        if (highbitdepth) {
            if (get_value(NV_ENC_CAPS_SUPPORT_10BIT_ENCODE, codec->caps)) {
                highBitDepthSupportCount++;
            } else if (inputParam->ctrl.fallbackBitdepth) {
                // fallbackが有効のときはここではなにもしない
            } else {
                message += strsprintf(_T("GPU #%d (%s) does not support %s 10bit depth encoding.\n"), (*gpu)->id(), (*gpu)->name().c_str(), CodecToStr(inputParam->codec_rgy).c_str());
                gpu = gpuList.erase(gpu);
                continue;
            }
        }
        if (inputParam->codec_rgy == RGY_CODEC_H264
            && (
                (inputParam->input.picstruct & RGY_PICSTRUCT_INTERLACED)
                && (inputParam->vppnv.deinterlace == cudaVideoDeinterlaceMode_Weave
                    && !inputParam->vpp.afs.enable
                    && !inputParam->vpp.nnedi.enable
                    && !inputParam->vpp.yadif.enable
                    && !inputParam->vpp.decomb.enable))
            && !get_value(NV_ENC_CAPS_SUPPORT_FIELD_ENCODING, codec->caps)) {
            message += strsprintf(_T("GPU #%d (%s) does not support H.264 interlaced encoding.\n"), (*gpu)->id(), (*gpu)->name().c_str());
            gpu = gpuList.erase(gpu);
            continue;
        }
        // デコードのチェック
        if (m_pFileReader->getInputCodec() != RGY_CODEC_UNKNOWN) {
            if (RGYInputAvcodec *pReader = dynamic_cast<RGYInputAvcodec *>(m_pFileReader.get()); pReader) {
                if (pReader->GetHWDecDeviceID().size() > 0 && !pReader->GetHWDecDeviceID().count((*gpu)->id())) {
                    message += strsprintf(_T("GPU #%d (%s) does not support required hw decoding.\n"), (*gpu)->id(), (*gpu)->name().c_str());
                    gpu = gpuList.erase(gpu);
                    continue;
                }
            }
        }

        if (inputParam->common.metric.enabled()) {
            //デコードのほうもチェックしてあげないといけない
           const auto& cuvid_csp = (*gpu)->cuvid_csp();
           if (cuvid_csp.count(inputParam->codec_rgy) == 0) {
               message += strsprintf(_T("GPU #%d (%s) does not support %s decoding required for ssim/psnr/vmaf calculation.\n"), (*gpu)->id(), (*gpu)->name().c_str(), CodecToStr(inputParam->codec_rgy).c_str());
               gpu = gpuList.erase(gpu);
               continue;
           }
           const bool yuv444 = RGY_CSP_CHROMA_FORMAT[inputParam->outputCsp] == RGY_CHROMAFMT_YUV444 || RGY_CSP_CHROMA_FORMAT[inputParam->outputCsp] == RGY_CHROMAFMT_RGB || RGY_CSP_CHROMA_FORMAT[inputParam->outputCsp] == RGY_CHROMAFMT_RGB_PACKED;
           const bool yuv422 = RGY_CSP_CHROMA_FORMAT[inputParam->outputCsp] == RGY_CHROMAFMT_YUV422;
           const auto targetCsp = (highbitdepth) ? ((yuv444) ? RGY_CSP_YUV444_10 : (yuv422 ? RGY_CSP_YUV422_10 : RGY_CSP_YV12_10))
                                                 : ((yuv444) ? RGY_CSP_YUV444    : (yuv422 ? RGY_CSP_YUV422    : RGY_CSP_YV12));
           const auto& cuvid_codec_csp = cuvid_csp.at(inputParam->codec_rgy);
           if (std::find(cuvid_codec_csp.begin(), cuvid_codec_csp.end(), targetCsp) == cuvid_codec_csp.end()) {
               message += strsprintf(_T("GPU #%d (%s) does not support %s %s decoding required for ssim/psnr/vmaf calculation.\n"), (*gpu)->id(), (*gpu)->name().c_str(), CodecToStr(inputParam->codec_rgy).c_str(), RGY_CSP_NAMES[targetCsp]);
               gpu = gpuList.erase(gpu);
               continue;
           }
        }

        //フィルタのチェック
        if (useNVVFX(inputParam) || useNVNGX(inputParam)) {
            //nvvfxにはturing以降(CC7.0)が必要
            const int nvvfxRequiredCCMajor = 7;
            if ((*gpu)->cc().first < nvvfxRequiredCCMajor) {
                message += strsprintf(_T("GPU #%d (%s) does not support nvvfx, CC 7.0 is required but GPU is CC %d.%d.\n"), (*gpu)->id(), (*gpu)->name().c_str(), (*gpu)->cc().first, (*gpu)->cc().second);
                gpu = gpuList.erase(gpu);
                continue;
            }
        }
        if (inputParam->vpp.fruc.enable) {
            //nvof-frucにはturing以降(CC7.0)が必要
            const int nvvfxRequiredCCMajor = 7;
            if ((*gpu)->cc().first < nvvfxRequiredCCMajor) {
                message += strsprintf(_T("GPU #%d (%s) does not support fruc, CC 7.0 is required but GPU is CC %d.%d.\n"), (*gpu)->id(), (*gpu)->name().c_str(), (*gpu)->cc().first, (*gpu)->cc().second);
                gpu = gpuList.erase(gpu);
                continue;
            }
        }

        PrintMes(RGY_LOG_DEBUG, _T("GPU #%d (%s) available for encode.\n"), (*gpu)->id(), (*gpu)->name().c_str());
        gpu++;
    }

    // 10bit深度のフォールバックが有効なとき
    if (encodeIsHighBitDepth(inputParam) && inputParam->ctrl.fallbackBitdepth) {
        if (highBitDepthSupportCount > 0) {
            // 10bit深度のサポートがあるGPUがあるときは、10bit深度をサポートしないGPUは外す
            for (auto gpu = gpuList.begin(); gpu != gpuList.end(); ) {
                const auto codec = std::find_if((*gpu)->nvenc_codec_features().begin(), (*gpu)->nvenc_codec_features().end(), [codec_rgy = inputParam->codec_rgy](const NVEncCodecFeature& codec) {
                    return codec.codec == codec_guid_rgy_to_enc(codec_rgy); });
                if (!get_value(NV_ENC_CAPS_SUPPORT_10BIT_ENCODE, codec->caps)) {
                    gpu = gpuList.erase(gpu);
                    continue;
                }
                gpu++;
            }
        } else {
            // 10bit深度のサポートがあるGPUがないときは、8bit深度に変更する
            PrintMes(RGY_LOG_WARN, _T("GPU(s) does not support %d bit %s depth encoding, fallback to 8bit.\n"), inputParam->outputDepth, CodecToStr(inputParam->codec_rgy).c_str());
            inputParam->outputDepth = 8;
            // プロファイルを8bit深度に変更する
            if (inputParam->codec_rgy == RGY_CODEC_H264) {
                inputParam->h264.profile = get_value_from_guid(NV_ENC_H264_PROFILE_HIGH_GUID, h264_profile_names);
            } else if (inputParam->codec_rgy == RGY_CODEC_HEVC) {
                inputParam->hevc.profile = get_value_from_guid(NV_ENC_HEVC_PROFILE_MAIN_GUID, h265_profile_names);
            }
        }
    }

    PrintMes((gpuList.size() == 0) ? RGY_LOG_ERROR : RGY_LOG_DEBUG, _T("%s\n"), message.c_str());
    if (gpuList.size() == 0) {
        return RGY_ERR_DEVICE_NOT_FOUND;
    }
    if (gpuList.size() == 1) {
        m_nDeviceId = gpuList.front()->id();
        return RGY_ERR_NONE;
    }

    auto remove_gpu_if_gpu_support_mismatch = [&](std::function<bool(const NVGPUInfo *)> check_func) {
        bool support_feature = false;
        //エンコード対象のサポートのあるGPUがあるかを確認する
        for (const auto& gpu : gpuList) {
            if (check_func(gpu.get())) {
                support_feature = true;
                break;
            }
        }
        // サポートのあるGPUがあれば、対応していないGPUは除外する
        if (support_feature) {
            for (auto gpu = gpuList.begin(); gpu != gpuList.end(); ) {
                if (!check_func(gpu->get())) {
                    gpu = gpuList.erase(gpu);
                    continue;
                }
                gpu++;
            }
        }
    };

    if (inputParam->bFrames.has_value() && inputParam->bFrames.value() > 0) {
        remove_gpu_if_gpu_support_mismatch([&](const NVGPUInfo *gpu) {
            const auto codec = std::find_if(gpu->nvenc_codec_features().begin(), gpu->nvenc_codec_features().end(), [codec_rgy = inputParam->codec_rgy](const NVEncCodecFeature& codec) {
                return codec.codec == codec_guid_rgy_to_enc(codec_rgy);
            });
            assert(codec != gpu->nvenc_codec_features().end());
            return get_value(NV_ENC_CAPS_NUM_MAX_BFRAMES, codec->caps) > 0;
        });
    }
    if (inputParam->codec_rgy != RGY_CODEC_H264 && inputParam->tuningInfo == NV_ENC_TUNING_INFO_ULTRA_HIGH_QUALITY) {
        remove_gpu_if_gpu_support_mismatch([&](const NVGPUInfo *gpu) {
            return gpu->cc().first >= 8 || (gpu->cc().first == 7 && gpu->cc().second >= 5);
        });
    }
    return RGY_ERR_NONE;
}

RGY_ERR NVEncCore::GPUAutoSelect(std::vector<std::unique_ptr<NVGPUInfo>> &gpuList, const InEncodeVideoParam *inputParam, const RGYDeviceUsageLockManager *devUsageLock) {
    UNREFERENCED_PARAMETER(inputParam);
    if (gpuList.size() <= 1) {
        m_nDeviceId = gpuList.front()->id();
        return RGY_ERR_NONE;
    }
    int maxDeviceUsageCount = 1;
    std::vector<std::pair<int, int64_t>> deviceUsage;
    if (gpuList.size() > 1) {
        deviceUsage = m_deviceUsage->getUsage(devUsageLock);
        for (size_t i = 0; i < deviceUsage.size(); i++) {
            maxDeviceUsageCount = std::max(maxDeviceUsageCount, deviceUsage[i].first);
            if (deviceUsage[i].first > 0) {
                PrintMes(RGY_LOG_DEBUG, _T("Device #%d: %d usage.\n"), i, deviceUsage[i].first);
            }
        }
    }

    const tstring PEPrefix = (inputParam->ctrl.parallelEnc.isChild()) ? strsprintf(_T("Parallel Enc %d: "), inputParam->ctrl.parallelEnc.parallelId) : _T("");
    std::map<int, double> gpuscore;
    for (const auto& gpu : gpuList) {
        const auto encoderCount = gpu->encoder_count(m_stCodecGUID);
        const int deviceUsageCount = (int)gpu->id() < (int)deviceUsage.size() ? deviceUsage[gpu->id()].first : 0;
        double usage_score = 100.0 * (maxDeviceUsageCount - deviceUsageCount) / (double)maxDeviceUsageCount;
        double core_score = gpu->cuda_cores() * inputParam->ctrl.gpuSelect.cores;
        double cc_score = (gpu->cc().first * 10.0 + gpu->cc().second) * inputParam->ctrl.gpuSelect.gen;
        double ve_score = std::max(encoderCount - 1, 0) * 25.0 * inputParam->ctrl.gpuSelect.ve;
        double gpu_score = 0.0;

        NVMLMonitorInfo info;
#if ENABLE_NVML
        NVMLMonitor monitor;
        auto nvml_ret = NVML_SUCCESS;
        if (gpu->pciBusId().length() > 0
            && (nvml_ret = monitor.Init(gpu->pciBusId())) == NVML_SUCCESS
            && monitor.getData(&info) == NVML_SUCCESS) {
#else
        NVSMIInfo nvsmi;
        if (nvsmi.getData(&info, gpu->pciBusId()) == 0) {
#endif
            ve_score  = 100.0 * (1.0 - std::pow(info.VEELoad / 100.0, 1.0)) * inputParam->ctrl.gpuSelect.ve;
            gpu_score = 100.0 * (1.0 - std::pow(info.GPULoad / 100.0, 1.5)) * inputParam->ctrl.gpuSelect.gpu;
            PrintMes(RGY_LOG_DEBUG, _T("GPU #%d (%s) Load: GPU %.1f, VE: %.1f.\n"), gpu->id(), gpu->name().c_str(), info.GPULoad, info.VEELoad);
        }
        gpuscore[gpu->id()] = usage_score + cc_score + ve_score + gpu_score + core_score;
        m_pLog->write(RGY_LOG_DEBUG, RGY_LOGT_CORE_GPU_SELECT, _T("%sGPU #%d (%s) score: %.1f: Use %.1f, VE %.1f, GPU %.1f, CC %.1f, Core %.1f.\n"), PEPrefix.c_str(), gpu->id(), gpu->name().c_str(),
            gpuscore[gpu->id()], usage_score, ve_score, gpu_score, cc_score, core_score);
    }
    std::sort(gpuList.begin(), gpuList.end(), [&](const std::unique_ptr<NVGPUInfo>& a, const std::unique_ptr<NVGPUInfo>& b) {
        if (gpuscore.at(a->id()) != gpuscore.at(b->id())) {
            return gpuscore.at(a->id()) > gpuscore.at(b->id());
        }
        return a->id() < b->id();
    });

    PrintMes(RGY_LOG_DEBUG, _T("GPU Priority\n"));
    for (const auto& gpu : gpuList) {
        PrintMes(RGY_LOG_DEBUG, _T("%sGPU #%d (%s): score %.1f\n"), PEPrefix.c_str(), gpu->id(), gpu->name().c_str(), gpuscore[gpu->id()]);
    }
    return RGY_ERR_NONE;
}

RGY_ERR NVEncCore::InitDevice(std::vector<std::unique_ptr<NVGPUInfo>> &gpuList, const InEncodeVideoParam *inputParam) {
    auto gpu = std::find_if(gpuList.begin(), gpuList.end(), [device_id = m_nDeviceId](const std::unique_ptr<NVGPUInfo> &gpuinfo) {
        return gpuinfo->id() == device_id;
    });
    if (gpu == gpuList.end()) {
        PrintMes(RGY_LOG_ERROR, _T("Selected device #%d not found\n"), m_nDeviceId);
        return RGY_ERR_DEVICE_NOT_FOUND;
    }
    PrintMes(RGY_LOG_DEBUG, _T("InitDevice: device #%d (%s) selected.\n"), (*gpu)->id(), (*gpu)->name().c_str());
    m_dev = std::move(*gpu);
    if (inputParam->codec_rgy == RGY_CODEC_RAW || inputParam->codec_rgy == RGY_CODEC_AVCODEC) {
        PrintMes(RGY_LOG_DEBUG, _T("raw or avcodec output selected, skip initializing encoder.\n"));
        return RGY_ERR_NONE;
    }
    m_encRunCtx.reset(new NVEncRunCtx(m_dev.get(), m_pLog.get()));
    if (auto err = m_encRunCtx->init(); err != RGY_ERR_NONE) {
        PrintMes(RGY_LOG_ERROR, _T("Failed to init EncoderRunCtx error: %s\n"), get_err_mes(err));
        return err;
    }
    if (auto err = m_dev->initEncoder(); err != RGY_ERR_NONE) {
        PrintMes(RGY_LOG_ERROR, _T("Failed to init Encoder error: %s\n"), get_err_mes(err));
        return err;
    }
    return RGY_ERR_NONE;
}

NVENCSTATUS NVEncCore::ProcessOutput(const EncodeBuffer *pEncodeBuffer) {
    if (pEncodeBuffer->stOutputBfr.hBitstreamBuffer == NULL && pEncodeBuffer->stOutputBfr.bEOSFlag == FALSE) {
        return NV_ENC_ERR_INVALID_PARAM;
    }

    if (pEncodeBuffer->stOutputBfr.bWaitOnEvent) {
        if (!pEncodeBuffer->stOutputBfr.hOutputEvent) {
            return NV_ENC_ERR_INVALID_PARAM;
        }
        NVTXRANGE(ProcessOutputWait);
        WaitForSingleObject(pEncodeBuffer->stOutputBfr.hOutputEvent, INFINITE);
    }

    if (pEncodeBuffer->stOutputBfr.bEOSFlag)
        return NV_ENC_SUCCESS;

    NVTXRANGE(ProcessOutput);
    NV_ENC_LOCK_BITSTREAM lockBitstreamData;
    memset(&lockBitstreamData, 0, sizeof(lockBitstreamData));
    m_dev->encoder()->setStructVer(lockBitstreamData);
    lockBitstreamData.outputBitstream = pEncodeBuffer->stOutputBfr.hBitstreamBuffer;
    lockBitstreamData.doNotWait = false;

    NVENCSTATUS nvStatus = m_dev->encoder()->NvEncLockBitstream(&lockBitstreamData);
    if (nvStatus == NV_ENC_SUCCESS) {
        RGYBitstream bitstream = RGYBitstreamInit(lockBitstreamData);
        if (m_videoQualityMetric) {
            if (!m_videoQualityMetric->decodeStarted()) {
                m_videoQualityMetric->initDecode(&bitstream);
            }
            m_videoQualityMetric->addBitstream(&bitstream);
        }
        PrintMes(RGY_LOG_TRACE, _T("Output frame %d: size %zu (%d), pts %lld, dts %lld\n"), m_pStatus->m_sData.frameOut, bitstream.size(), lockBitstreamData.alphaLayerSizeInBytes, bitstream.pts(), bitstream.dts());
        auto outErr = m_pFileWriter->WriteNextFrame(&bitstream);
        bitstream.clear();
        nvStatus = m_dev->encoder()->NvEncUnlockBitstream(pEncodeBuffer->stOutputBfr.hBitstreamBuffer);
        if (nvStatus == NV_ENC_SUCCESS && outErr != RGY_ERR_NONE) {
            nvStatus = NV_ENC_ERR_GENERIC;
        }
    }
    return nvStatus;
}

NVENCSTATUS NVEncCore::FlushEncoder() {
    if (!m_dev->encoder()) {
        return NV_ENC_SUCCESS;
    }
    NVENCSTATUS nvStatus = m_dev->encoder()->NvEncFlushEncoderQueue(m_stEOSOutputBfr.hOutputEvent);
    if (nvStatus != NV_ENC_SUCCESS) {
        return nvStatus;
    }

    EncodeBuffer *pEncodeBufer = m_EncodeBufferQueue.GetPending();
    while (pEncodeBufer) {
        auto ret = ProcessOutput(pEncodeBufer);
        if (ret != NV_ENC_SUCCESS) {
            PrintMes(RGY_LOG_ERROR, _T("Error occurred in ProcessOutput: %d\n"), ret);
            nvStatus = ret;
        }
        pEncodeBufer = m_EncodeBufferQueue.GetPending();
    }

    if (m_stEOSOutputBfr.hOutputEvent && WaitForSingleObject(m_stEOSOutputBfr.hOutputEvent, 500) != WAIT_OBJECT_0) {
        PrintMes(RGY_LOG_ERROR, _T("m_stEOSOutputBfr.hOutputEvent%s"), (FOR_AUO) ? _T("が終了しません。") : _T(" does not finish within proper time."));
        nvStatus = NV_ENC_ERR_GENERIC;
    }

    return nvStatus;
}

RGY_ERR NVEncCore::Deinitialize() {
    m_pipelineTasks.clear();
    m_videoQualityMetric.reset();
    m_dovirpu.reset();
    m_hdr10plus.reset();
    m_hdrseiOut.reset();
    m_AudioReaders.clear();
    m_pFileReader.reset();
    m_pFileWriter.reset();
    m_pFileWriterListAudio.clear();
    m_outputFrameHostRaw.reset();
    m_encRunCtx.reset();

    if (m_dev) {
        if (m_vpFilters.size()) {
            NVEncCtxAutoLock(ctxlock(m_dev->vidCtxLock()));
            m_vpFilters.clear();
        }
        if (m_qpTable) {
            NVEncCtxAutoLock(ctxlock(m_dev->vidCtxLock()));
            m_qpTable.reset();
        }
        ReleaseIOBuffers();
    }
    m_inputHostBuffer.clear();
    m_pDecoder.reset();
    if (m_dev) {
        m_dev->close_device();
    }

    m_timecode.reset();
    m_keyFile.clear();
    m_Chapters.clear();

    m_dynamicRC.clear();
    m_videoQualityMetric.reset();
    m_pLastFilterParam.reset();
    m_deviceUsage.reset();
    m_pStatus.reset();
    m_parallelEnc.reset();
    PrintMes(RGY_LOG_DEBUG, _T("Closed EncodeStatus.\n"));

    PrintMes(RGY_LOG_DEBUG, _T("Closing perf monitor...\n"));
    m_pPerfMonitor.reset();

    PrintMes(RGY_LOG_DEBUG, _T("Closing logger...\n"));
    m_pLog.reset();
    m_pAbortByUser = nullptr;
    m_trimParam.list.clear();
    m_trimParam.offset = 0;
    //すべてのエラーをflush - 次回に影響しないように
    auto cudaerr = cudaGetLastError();
    UNREFERENCED_PARAMETER(cudaerr);
    return RGY_ERR_NONE;
}

RGY_ERR NVEncCore::AllocateBufferInputHost(const VideoInfo *pInputInfo) {
    if (m_pDecoder) {
        return RGY_ERR_NONE;
    }

    m_inputHostBuffer.resize(m_pipelineDepth);
    const int bufWidth  = pInputInfo->srcWidth  - pInputInfo->crop.e.left - pInputInfo->crop.e.right;
    const int bufHeight = pInputInfo->srcHeight - pInputInfo->crop.e.bottom - pInputInfo->crop.e.up;
    PrintMes(RGY_LOG_DEBUG, _T("Allocate Host buffers: %s %dx%d, buffer count %d\n"),
        RGY_CSP_NAMES[pInputInfo->csp], bufWidth, bufHeight, m_pipelineDepth);

    for (uint32_t i = 0; i < m_inputHostBuffer.size(); i++) {
        m_inputHostBuffer[i].cubuf = std::make_unique<CUFrameBuf>();
        m_inputHostBuffer[i].cubuf->frame.singleAlloc = true;
        m_inputHostBuffer[i].cubuf->frame.width = bufWidth;
        m_inputHostBuffer[i].cubuf->frame.height = bufHeight;
        m_inputHostBuffer[i].cubuf->frame.csp = pInputInfo->csp;
        m_inputHostBuffer[i].cubuf->frame.picstruct = pInputInfo->picstruct;
        m_inputHostBuffer[i].cubuf->frame.flags = RGY_FRAME_FLAG_NONE;
        m_inputHostBuffer[i].cubuf->frame.duration = 0;
        m_inputHostBuffer[i].cubuf->frame.timestamp = 0;
        m_inputHostBuffer[i].cubuf->frame.mem_type = RGY_MEM_TYPE_CPU;
        m_inputHostBuffer[i].heTransferFin = unique_ptr<void, handle_deleter>(CreateEvent(NULL, FALSE, TRUE, NULL), handle_deleter());

        CCtxAutoLock ctxLock(m_dev->vidCtxLock());
        auto err = m_inputHostBuffer[i].cubuf->allocHost();
        if (err != RGY_ERR_NONE) {
            PrintMes(RGY_LOG_ERROR, _T("Error CUFrameBuf::allocMemory: %s.\n"), get_err_mes(err));
            return err;
        }
    }
    return RGY_ERR_NONE;
}

RGY_ERR NVEncCore::AllocateBufferEncoder(const uint32_t uInputWidth, const uint32_t uInputHeight, const NV_ENC_BUFFER_FORMAT inputFormat, const bool alphaChannel) {
    m_stEOSOutputBfr.bEOSFlag = TRUE;
    if (!m_dev->encoder()) {
        return RGY_ERR_NONE;
    }

    m_EncodeBufferQueue.Initialize(m_stEncodeBuffer, m_encodeBufferCount);

    uint32_t uInputWidthByte = 0;
    uint32_t uInputHeightTotal = 0;
    switch (inputFormat) {
    case NV_ENC_BUFFER_FORMAT_UNDEFINED: /**< Undefined buffer format */
    case NV_ENC_BUFFER_FORMAT_YV12:      /**< Planar YUV [Y plane followed by V and U planes] */
    case NV_ENC_BUFFER_FORMAT_IYUV:      /**< Planar YUV [Y plane followed by U and V planes] */
        return RGY_ERR_UNSUPPORTED;
    case NV_ENC_BUFFER_FORMAT_YUV444:    /**< Planar YUV [Y plane followed by U and V planes] */
        uInputWidthByte = uInputWidth;
        uInputHeightTotal = uInputHeight * 3;
        break;
    case NV_ENC_BUFFER_FORMAT_YUV420_10BIT: /**< 10 bit Semi-Planar YUV [Y plane followed by interleaved UV plane]. Each pixel of size 2 bytes. Most Significant 10 bits contain pixel data. */
        uInputWidthByte = uInputWidth * 2;
        uInputHeightTotal = uInputHeight * 3 / 2;
        break;
    case NV_ENC_BUFFER_FORMAT_YUV444_10BIT: /**< 10 bit Planar YUV444 [Y plane followed by U and V planes]. Each pixel of size 2 bytes. Most Significant 10 bits contain pixel data.  */
        uInputWidthByte = uInputWidth * 2;
        uInputHeightTotal = uInputHeight * 3;
        break;
    case NV_ENC_BUFFER_FORMAT_ARGB:    /**< 8 bit Packed A8R8G8B8 */
    case NV_ENC_BUFFER_FORMAT_ARGB10:  /**< 10 bit Packed A2R10G10B10. Each pixel of size 2 bytes. Most Significant 10 bits contain pixel data.  */
    case NV_ENC_BUFFER_FORMAT_AYUV:    /**< 8 bit Packed A8Y8U8V8 */
    case NV_ENC_BUFFER_FORMAT_ABGR:    /**< 8 bit Packed A8B8G8R8 */
    case NV_ENC_BUFFER_FORMAT_ABGR10:  /**< 10 bit Packed A2B10G10R10. Each pixel of size 2 bytes. Most Significant 10 bits contain pixel data.  */
        return RGY_ERR_UNSUPPORTED;
    case NV_ENC_BUFFER_FORMAT_NV12:    /**< Semi-Planar YUV [Y plane followed by interleaved UV plane] */
        uInputWidthByte = uInputWidth;
        uInputHeightTotal = uInputHeight * 3 / 2;
        break;
    default:
        return RGY_ERR_UNSUPPORTED;
    }
    PrintMes(RGY_LOG_DEBUG, _T("AllocateIOBuffers: %s %dx%d (width byte %d, height total %d), buffer count %d\n"),
        RGY_CSP_NAMES[csp_enc_to_rgy(inputFormat)], uInputWidth, uInputHeight, uInputWidthByte, uInputHeightTotal, m_encodeBufferCount);

    for (int i = 0; i < m_encodeBufferCount; i++) {
        if (m_stPicStruct == NV_ENC_PIC_STRUCT_FRAME) {
            cuvidCtxLock(m_dev->vidCtxLock(), 0);
            const auto allocHeight = uInputHeightTotal * (alphaChannel ? 2 : 1);
            auto cudaerr = cudaMallocPitch((void **)&m_stEncodeBuffer[i].stInputBfr.pNV12devPtr,
                (size_t *)&m_stEncodeBuffer[i].stInputBfr.uNV12Stride, uInputWidthByte, allocHeight);
            //初期化
            auto sts = err_to_rgy(cudaMemset2D((void *)m_stEncodeBuffer[i].stInputBfr.pNV12devPtr, m_stEncodeBuffer[i].stInputBfr.uNV12Stride, -128, uInputWidthByte, allocHeight));
            if (sts != RGY_ERR_NONE) {
                PrintMes(RGY_LOG_ERROR, _T("Failed to init alpha buffer: %s\n"), get_err_mes(sts));
                return sts;
            }
            cuvidCtxUnlock(m_dev->vidCtxLock(), 0);
            if (cudaerr != cudaSuccess) {
                PrintMes(RGY_LOG_ERROR, _T("Failed to cuMemAllocPitch, %d (%s)\n"), cudaerr, char_to_tstring(_cudaGetErrorEnum(cudaerr)).c_str());
                return err_to_rgy(cudaerr);
            }

            auto nvStatus = m_dev->encoder()->NvEncRegisterResource(NV_ENC_INPUT_RESOURCE_TYPE_CUDADEVICEPTR,
                (void*)m_stEncodeBuffer[i].stInputBfr.pNV12devPtr,
                uInputWidth, uInputHeight, m_stEncodeBuffer[i].stInputBfr.uNV12Stride, inputFormat,
                &m_stEncodeBuffer[i].stInputBfr.nvRegisteredResource);
            if (nvStatus != NV_ENC_SUCCESS) {
                PrintMes(RGY_LOG_ERROR, _T("Failed to register input device memory.\n"));
                return err_to_rgy(nvStatus);
            }
            // alpha channelが必要な場合、メモリ確保は連続で行い、NvEncRegisterResourceを分割する
            if (alphaChannel) {
                m_stEncodeBuffer[i].stInputBfrAlpha.pNV12devPtr = (CUdeviceptr)nullptr;
                m_stEncodeBuffer[i].stInputBfrAlpha.dwHeight = uInputWidth;
                m_stEncodeBuffer[i].stInputBfrAlpha.dwHeight = uInputHeight;
                m_stEncodeBuffer[i].stInputBfrAlpha.bufferFmt = inputFormat;
                m_stEncodeBuffer[i].stInputBfrAlpha.uNV12Stride = m_stEncodeBuffer[i].stInputBfr.uNV12Stride;
                uint8_t *ptr = (uint8_t *)m_stEncodeBuffer[i].stInputBfr.pNV12devPtr;
                ptr += m_stEncodeBuffer[i].stInputBfr.uNV12Stride * uInputHeightTotal;
                nvStatus = m_dev->encoder()->NvEncRegisterResource(NV_ENC_INPUT_RESOURCE_TYPE_CUDADEVICEPTR,
                    (void*)ptr, uInputWidth, uInputHeight, m_stEncodeBuffer[i].stInputBfrAlpha.uNV12Stride, inputFormat,
                    &m_stEncodeBuffer[i].stInputBfrAlpha.nvRegisteredResource);
                if (nvStatus != NV_ENC_SUCCESS) {
                    PrintMes(RGY_LOG_ERROR, _T("Failed to register input device memory.\n"));
                    return err_to_rgy(nvStatus);
                }
            }
        } else {
            //インタレ保持の場合は、NvEncCreateInputBuffer経由でフレームを渡さないと正常にエンコードできない
            if (alphaChannel) {
                PrintMes(RGY_LOG_ERROR, _T("alpha channel encoding not supported with interlaced encoding.\n"));
                return RGY_ERR_UNSUPPORTED;
            }
            auto nvStatus = m_dev->encoder()->NvEncCreateInputBuffer(uInputWidth, uInputHeight, &m_stEncodeBuffer[i].stInputBfr.hInputSurface, inputFormat);
            if (nvStatus != NV_ENC_SUCCESS) {
                PrintMes(RGY_LOG_ERROR, _T("Failed to allocate Input Buffer, Please reduce MAX_FRAMES_TO_PRELOAD\n"));
                return err_to_rgy(nvStatus);
            }
        }

        m_stEncodeBuffer[i].stInputBfr.bufferFmt = inputFormat;
        m_stEncodeBuffer[i].stInputBfr.dwWidth = uInputWidth;
        m_stEncodeBuffer[i].stInputBfr.dwHeight = uInputHeight;

        auto nvStatus = m_dev->encoder()->NvEncCreateBitstreamBuffer(BITSTREAM_BUFFER_SIZE, &m_stEncodeBuffer[i].stOutputBfr.hBitstreamBuffer);
        if (nvStatus != NV_ENC_SUCCESS) {
            PrintMes(RGY_LOG_ERROR, _T("Failed to allocate Output Buffer, Please reduce MAX_FRAMES_TO_PRELOAD\n"));
            return err_to_rgy(nvStatus);
        }
        m_stEncodeBuffer[i].stOutputBfr.dwBitstreamBufferSize = BITSTREAM_BUFFER_SIZE;

        nvStatus = m_dev->encoder()->NvEncRegisterAsyncEvent(&m_stEncodeBuffer[i].stOutputBfr.hOutputEvent);
        if (nvStatus != NV_ENC_SUCCESS) {
            return err_to_rgy(nvStatus);
        }
        m_stEncodeBuffer[i].stOutputBfr.bWaitOnEvent = ENABLE_ASYNC != 0;

        nvStatus = m_dev->encoder()->NvEncRegisterAsyncEvent(&m_stEOSOutputBfr.hOutputEvent);
        if (nvStatus != NV_ENC_SUCCESS) {
            return err_to_rgy(nvStatus);
        }
    }
    return RGY_ERR_NONE;
}

RGY_ERR NVEncCore::AllocateBufferRawOutput(const uint32_t uInputWidth, const uint32_t uInputHeight, const RGY_CSP csp) {
    if (m_dev->encoder()) {
        return RGY_ERR_NONE;
    }
    m_encodeBufferCount = 0;

    RGYFrameInfo outFrame;
    outFrame.csp = csp;
    outFrame.width = uInputWidth;
    outFrame.height = uInputHeight;
    outFrame.mem_type = RGY_MEM_TYPE_CPU;
    m_outputFrameHostRaw = std::make_unique<CUFrameBuf>(outFrame);
    if (auto sts = m_outputFrameHostRaw->allocHost(); sts != RGY_ERR_NONE) {
        PrintMes(RGY_LOG_ERROR, _T("Failed to allocate raw output buffer: %s\n"), get_err_mes(sts));
        return sts;
    }
    return RGY_ERR_NONE;
}

NVENCSTATUS NVEncCore::ReleaseIOBuffers() {
    for (int i = 0; i < m_encodeBufferCount; i++) {
        if (m_stEncodeBuffer[i].stInputBfr.pNV12devPtr) {
#if ENABLE_AVSW_READER
            cuvidCtxLock(m_dev->vidCtxLock(), 0);
#endif //#if ENABLE_AVSW_READER
            cuMemFree(m_stEncodeBuffer[i].stInputBfr.pNV12devPtr);
#if ENABLE_AVSW_READER
            cuvidCtxUnlock(m_dev->vidCtxLock(), 0);
#endif //#if ENABLE_AVSW_READER
            m_stEncodeBuffer[i].stInputBfr.pNV12devPtr = 0;
        } else {
            //インタレ保持の場合にはこちらを使用
            if (m_stEncodeBuffer[i].stInputBfr.hInputSurface) {
                m_dev->encoder()->NvEncDestroyInputBuffer(m_stEncodeBuffer[i].stInputBfr.hInputSurface);
                m_stEncodeBuffer[i].stInputBfr.hInputSurface = NULL;
            }
        }

        if (m_stEncodeBuffer[i].stOutputBfr.hBitstreamBuffer) {
            m_dev->encoder()->NvEncDestroyBitstreamBuffer(m_stEncodeBuffer[i].stOutputBfr.hBitstreamBuffer);
            m_stEncodeBuffer[i].stOutputBfr.hBitstreamBuffer = NULL;
        }
        if (m_stEncodeBuffer[i].stOutputBfr.hOutputEvent) {
            m_dev->encoder()->NvEncUnregisterAsyncEvent(m_stEncodeBuffer[i].stOutputBfr.hOutputEvent);
            CloseEvent(m_stEncodeBuffer[i].stOutputBfr.hOutputEvent);
            m_stEncodeBuffer[i].stOutputBfr.hOutputEvent = NULL;
        }
    }

    if (m_stEOSOutputBfr.hOutputEvent) {
        m_dev->encoder()->NvEncUnregisterAsyncEvent(m_stEOSOutputBfr.hOutputEvent);
        CloseEvent(m_stEOSOutputBfr.hOutputEvent);
        m_stEOSOutputBfr.hOutputEvent = NULL;
    }
    PrintMes(RGY_LOG_DEBUG, _T("Released IO Buffers.\n"));
    return NV_ENC_SUCCESS;
}

bool NVEncCore::enableCuvidResize(const InEncodeVideoParam *inputParam) const {
    const bool interlacedEncode = ((inputParam->input.picstruct & RGY_PICSTRUCT_INTERLACED)
        && (inputParam->vppnv.deinterlace == cudaVideoDeinterlaceMode_Weave
            && !inputParam->vpp.afs.enable
            && !inputParam->vpp.nnedi.enable
            && !inputParam->vpp.yadif.enable));
    return
         //デフォルトの補間方法
        inputParam->vpp.resize_algo == RGY_VPP_RESIZE_AUTO
        //deinterlace bobとリサイズを有効にすると色成分が正常に出力されない場合がある
        //deinterlace normalとリサイズを有効にすると輝度成分も含めて正常に出力されない場合がある
        && inputParam->vppnv.deinterlace == cudaVideoDeinterlaceMode_Weave
#if CUVID_DISABLE_CROP
        //cropが行われていない (cuvidのcropはよくわからん)
        && !cropEnabled(inputParam->input.crop)
#endif //#if CUVID_DISABLE_CROP
        //インタレ保持でない (インタレ保持リサイズをするにはCUDAで行う必要がある)
        && !interlacedEncode
        //フィルタ処理が必要
        && !(  inputParam->vpp.delogo.enable
            || inputParam->vppnv.gaussMaskSize > 0
            || inputParam->vpp.unsharp.enable
            || inputParam->vpp.convolution3d.enable
            || inputParam->vpp.knn.enable
            || inputParam->vpp.nlmeans.enable
            || inputParam->vpp.pmd.enable
            || inputParam->vpp.dct.enable
            || inputParam->vpp.smooth.enable
            || inputParam->vpp.fft3d.enable
            || inputParam->vppnv.nvvfxDenoise.enable
            || inputParam->vppnv.nvvfxArtifactReduction.enable
            || inputParam->vpp.deband.enable
            || inputParam->vpp.libplacebo_deband.enable
            || inputParam->vpp.edgelevel.enable
            || inputParam->vpp.warpsharp.enable
            || inputParam->vpp.afs.enable
            || inputParam->vpp.nnedi.enable
            || inputParam->vpp.yadif.enable
            || inputParam->vpp.decomb.enable
            || inputParam->vpp.tweak.enable
            || inputParam->vpp.curves.enable
            || inputParam->vpp.transform.enable
            || inputParam->vpp.colorspace.enable
            || inputParam->vpp.libplacebo_tonemapping.enable
            || inputParam->vpp.subburn.size() > 0
            || inputParam->vpp.libplacebo_shader.size() > 0
            || inputParam->vpp.pad.enable
            || inputParam->vpp.selectevery.enable
            || inputParam->vpp.decimate.enable
            || inputParam->vpp.mpdecimate.enable
            || inputParam->vpp.overlay.size() > 0
            || inputParam->vppnv.ngxTrueHDR.enable
            || inputParam->vpp.fruc.enable);
}

#pragma warning(push)
#pragma warning(disable: 4100)
RGY_ERR NVEncCore::InitDecoder(const InEncodeVideoParam *inputParam) {
    if (m_pFileReader->getInputCodec() != RGY_CODEC_UNKNOWN) {
        const AVStream *streamIn = nullptr;
        RGYInputAvcodec *pReader = dynamic_cast<RGYInputAvcodec *>(m_pFileReader.get());
        if (pReader != nullptr) {
            streamIn = pReader->GetInputVideoStream();
        }
        if (streamIn == nullptr) {
            PrintMes(RGY_LOG_ERROR, _T("failed to get stream info when initializing cuvid decoder.\n"));
            return RGY_ERR_UNSUPPORTED;
        }

        m_pDecoder.reset(new CuvidDecode());

        auto result = m_pDecoder->InitDecode(m_dev->vidCtxLock(), &inputParam->input, &inputParam->vppnv, streamIn->time_base, m_pLog, inputParam->nHWDecType, enableCuvidResize(inputParam), inputParam->ctrl.lowLatency);
        if (result != CUDA_SUCCESS) {
            PrintMes(RGY_LOG_ERROR, _T("failed to init decoder.\n"));
            return RGY_ERR_UNSUPPORTED;
        }
    }
    return RGY_ERR_NONE;
}
#pragma warning(pop)

RGY_ERR NVEncCore::SetInputParam(InEncodeVideoParam *inputParam) {
    //解像度の決定
    //この段階では、フィルタを使用した場合は解像度を変更しないものとする
    m_uEncWidth  = (m_pLastFilterParam) ? m_pLastFilterParam->frameOut.width  : inputParam->input.srcWidth  - inputParam->input.crop.e.left - inputParam->input.crop.e.right;
    m_uEncHeight = (m_pLastFilterParam) ? m_pLastFilterParam->frameOut.height : inputParam->input.srcHeight - inputParam->input.crop.e.bottom - inputParam->input.crop.e.up;

    //この段階では、フィルタを使用した場合は解像度を変更しないものとする
    if (!m_pLastFilterParam) {
        if (inputParam->input.dstWidth && inputParam->input.dstHeight) {
            if (m_pFileReader->getInputCodec() != RGY_CODEC_UNKNOWN) {
                m_uEncWidth  = inputParam->input.dstWidth;
                m_uEncHeight = inputParam->input.dstHeight;
            } else if (m_uEncWidth != inputParam->input.srcWidth || m_uEncHeight != inputParam->input.srcHeight) {
                PrintMes(RGY_LOG_ERROR, _T("resizing requires to be used with avhw reader.\n"));
                PrintMes(RGY_LOG_ERROR, _T(" input %dx%d -> output %dx%d.\n"), m_uEncWidth, m_uEncHeight, inputParam->input.dstWidth, inputParam->input.dstHeight);
                return RGY_ERR_UNSUPPORTED;
            }
        }
    }

    //制限事項チェック
    if (inputParam->input.srcWidth < 0 && inputParam->input.srcHeight < 0) {
        PrintMes(RGY_LOG_ERROR, _T("%s: %dx%d\n"), FOR_AUO ? _T("解像度が無効です。") : _T("Invalid resolution."), inputParam->input.srcWidth, inputParam->input.srcHeight);
        return RGY_ERR_UNSUPPORTED;
    }
    if (   (int)inputParam->input.srcWidth  <= inputParam->input.crop.e.left + inputParam->input.crop.e.right
        && (int)inputParam->input.srcHeight <= inputParam->input.crop.e.up   + inputParam->input.crop.e.bottom) {
        PrintMes(RGY_LOG_ERROR, _T("%s: %dx%d, Crop [%d,%d,%d,%d]\n"),
             FOR_AUO ? _T("Crop値が無効です。") : _T("Invalid crop value."),
            inputParam->input.srcWidth, inputParam->input.srcHeight,
            inputParam->input.crop.c[0], inputParam->input.crop.c[1], inputParam->input.crop.c[2], inputParam->input.crop.c[3]);
        return RGY_ERR_UNSUPPORTED;
    }

    const int height_check_mask = 1 + 2 * !!is_interlaced(m_stPicStruct);
    if ((m_uEncWidth & 1) || (m_uEncHeight & height_check_mask)) {
        PrintMes(RGY_LOG_ERROR, _T("%s: %dx%d\n"), FOR_AUO ? _T("解像度が無効です。") : _T("Invalid resolution."), m_uEncWidth, m_uEncHeight);
        PrintMes(RGY_LOG_ERROR, FOR_AUO ? _T("縦横の解像度は2の倍数である必要があります。\n") : _T("Relosution of mod2 required.\n"));
        if (is_interlaced(m_stPicStruct)) {
            PrintMes(RGY_LOG_ERROR, FOR_AUO ? _T("さらに、インタレ保持エンコードでは縦解像度は4の倍数である必要があります。\n") : _T("For interlaced encoding, mod4 is required for height.\n"));
        }
        return RGY_ERR_UNSUPPORTED;
    }
    if ((inputParam->input.crop.e.left & 1) || (inputParam->input.crop.e.right & 1)
        || (inputParam->input.crop.e.up & height_check_mask) || (inputParam->input.crop.e.bottom & height_check_mask)) {
        PrintMes(RGY_LOG_ERROR, _T("%s: %dx%d, Crop [%d,%d,%d,%d]\n"),
             FOR_AUO ? _T("Crop値が無効です。") : _T("Invalid crop value."),
            inputParam->input.srcWidth, inputParam->input.srcHeight,
            inputParam->input.crop.c[0], inputParam->input.crop.c[1], inputParam->input.crop.c[2], inputParam->input.crop.c[3]);
        PrintMes(RGY_LOG_ERROR, FOR_AUO ? _T("Crop値は2の倍数である必要があります。\n") : _T("Crop value of mod2 required.\n"));
        if (is_interlaced(m_stPicStruct)) {
            PrintMes(RGY_LOG_ERROR, FOR_AUO ? _T("さらに、インタレ保持エンコードでは縦Crop値は4の倍数である必要があります。\n") : _T("For interlaced encoding, mod4 is required for height.\n"));
        }
        return RGY_ERR_UNSUPPORTED;
    }

    //SAR自動設定
    auto par = std::make_pair(inputParam->par[0], inputParam->par[1]);
    if ((!inputParam->par[0] || !inputParam->par[1]) //SAR比の指定がない
        && inputParam->input.sar[0] && inputParam->input.sar[1] //入力側からSAR比を取得ずみ
        && (m_uEncWidth == inputParam->input.srcWidth && m_uEncHeight == inputParam->input.srcHeight)) {//リサイズは行われない
        par = std::make_pair(inputParam->input.sar[0], inputParam->input.sar[1]);
    }
    adjust_sar(&par.first, &par.second, m_uEncWidth, m_uEncHeight);
    m_sar = rgy_rational<int>(par.first, par.second);

    if (m_encFps.n() <= 0 || m_encFps.d() <= 0) {
        PrintMes(RGY_LOG_ERROR, _T("Invalid fps: %d/%d.\n"), m_encFps.n(), m_encFps.d());
        return RGY_ERR_UNSUPPORTED;
    }

    //picStructの設定
    m_stPicStruct = picstruct_rgy_to_enc(inputParam->input.picstruct);

    if (inputParam->vppnv.deinterlace != cudaVideoDeinterlaceMode_Weave) {
        if (m_pFileReader->getInputCodec() == RGY_CODEC_UNKNOWN) {
            PrintMes(RGY_LOG_ERROR, _T("vpp-deinterlace requires to be used with avhw reader.\n"));
            return RGY_ERR_UNSUPPORTED;
        }
        m_stPicStruct = NV_ENC_PIC_STRUCT_FRAME;
    } else if (inputParam->vpp.afs.enable || inputParam->vpp.nnedi.enable || inputParam->vpp.yadif.enable || inputParam->vpp.decomb.enable) {
        m_stPicStruct = NV_ENC_PIC_STRUCT_FRAME;
    }

    if (inputParam->codec_rgy == RGY_CODEC_RAW || inputParam->codec_rgy == RGY_CODEC_AVCODEC) {
        PrintMes(RGY_LOG_DEBUG, _T("raw or avcodec output selected, skip initializing encoder.\n"));
        return RGY_ERR_NONE;
    }

    m_dev->encoder()->setStructVer(m_stCreateEncodeParams);
    m_dev->encoder()->setStructVer(m_stEncConfig);
    m_dev->encoder()->setStructVer(m_stEncConfig.rcParams);

    //コーデックの決定とチェックNV_ENC_PIC_PARAMS
    if (inputParam->codec_rgy == RGY_CODEC_UNKNOWN) {
        PrintMes(RGY_LOG_ERROR, _T("Unknown codec.\n"));
        return RGY_ERR_UNSUPPORTED;
    }
    if (inputParam->codec_rgy == RGY_CODEC_AV1 && !m_dev->encoder()->checkAPIver(12, 0)) {
        PrintMes(RGY_LOG_ERROR, _T("Selected codec %s requires NVENC API v12.0 or later.\n"), CodecToStr(inputParam->codec_rgy).c_str());
        return RGY_ERR_UNSUPPORTED;
    }

    m_stCodecGUID = codec_guid_rgy_to_enc(inputParam->codec_rgy);
    auto codecFeature = m_dev->encoder()->getCodecFeature(m_stCodecGUID);
    if (codecFeature == nullptr) {
        PrintMes(RGY_LOG_ERROR, _T("Selected codec (%s) is not supported.\n"), CodecToStr(inputParam->codec_rgy).c_str());
        return RGY_ERR_UNSUPPORTED;
    }

    { // プロファイルの反映
        m_stEncConfig.profileGUID = NV_ENC_CODEC_PROFILE_AUTOSELECT_GUID;
        if (inputParam->codec_rgy == RGY_CODEC_H264) {
            m_stEncConfig.profileGUID = get_guid_from_value(inputParam->h264.profile, h264_profile_names);
        } else if (inputParam->codec_rgy == RGY_CODEC_HEVC) {
            m_stEncConfig.profileGUID = get_guid_from_value(inputParam->hevc.profile, h265_profile_names);
        } else if (inputParam->codec_rgy == RGY_CODEC_AV1) {
            m_stEncConfig.profileGUID = get_guid_from_value(inputParam->av1.profile, av1_profile_names);
        } else {
            PrintMes(RGY_LOG_ERROR, _T("Unknown codec.\n"));
            return RGY_ERR_UNSUPPORTED;
        }
    }
    if (!codecFeature->checkProfileSupported(m_stEncConfig.profileGUID)) {
        m_stEncConfig.profileGUID = NV_ENC_CODEC_PROFILE_AUTOSELECT_GUID;
        PrintMes(RGY_LOG_WARN, _T("Selected profile is not supported, profile will be auto selected by NVENC API!\n"), get_name_from_guid(m_stEncConfig.profileGUID, get_codec_profile_list(inputParam->codec_rgy)));
    }

    { //プリセットのチェック
        GUID presetGUID;
        if (m_dev->encoder()->checkAPIver(10, 0)) {
            presetGUID = get_guid_from_value(inputParam->preset, list_nvenc_preset_names_ver10);
        } else {
            presetGUID = get_guid_from_value(inputParam->preset, list_nvenc_preset_names_ver9_2);
        }
        if (!codecFeature->checkPresetSupported(presetGUID)) {
            PrintMes(RGY_LOG_ERROR, FOR_AUO ? _T("指定されたプリセットはサポートされていません。\n") : _T("Selected preset is not supported.\n"));
            return RGY_ERR_UNSUPPORTED;
        }
    }

    //QP上限のチェック
    const int qpMaxCodec = (inputParam->codec_rgy == RGY_CODEC_AV1) ? 255 : (encodeIsHighBitDepth(inputParam) ? 63 : 51);
    inputParam->rcParam.qp.applyQPMinMax(0, qpMaxCodec);
    inputParam->qpInit.applyQPMinMax(0, qpMaxCodec);
    inputParam->qpMin.applyQPMinMax(0, qpMaxCodec);
    inputParam->qpMax.applyQPMinMax(0, qpMaxCodec);

    //入力フォーマットはここでは気にしない
    //NV_ENC_BUFFER_FORMAT_NV12_TILED64x16
    //if (!checkSurfaceFmtSupported(NV_ENC_BUFFER_FORMAT_NV12_TILED64x16)) {
    //    PrintMes(RGY_LOG_ERROR, FOR_AUO ? _T("入力フォーマットが決定できません。\n") : _T("Input format is not supported.\n"));
    //    return NV_ENC_ERR_UNSUPPORTED_PARAM;
    //}

    static const auto VBR_RC_LIST = make_array<NV_ENC_PARAMS_RC_MODE>(NV_ENC_PARAMS_RC_VBR, NV_ENC_PARAMS_RC_VBR_MINQP, NV_ENC_PARAMS_RC_2_PASS_VBR, NV_ENC_PARAMS_RC_CBR, NV_ENC_PARAMS_RC_CBR2, NV_ENC_PARAMS_RC_CBR_HQ, NV_ENC_PARAMS_RC_VBR_HQ);

    if ((inputParam->common.AVSyncMode & RGY_AVSYNC_FORCE_CFR) != 0 && inputParam->common.nTrimCount > 0) {
        PrintMes(RGY_LOG_ERROR, _T("avsync forcecfr + trim is not supported.\n"));
        return RGY_ERR_UNSUPPORTED;
    }
    //環境による制限
    auto error_resolution_over_limit = [&](const TCHAR *feature, uint32_t featureValue, NV_ENC_CAPS featureID) {
        const TCHAR *error_mes = FOR_AUO ? _T("解像度が上限を超えています。") : _T("Resolution is over limit.");
        if (nullptr == feature)
            PrintMes(RGY_LOG_ERROR, _T("%s: %dx%d [上限: %dx%d]\n"), error_mes, m_uEncWidth, m_uEncHeight, codecFeature->getCapLimit(NV_ENC_CAPS_WIDTH_MAX), codecFeature->getCapLimit(NV_ENC_CAPS_HEIGHT_MAX));
        else
            PrintMes(RGY_LOG_ERROR, _T("%s: %dx%d, [%s]: %d [上限: %d]\n"), error_mes, m_uEncWidth, m_uEncHeight, feature, featureValue, codecFeature->getCapLimit(featureID));
    };

    if (m_uEncWidth > codecFeature->getCapLimit(NV_ENC_CAPS_WIDTH_MAX) || m_uEncHeight > codecFeature->getCapLimit(NV_ENC_CAPS_HEIGHT_MAX)) {
        error_resolution_over_limit(nullptr, 0, (NV_ENC_CAPS)0);
        return RGY_ERR_UNSUPPORTED;
    }
    uint32_t heightMod = 16 * (1 + !!is_interlaced(m_stPicStruct));
    uint32_t targetMB = ((m_uEncWidth + 15) / 16) * ((m_uEncHeight + (heightMod - 1)) / heightMod);
    if (targetMB > (uint32_t)codecFeature->getCapLimit(NV_ENC_CAPS_MB_NUM_MAX)) {
        error_resolution_over_limit(_T("MB"), targetMB, NV_ENC_CAPS_MB_NUM_MAX);
        return RGY_ERR_UNSUPPORTED;
    }
    //MB/sの制限は特にチェックする必要がなさそう
    //uint32_t targetMBperSec = (targetMB * inputParam->input.rate + inputParam->input.scale - 1) / inputParam->input.scale;
    //if (targetMBperSec > (uint32_t)getCapLimit(NV_ENC_CAPS_MB_PER_SEC_MAX)) {
    //    error_resolution_over_limit(_T("MB/s"), targetMBperSec, NV_ENC_CAPS_MB_PER_SEC_MAX);
    //    return NV_ENC_ERR_UNSUPPORTED_PARAM;
    //}

    // tuneのチェック
    if (ENABLE_NVENC_SDK_TUNE) {
        inputParam->tuningInfo = (inputParam->tuningInfo == NV_ENC_TUNING_INFO_UNDEFINED) ? NV_ENC_TUNING_INFO_HIGH_QUALITY : inputParam->tuningInfo;
        if (inputParam->tuningInfo == NV_ENC_TUNING_INFO_ULTRA_HIGH_QUALITY) {
            if (!m_dev->encoder()->checkAPIver(12, 2)) {
                PrintMes(RGY_LOG_WARN, _T("tune uhq disabled as it requires NVENC API 12.2.\n"));
                inputParam->tuningInfo = NV_ENC_TUNING_INFO_HIGH_QUALITY;
            }
            if (inputParam->codec_rgy == RGY_CODEC_H264) {
                PrintMes(RGY_LOG_WARN, _T("tune uhq disabled as it is not supported with H.264 encoding.\n"));
                inputParam->tuningInfo = NV_ENC_TUNING_INFO_HIGH_QUALITY;
            }
            if (m_dev->cc().first <= 6
                || (m_dev->cc().first == 7 && m_dev->cc().second < 5)) {
                PrintMes(RGY_LOG_WARN, _T("tune uhq disabled as it requires GPUs Turing or above.\n"));
                inputParam->tuningInfo = NV_ENC_TUNING_INFO_HIGH_QUALITY;
            }
        }
    } else {
        inputParam->tuningInfo = NV_ENC_TUNING_INFO_HIGH_QUALITY;
    }

    if (inputParam->ctrl.lowLatency && m_dev->encoder()->checkAPIver(10, 0)) {
        inputParam->tuningInfo = NV_ENC_TUNING_INFO_ULTRA_LOW_LATENCY;
    }

    { // プリセットのデフォルトのロード
        const auto profileGUID = m_stEncConfig.profileGUID;
        GUID presetGUID;
        if (m_dev->encoder()->checkAPIver(10, 0)) {
            presetGUID = get_guid_from_value(inputParam->preset, list_nvenc_preset_names_ver10);
        } else {
            presetGUID = get_guid_from_value(inputParam->preset, list_nvenc_preset_names_ver9_2);
        }
        NV_ENC_PRESET_CONFIG presetConfig;
        auto nvStatus = m_dev->encoder()->getPresetDefaultParams(m_stCodecGUID, profileGUID, presetGUID, inputParam->tuningInfo, presetConfig);
        if (nvStatus != NV_ENC_SUCCESS) {
            tstring presetName;
            if (m_dev->encoder()->checkAPIver(10, 0)) {
                presetName = get_name_from_guid(presetGUID, list_nvenc_preset_names_ver10);
            } else {
                presetName = get_name_from_guid(presetGUID, list_nvenc_preset_names_ver9_2);
            }
            PrintMes(RGY_LOG_WARN, _T("Failed to get preset default params for %s, %s, %s, tune %s: %s.\n"),
                CodecToStr(inputParam->codec_rgy).c_str(),
                get_name_from_guid(profileGUID, get_codec_profile_list(inputParam->codec_rgy)),
                presetName.c_str(),
                get_cx_desc(list_tuning_info, inputParam->tuningInfo),
                get_err_mes(err_to_rgy(nvStatus)));
            PrintMes(RGY_LOG_WARN, _T("Using application default params instead.\n"));
            if (inputParam->codec_rgy == RGY_CODEC_H264) {
                presetConfig.presetCfg.encodeCodecConfig = DefaultParamH264();
            } else if (inputParam->codec_rgy == RGY_CODEC_HEVC) {
                presetConfig.presetCfg.encodeCodecConfig = DefaultParamHEVC();
            } else if (inputParam->codec_rgy == RGY_CODEC_AV1) {
                presetConfig.presetCfg.encodeCodecConfig = DefaultParamAV1();
            } else {
                PrintMes(RGY_LOG_ERROR, _T("Unknown codec.\n"));
                return RGY_ERR_UNSUPPORTED;
            }
        }
        m_stEncConfig = presetConfig.presetCfg;
        m_stEncConfig.profileGUID = profileGUID;
    }

    { // レベル/ティアの反映
        if (inputParam->codec_rgy == RGY_CODEC_H264) {
            set_level(m_stEncConfig.encodeCodecConfig, inputParam->codec_rgy, inputParam->h264.level);
        } else if (inputParam->codec_rgy == RGY_CODEC_HEVC) {
            set_level(m_stEncConfig.encodeCodecConfig, inputParam->codec_rgy, inputParam->hevc.level);
            set_tier(m_stEncConfig.encodeCodecConfig, inputParam->codec_rgy, inputParam->hevc.tier);
        } else if (inputParam->codec_rgy == RGY_CODEC_AV1) {
            set_level(m_stEncConfig.encodeCodecConfig, inputParam->codec_rgy, inputParam->av1.level);
            set_tier(m_stEncConfig.encodeCodecConfig, inputParam->codec_rgy, inputParam->av1.tier);
        } else {
            PrintMes(RGY_LOG_ERROR, _T("Unknown codec.\n"));
            return RGY_ERR_UNSUPPORTED;
        }
    }

    auto error_feature_unsupported = [&](RGYLogLevel log_level, const TCHAR *feature_name) {
        PrintMes(log_level, FOR_AUO ? _T("%sはサポートされていません。\n") : _T("%s unsupported.\n"), feature_name);
    };

    if (is_interlaced(m_stPicStruct) && !codecFeature->getCapLimit(NV_ENC_CAPS_SUPPORT_FIELD_ENCODING)) {
        if (inputParam->codec_rgy != RGY_CODEC_H264) {
            PrintMes(RGY_LOG_ERROR, _T("interlaced output is only supported for H.264 codec.\n"));
        } else {
            PrintMes(RGY_LOG_ERROR, _T("interlaced output is not supported for current setting.\n"));
        }
        return RGY_ERR_UNSUPPORTED;
    }
    // レート制御の設定
    m_stEncConfig.rcParams.rateControlMode = inputParam->rcParam.rc_mode;
    m_stEncConfig.rcParams.multiPass       = inputParam->multipass;
    // API v10.0で追加されたmultipass関係の互換性維持
    if (m_dev->encoder()->checkAPIver(10, 0)) {
        if (m_stEncConfig.rcParams.rateControlMode == NV_ENC_PARAMS_RC_CBR_HQ) {
            m_stEncConfig.rcParams.rateControlMode = NV_ENC_PARAMS_RC_CBR;
            m_stEncConfig.rcParams.multiPass = NV_ENC_TWO_PASS_FULL_RESOLUTION;
        } else if (m_stEncConfig.rcParams.rateControlMode == NV_ENC_PARAMS_RC_VBR_HQ) {
            m_stEncConfig.rcParams.rateControlMode = NV_ENC_PARAMS_RC_VBR;
            m_stEncConfig.rcParams.multiPass = NV_ENC_TWO_PASS_FULL_RESOLUTION;
        }
    } else {
        if (m_stEncConfig.rcParams.multiPass != NV_ENC_MULTI_PASS_DISABLED) {
            m_stEncConfig.rcParams.multiPass = NV_ENC_MULTI_PASS_DISABLED;
            if (m_stEncConfig.rcParams.rateControlMode == NV_ENC_PARAMS_RC_CBR) {
                m_stEncConfig.rcParams.rateControlMode = NV_ENC_PARAMS_RC_CBR_HQ;
            } else if (m_stEncConfig.rcParams.rateControlMode == NV_ENC_PARAMS_RC_VBR) {
                m_stEncConfig.rcParams.rateControlMode = NV_ENC_PARAMS_RC_VBR_HQ;
            }
        }
    }
    // QVBR指定で値が設定されていないときは、DEFAULT_QVBR_TARGETを適用する
    if (m_stEncConfig.rcParams.rateControlMode == NV_ENC_PARAMS_RC_QVBR
        && (inputParam->rcParam.targetQuality < 0 || inputParam->rcParam.targetQualityLSB < 0)) {
        inputParam->rcParam.targetQuality = DEFAULT_QVBR_TARGET;
        inputParam->rcParam.targetQualityLSB = 0;
    }

    //その他のレート制御パラメータの設定
    m_stEncConfig.rcParams.averageBitRate    = inputParam->rcParam.avg_bitrate;
    m_stEncConfig.rcParams.maxBitRate        = inputParam->rcParam.max_bitrate;
    m_stEncConfig.rcParams.targetQuality     = (uint8_t)std::max(inputParam->rcParam.targetQuality, 0);
    m_stEncConfig.rcParams.targetQualityLSB  = (uint8_t)std::max(inputParam->rcParam.targetQualityLSB, 0);
    setQP(m_stEncConfig.rcParams.constQP,      inputParam->rcParam.qp);
    setQP(m_stEncConfig.rcParams.initialRCQP,  inputParam->qpInit);
    setQP(m_stEncConfig.rcParams.minQP,        inputParam->qpMin);
    setQP(m_stEncConfig.rcParams.maxQP,        inputParam->qpMax);
    m_stEncConfig.rcParams.multiPass         = inputParam->multipass;
    m_stEncConfig.rcParams.enableInitialRCQP = inputParam->qpInit.enable ? 1 : 0;
    m_stEncConfig.rcParams.enableMinQP       = inputParam->qpMin.enable ? 1 : 0;
    m_stEncConfig.rcParams.enableMaxQP       = inputParam->qpMax.enable ? 1 : 0;
    if (inputParam->strictGOP.has_value()) {
        m_stEncConfig.rcParams.strictGOPTarget = inputParam->strictGOP.value() ? 1 : 0;
    }
    if (inputParam->nonrefP.has_value()) {
        m_stEncConfig.rcParams.enableNonRefP = inputParam->nonrefP.value() ? 1 : 0;
    }
    if (inputParam->enableLookahead.has_value()) {
        m_stEncConfig.rcParams.enableLookahead = inputParam->enableLookahead.value() ? 1 : 0;
    }
    if (inputParam->lookahead.has_value()) {
        m_stEncConfig.rcParams.lookaheadDepth = (uint16_t)inputParam->lookahead.value();
    }
    m_stEncConfig.rcParams.vbvBufferSize     = inputParam->vbvBufferSize;
    if (inputParam->disableIadapt.has_value()) {
        m_stEncConfig.rcParams.disableIadapt = inputParam->disableIadapt.value() ? 1 : 0;
    }
    if (inputParam->disableBadapt.has_value()) {
        m_stEncConfig.rcParams.disableBadapt = inputParam->disableBadapt.value() ? 1 : 0;
    }
    if (inputParam->enableAQ.has_value()) {
        m_stEncConfig.rcParams.enableAQ = inputParam->enableAQ.value() ? 1 : 0;
    }
    if (inputParam->enableAQTemporal.has_value()) {
        m_stEncConfig.rcParams.enableTemporalAQ = inputParam->enableAQTemporal.value() ? 1 : 0;
    }
    if (inputParam->aqStrength.has_value()) {
        m_stEncConfig.rcParams.aqStrength = inputParam->aqStrength.value();
    }
    //その他のパラメータの設定
    if (inputParam->bFrames.has_value()) {
        m_stEncConfig.frameIntervalP = inputParam->bFrames.value() + 1;
    }
    m_stEncConfig.gopLength                  = inputParam->gopLength;
    m_stEncConfig.mvPrecision                = inputParam->mvPrecision;
    m_stEncConfig.frameFieldMode             = NV_ENC_PARAMS_FRAME_FIELD_MODE_FRAME;

    // QVBRの時は、VBRに変更して、averageBitRateとmaxBitRateを0にする
    if (m_stEncConfig.rcParams.rateControlMode == NV_ENC_PARAMS_RC_QVBR) {
        m_stEncConfig.rcParams.rateControlMode = NV_ENC_PARAMS_RC_VBR;
        m_stEncConfig.rcParams.averageBitRate = 0;
    }

    if (m_stEncConfig.rcParams.rateControlMode != (m_stEncConfig.rcParams.rateControlMode & codecFeature->getCapLimit(NV_ENC_CAPS_SUPPORTED_RATECONTROL_MODES))) {
        error_feature_unsupported(RGY_LOG_ERROR, FOR_AUO ? _T("選択されたレート制御モード") : _T("Selected encode mode"));
        return RGY_ERR_UNSUPPORTED;
    }
    if (m_stEncConfig.frameIntervalP < 0) {
        PrintMes(RGY_LOG_ERROR, _T("%s: %d\n"),
            FOR_AUO ? _T("Bフレーム設定が無効です。正の値を使用してください。\n") : _T("B frame settings are invalid. Please use a number > 0.\n"),
            m_stEncConfig.frameIntervalP - 1);
        return RGY_ERR_UNSUPPORTED;
    }
    if (m_stEncConfig.rcParams.enableLookahead && !codecFeature->getCapLimit(NV_ENC_CAPS_SUPPORT_LOOKAHEAD)) {
        error_feature_unsupported(RGY_LOG_WARN, _T("Lookahead"));
        m_stEncConfig.rcParams.enableLookahead = 0;
        m_stEncConfig.rcParams.lookaheadDepth = 0;
        m_stEncConfig.rcParams.disableBadapt = 0;
        m_stEncConfig.rcParams.disableIadapt = 0;
    }
    if (m_stEncConfig.rcParams.enableTemporalAQ && !codecFeature->getCapLimit(NV_ENC_CAPS_SUPPORT_TEMPORAL_AQ)) {
        error_feature_unsupported(RGY_LOG_WARN, _T("Temporal AQ"));
        m_stEncConfig.rcParams.enableTemporalAQ = 0;
    }
    if (inputParam->bluray) {
        if (inputParam->codec_rgy != RGY_CODEC_H264) {
            PrintMes(RGY_LOG_ERROR, _T("Bluray output is not supported only for H.264 codec.\n"));
            return RGY_ERR_UNSUPPORTED;
        }
        if (std::find(VBR_RC_LIST.begin(), VBR_RC_LIST.end(), m_stEncConfig.rcParams.rateControlMode) == VBR_RC_LIST.end()) {
            PrintMes(RGY_LOG_ERROR, FOR_AUO ? _T("Bluray用出力では、VBRモードを使用してください。\n") :  _T("Please use VBR mode for bluray output.\n"));
            return RGY_ERR_UNSUPPORTED;
        }
        if (!codecFeature->getCapLimit(NV_ENC_CAPS_SUPPORT_CUSTOM_VBV_BUF_SIZE)) {
            error_feature_unsupported(RGY_LOG_ERROR, FOR_AUO ? _T("VBVバッファサイズの指定") : _T("Custom VBV Bufsize"));
            PrintMes(RGY_LOG_ERROR, FOR_AUO ? _T("Bluray用出力を行えません。\n") :  _T("Therfore you cannot output for bluray.\n"));
            return RGY_ERR_UNSUPPORTED;
        }
    }
    if (m_stEncConfig.frameIntervalP - 1 > codecFeature->getCapLimit(NV_ENC_CAPS_NUM_MAX_BFRAMES)) {
        m_stEncConfig.frameIntervalP = codecFeature->getCapLimit(NV_ENC_CAPS_NUM_MAX_BFRAMES) + 1;
        PrintMes(RGY_LOG_WARN, FOR_AUO ? _T("Bフレームの最大数は%dです。\n") : _T("Max B frames are %d frames.\n"), codecFeature->getCapLimit(NV_ENC_CAPS_NUM_MAX_BFRAMES));
    }
    // 参照数 (L0/L1), maxRef の反映
    if (inputParam->refL0 != NV_ENC_NUM_REF_FRAMES_AUTOSELECT) {
        set_numRefL0(m_stEncConfig.encodeCodecConfig, inputParam->codec_rgy, inputParam->refL0);
    }
    if (inputParam->refL1 != NV_ENC_NUM_REF_FRAMES_AUTOSELECT) {
        set_numRefL1(m_stEncConfig.encodeCodecConfig, inputParam->codec_rgy, inputParam->refL1);
    }
    numRefFrames(m_stEncConfig.encodeCodecConfig, inputParam->codec_rgy)
        = inputParam->maxRef.value_or((inputParam->codec_rgy == RGY_CODEC_H264) ? DEFAULT_REF_FRAMES_H264 : DEFAULT_REF_FRAMES);

    if (  (    get_numRefL0(m_stEncConfig.encodeCodecConfig, inputParam->codec_rgy) != NV_ENC_NUM_REF_FRAMES_AUTOSELECT
            || get_numRefL1(m_stEncConfig.encodeCodecConfig, inputParam->codec_rgy) != NV_ENC_NUM_REF_FRAMES_AUTOSELECT)) {
        if (!m_dev->encoder()->checkAPIver(9, 1)
            || !codecFeature->getCapLimit(NV_ENC_CAPS_SUPPORT_MULTIPLE_REF_FRAMES)) {
            set_numRefL0(m_stEncConfig.encodeCodecConfig, inputParam->codec_rgy, NV_ENC_NUM_REF_FRAMES_AUTOSELECT);
            set_numRefL1(m_stEncConfig.encodeCodecConfig, inputParam->codec_rgy, NV_ENC_NUM_REF_FRAMES_AUTOSELECT);
            error_feature_unsupported(RGY_LOG_WARN, _T("Multiple Refs"));
        } else if (inputParam->codec_rgy == RGY_CODEC_HEVC) {
            //multirefの制約
            int maxRefL0 = 8, maxRefL1 = 4;
            if (m_stEncConfig.frameIntervalP - 1 > 0) { //Bフレームあり
                maxRefL0--; maxRefL1--;
                if (m_stEncConfig.encodeCodecConfig.hevcConfig.useBFramesAsRef != NV_ENC_BFRAME_REF_MODE_DISABLED) {
                    maxRefL0--; maxRefL1--;
                }
            }
            if ((int)get_numRefL0(m_stEncConfig.encodeCodecConfig, inputParam->codec_rgy) > maxRefL0) {
                PrintMes(RGY_LOG_WARN, _T("multiref(L0) is lowered %d -> %d due to HEVC spec.\n"), get_numRefL0(m_stEncConfig.encodeCodecConfig, inputParam->codec_rgy), maxRefL0);
                set_numRefL0(m_stEncConfig.encodeCodecConfig, inputParam->codec_rgy, (NV_ENC_NUM_REF_FRAMES)maxRefL0);
            }
            if ((int)get_numRefL1(m_stEncConfig.encodeCodecConfig, inputParam->codec_rgy) > maxRefL1) {
                PrintMes(RGY_LOG_WARN, _T("multiref(L1) is lowered %d -> %d due to HEVC spec.\n"), get_numRefL1(m_stEncConfig.encodeCodecConfig, inputParam->codec_rgy), maxRefL1);
                set_numRefL1(m_stEncConfig.encodeCodecConfig, inputParam->codec_rgy, (NV_ENC_NUM_REF_FRAMES)maxRefL1);
            }
        }
    }
    if (inputParam->brefMode > NV_ENC_BFRAME_REF_MODE_DISABLED) {
        const int cap = codecFeature->getCapLimit(NV_ENC_CAPS_SUPPORT_BFRAME_REF_MODE);
        if ((cap & inputParam->brefMode) != inputParam->brefMode) {
            error_feature_unsupported(RGY_LOG_WARN, strsprintf(_T("B Ref Mode %s"), get_chr_from_value(list_bref_mode, inputParam->brefMode)).c_str());
            inputParam->brefMode = NV_ENC_BFRAME_REF_MODE_DISABLED;
        }
    }
    if (inputParam->codec_rgy == RGY_CODEC_H264) {
        if (NV_ENC_H264_ENTROPY_CODING_MODE_CABAC == m_stEncConfig.encodeCodecConfig.h264Config.entropyCodingMode && !codecFeature->getCapLimit(NV_ENC_CAPS_SUPPORT_CABAC)) {
            m_stEncConfig.encodeCodecConfig.h264Config.entropyCodingMode = NV_ENC_H264_ENTROPY_CODING_MODE_CAVLC;
            error_feature_unsupported(RGY_LOG_WARN, _T("CABAC"));
        }
        if (NV_ENC_H264_FMO_ENABLE == m_stEncConfig.encodeCodecConfig.h264Config.fmoMode && !codecFeature->getCapLimit(NV_ENC_CAPS_SUPPORT_FMO)) {
            m_stEncConfig.encodeCodecConfig.h264Config.fmoMode = NV_ENC_H264_FMO_DISABLE;
            error_feature_unsupported(RGY_LOG_WARN, _T("FMO"));
        }
        if (NV_ENC_H264_BDIRECT_MODE_TEMPORAL & m_stEncConfig.encodeCodecConfig.h264Config.bdirectMode && !codecFeature->getCapLimit(NV_ENC_CAPS_SUPPORT_BDIRECT_MODE)) {
            m_stEncConfig.encodeCodecConfig.h264Config.bdirectMode = NV_ENC_H264_BDIRECT_MODE_DISABLE;
            error_feature_unsupported(RGY_LOG_WARN, _T("B Direct mode"));
        }
        if (NV_ENC_H264_ADAPTIVE_TRANSFORM_ENABLE != m_stEncConfig.encodeCodecConfig.h264Config.adaptiveTransformMode && !codecFeature->getCapLimit(NV_ENC_CAPS_SUPPORT_ADAPTIVE_TRANSFORM)) {
            m_stEncConfig.encodeCodecConfig.h264Config.adaptiveTransformMode = NV_ENC_H264_ADAPTIVE_TRANSFORM_DISABLE;
            error_feature_unsupported(RGY_LOG_WARN, _T("Adaptive Tranform"));
        }
        if (m_stEncConfig.encodeCodecConfig.h264Config.hierarchicalPFrames && !codecFeature->getCapLimit(NV_ENC_CAPS_SUPPORT_HIERARCHICAL_PFRAMES)) {
            m_stEncConfig.encodeCodecConfig.h264Config.hierarchicalPFrames = 0;
            error_feature_unsupported(RGY_LOG_WARN, _T("Hierarchical Pframes"));
        }
        if (m_stEncConfig.encodeCodecConfig.h264Config.hierarchicalBFrames && !codecFeature->getCapLimit(NV_ENC_CAPS_SUPPORT_HIERARCHICAL_BFRAMES)) {
            m_stEncConfig.encodeCodecConfig.h264Config.hierarchicalBFrames = 0;
            error_feature_unsupported(RGY_LOG_WARN, _T("Hierarchical Bframes"));
        }
        // hierarchicalP/BFramesにはnumTemporalLayersが必要
        if (m_stEncConfig.encodeCodecConfig.h264Config.hierarchicalPFrames || m_stEncConfig.encodeCodecConfig.h264Config.hierarchicalBFrames) {
            m_stEncConfig.encodeCodecConfig.h264Config.numTemporalLayers = 2;
        }
        // どうも最大3までしかうまく扱えていなさそう (4にするとエラーが発生)
        const decltype(m_stEncConfig.encodeCodecConfig.h264Config.maxTemporalLayers) maxTemporalLayersAvail = std::min(3, codecFeature->getCapLimit(NV_ENC_CAPS_NUM_MAX_TEMPORAL_LAYERS));
        if (m_stEncConfig.encodeCodecConfig.h264Config.maxTemporalLayers > maxTemporalLayersAvail) {
            PrintMes(RGY_LOG_WARN, _T("maxTemporalLayers is lowered %d -> %d.\n"), m_stEncConfig.encodeCodecConfig.h264Config.maxTemporalLayers, maxTemporalLayersAvail);
            m_stEncConfig.encodeCodecConfig.h264Config.maxTemporalLayers = maxTemporalLayersAvail;
        }
        if (m_stEncConfig.encodeCodecConfig.h264Config.numTemporalLayers > maxTemporalLayersAvail) {
            PrintMes(RGY_LOG_WARN, _T("numTemporalLayers is lowered %d -> %d.\n"), m_stEncConfig.encodeCodecConfig.h264Config.numTemporalLayers, maxTemporalLayersAvail);
            m_stEncConfig.encodeCodecConfig.h264Config.numTemporalLayers = maxTemporalLayersAvail;

            const auto requiredRef = (m_stEncConfig.encodeCodecConfig.h264Config.numTemporalLayers >= 2) ? (m_stEncConfig.encodeCodecConfig.h264Config.numTemporalLayers - 2) * 2 : 0;
            if (m_stEncConfig.encodeCodecConfig.h264Config.maxNumRefFrames != 0
                && m_stEncConfig.encodeCodecConfig.h264Config.maxNumRefFrames < requiredRef) {
                m_stEncConfig.encodeCodecConfig.h264Config.maxNumRefFrames = requiredRef;
            }
        }
    }
    if ((NV_ENC_MV_PRECISION_QUARTER_PEL == m_stEncConfig.mvPrecision) && !codecFeature->getCapLimit(NV_ENC_CAPS_SUPPORT_QPELMV)) {
        m_stEncConfig.mvPrecision = NV_ENC_MV_PRECISION_HALF_PEL;
        error_feature_unsupported(RGY_LOG_WARN, FOR_AUO ? _T("1/4画素精度MV探索") : _T("Q-Pel MV"));
    }
    if (0 != m_stEncConfig.rcParams.vbvBufferSize && !codecFeature->getCapLimit(NV_ENC_CAPS_SUPPORT_CUSTOM_VBV_BUF_SIZE)) {
        m_stEncConfig.rcParams.vbvBufferSize = 0;
        error_feature_unsupported(RGY_LOG_WARN, FOR_AUO ? _T("VBVバッファサイズの指定") : _T("Custom VBV Bufsize"));
    }
    if (inputParam->lossless && !codecFeature->getCapLimit(NV_ENC_CAPS_SUPPORT_LOSSLESS_ENCODE)) {
        error_feature_unsupported(RGY_LOG_ERROR, _T("lossless"));
        return RGY_ERR_UNSUPPORTED;
    }
    if (inputParam->lookaheadLevel.value_or(NV_ENC_LOOKAHEAD_LEVEL_0) != NV_ENC_LOOKAHEAD_LEVEL_0 && !codecFeature->getCapLimit(NV_ENC_CAPS_SUPPORT_LOOKAHEAD_LEVEL)) {
        error_feature_unsupported(RGY_LOG_WARN, _T("lookahead-level"));
        inputParam->lookaheadLevel = NV_ENC_LOOKAHEAD_LEVEL_0;
    }
    if (inputParam->codec_rgy == RGY_CODEC_HEVC) {
        if (m_stEncConfig.encodeCodecConfig.hevcConfig.tier == NV_ENC_TIER_HEVC_HIGH) {
            if (m_stEncConfig.encodeCodecConfig.hevcConfig.level != 0
                && !is_avail_high_tier_hevc(m_stEncConfig.encodeCodecConfig.hevcConfig.level)) {
                PrintMes(RGY_LOG_WARN, _T("HEVC Level %s does not support High tier, switching to Main tier.\n"), get_codec_level_name(inputParam->codec_rgy, m_stEncConfig.encodeCodecConfig.hevcConfig.level).c_str());
                m_stEncConfig.encodeCodecConfig.hevcConfig.tier = NV_ENC_TIER_HEVC_MAIN;
            }
        }
        if (inputParam->temporalFilterLevel.value_or(NV_ENC_TEMPORAL_FILTER_LEVEL_0) != NV_ENC_TEMPORAL_FILTER_LEVEL_0 && !codecFeature->getCapLimit(NV_ENC_CAPS_SUPPORT_TEMPORAL_FILTER)) {
            error_feature_unsupported(RGY_LOG_WARN, _T("tf-level"));
            inputParam->temporalFilterLevel = NV_ENC_TEMPORAL_FILTER_LEVEL_0;
        }
    }
    if (rgy_csp_has_alpha(inputParam->outputCsp)) {
        if (inputParam->codec_rgy != RGY_CODEC_HEVC) {
            PrintMes(RGY_LOG_ERROR, _T("Alpha channel encoding only supported in HEVC codec.\n"));
            return RGY_ERR_UNSUPPORTED;
        }
        if (!codecFeature->getCapLimit(NV_ENC_CAPS_SUPPORT_ALPHA_LAYER_ENCODING)) {
            error_feature_unsupported(RGY_LOG_ERROR, _T("alpha channel"));
            return RGY_ERR_UNSUPPORTED;
        }
        if (RGY_CSP_CHROMA_FORMAT[inputParam->outputCsp] != RGY_CHROMAFMT_YUV420) {
            PrintMes(RGY_LOG_ERROR, _T("Alpha channel encoding only supported with YUV420 encoding.\n"));
            return RGY_ERR_UNSUPPORTED;
        }
        if (encodeIsHighBitDepth(inputParam)) {
            PrintMes(RGY_LOG_ERROR, _T("Alpha channel encoding not supported with high bitdepth encoding.\n"));
            return RGY_ERR_UNSUPPORTED;
        }
    }
    if (inputParam->unidirectB && !codecFeature->getCapLimit(NV_ENC_CAPS_SUPPORT_UNIDIRECTIONAL_B)) {
        error_feature_unsupported(RGY_LOG_WARN, _T("Unidirectional B"));
        inputParam->unidirectB = false;
    }
    if (m_dynamicRC.size() > 0 && !codecFeature->getCapLimit(NV_ENC_CAPS_SUPPORT_DYN_BITRATE_CHANGE)) {
        error_feature_unsupported(RGY_LOG_ERROR, _T("dynamic RC Change"));
        return RGY_ERR_UNSUPPORTED;
    }
    if (!m_dev->encoder()->checkAPIver(13, 0)) {
        if (inputParam->splitEncMode == NV_ENC_SPLIT_FOUR_FORCED_MODE) {
            error_feature_unsupported(RGY_LOG_WARN, _T("split-enc = forced_4"));
            inputParam->splitEncMode = NV_ENC_SPLIT_THREE_FORCED_MODE;
        }
    }
    // 自動決定パラメータ
    if (0 == m_stEncConfig.gopLength) {
        m_stEncConfig.gopLength = (int)(m_encFps.n() / (double)m_encFps.d() + 0.5) * 10;
    }
    // LTR 有効/数の反映
    if (inputParam->enableLTR.has_value()) {
        set_enableLTR(m_stEncConfig.encodeCodecConfig, inputParam->codec_rgy, inputParam->enableLTR.value() ? 1 : 0);
    }
    if (inputParam->ltrNumFrames.has_value()) {
        set_ltrNumFrames(m_stEncConfig.encodeCodecConfig, inputParam->codec_rgy, (decltype(get_ltrNumFrames(m_stEncConfig.encodeCodecConfig, inputParam->codec_rgy)))inputParam->ltrNumFrames.value());
    }
    if (get_enableLTR(m_stEncConfig.encodeCodecConfig, inputParam->codec_rgy) && get_ltrNumFrames(m_stEncConfig.encodeCodecConfig, inputParam->codec_rgy) == 0) {
        set_enableLTR(m_stEncConfig.encodeCodecConfig, inputParam->codec_rgy, numRefFrames(m_stEncConfig.encodeCodecConfig, inputParam->codec_rgy));
    }

    auto codecLevel = createCodecLevel(inputParam->codec_rgy);
    const int codecProfile = get_value_from_guid(m_stEncConfig.profileGUID, get_codec_profile_list(inputParam->codec_rgy));
    const bool hevc_high_tier = (inputParam->codec_rgy == RGY_CODEC_HEVC) ? m_stEncConfig.encodeCodecConfig.hevcConfig.tier == NV_ENC_TIER_HEVC_HIGH : false;
    const int av1_tile_col = (inputParam->codec_rgy == RGY_CODEC_AV1) ? m_stEncConfig.encodeCodecConfig.av1Config.numTileColumns : 1;
    const int av1_tile_row = (inputParam->codec_rgy == RGY_CODEC_AV1) ? m_stEncConfig.encodeCodecConfig.av1Config.numTileRows : 1;
    // levelのチェック
    {
        const int level = get_level(m_stEncConfig.encodeCodecConfig, inputParam->codec_rgy);
        if (level != codecLevel->level_auto()) {
            const int required_level = codecLevel->calc_auto_level(m_uEncWidth, m_uEncHeight, 0, is_interlaced(m_stPicStruct),
                m_encFps.n(), m_encFps.d(), codecProfile, hevc_high_tier, 0, 0, 1, 1);
            if (level < required_level) {
                PrintMes(RGY_LOG_WARN, _T("Level %s does not support the current settings (%s @ %s, %dx%d, %d/%d fps), switching level selection to auto.\n"),
                    get_codec_level_name(inputParam->codec_rgy, level).c_str(),
                    CodecToStr(inputParam->codec_rgy).c_str(), get_name_from_guid(m_stEncConfig.profileGUID, get_codec_profile_list(inputParam->codec_rgy)),
                    m_uEncWidth, m_uEncHeight, m_encFps.n(), m_encFps.d());
                set_level(m_stEncConfig.encodeCodecConfig, inputParam->codec_rgy, get_cx_value(get_level_list(inputParam->codec_rgy), _T("auto")));
            } else {
                // refの制限を超えている場合、refを下げる
                const int maxRef = codecLevel->get_max_ref(m_uEncWidth, m_uEncHeight, level, codecProfile);
                if ((int)numRefFrames(m_stEncConfig.encodeCodecConfig, inputParam->codec_rgy) > maxRef) {
                    PrintMes(RGY_LOG_WARN, _T("Ref frames is lowered %d -> %d due to level %s restriction.\n"), numRefFrames(m_stEncConfig.encodeCodecConfig, inputParam->codec_rgy), maxRef, get_codec_level_name(inputParam->codec_rgy, level).c_str());
                    numRefFrames(m_stEncConfig.encodeCodecConfig, inputParam->codec_rgy) = maxRef;
                }
                // 最大bitrateの制限を超えている場合、最大bitrateを下げる
                if (m_stEncConfig.rcParams.rateControlMode != NV_ENC_PARAMS_RC_CONSTQP) {
                    int64_t max_bitrate_kbps = codecLevel->get_max_bitrate(level, codecProfile, hevc_high_tier);
                    if (inputParam->codec_rgy == RGY_CODEC_H264) {
                        if (codecProfile >= 100) {
                            //なぜかhigh profileではぎりぎりを指定するとエラー終了するので、すこし減らす
                            max_bitrate_kbps = (int64_t)(max_bitrate_kbps * 0.96 + 0.5);
                        }
                    } else if (inputParam->codec_rgy == RGY_CODEC_HEVC) {
                        //なぜかぎりぎりを指定するとエラー終了するので、すこし減らす
                        max_bitrate_kbps = (int64_t)(max_bitrate_kbps * 0.96 + 0.5);
                    } else if (inputParam->codec_rgy == RGY_CODEC_AV1) {
                        //なぜか制限値の2/3を指定する必要がある (そうしないとエラーになる)
                        max_bitrate_kbps = (int64_t)(max_bitrate_kbps * 2 / 3);
                    }
                    if ((int64_t)m_stEncConfig.rcParams.averageBitRate > max_bitrate_kbps * 1000) {
                        PrintMes(RGY_LOG_WARN, _T("Bitrate is lowered %d -> %d due to level %s restriction.\n"), m_stEncConfig.rcParams.averageBitRate / 1000, max_bitrate_kbps, get_codec_level_name(inputParam->codec_rgy, level).c_str());
                        m_stEncConfig.rcParams.averageBitRate = (uint32_t)(max_bitrate_kbps * 1000);
                    }
                    if ((int64_t)m_stEncConfig.rcParams.maxBitRate > max_bitrate_kbps * 1000) {
                        PrintMes(RGY_LOG_WARN, _T("Max bitrate is lowered %d -> %d due to level %s restriction.\n"), m_stEncConfig.rcParams.maxBitRate / 1000, max_bitrate_kbps, get_codec_level_name(inputParam->codec_rgy, level).c_str());
                        m_stEncConfig.rcParams.maxBitRate = (uint32_t)(max_bitrate_kbps * 1000);
                    }
                    if (inputParam->codec_rgy == RGY_CODEC_H264) {
                        int max_vbv_buffer_size = codecLevel->get_max_vbv_buf(level, codecProfile);
                        if (codecProfile >= 100) {
                            //なぜかhigh profileではぎりぎりを指定するとエラー終了するので、すこし減らす
                            max_vbv_buffer_size = (int)(max_vbv_buffer_size * 0.96 + 0.5);
                        }
                        if ((int)m_stEncConfig.rcParams.vbvBufferSize > max_vbv_buffer_size * 1000) {
                            PrintMes(RGY_LOG_WARN, _T("VBV buffer size is lowered %d -> %d due to level %s restriction.\n"), m_stEncConfig.rcParams.vbvBufferSize / 1000, max_vbv_buffer_size, get_codec_level_name(inputParam->codec_rgy, level).c_str());
                            m_stEncConfig.rcParams.vbvBufferSize = max_vbv_buffer_size * 1000;
                        }
                    }
                }
            }
        } else if (is_interlaced(m_stPicStruct) && inputParam->codec_rgy == RGY_CODEC_H264) {
            // インタレ保持の場合 Level4.2が上限であり、特別にチェックが必要
            const int maxInterlacedLevel = 42;
            const int maxRef = codecLevel->get_max_ref(m_uEncWidth, m_uEncHeight, maxInterlacedLevel, codecProfile);
            if ((int)numRefFrames(m_stEncConfig.encodeCodecConfig, inputParam->codec_rgy) > maxRef) {
                PrintMes(inputParam->maxRef.has_value() ? RGY_LOG_WARN : RGY_LOG_DEBUG, _T("Ref frames is lowered %d -> %d due to level %s restriction.\n"), numRefFrames(m_stEncConfig.encodeCodecConfig, inputParam->codec_rgy), maxRef, get_codec_level_name(inputParam->codec_rgy, level).c_str());
                numRefFrames(m_stEncConfig.encodeCodecConfig, inputParam->codec_rgy) = maxRef;
            }
        }
    }

    //最大ビットレート自動
    if (m_stEncConfig.rcParams.rateControlMode == NV_ENC_PARAMS_RC_CONSTQP) {
        //CQPモードでは、最大ビットレートの指定は不要
        m_stEncConfig.rcParams.maxBitRate = 0;
    } else if (m_stEncConfig.rcParams.maxBitRate == 0) {
        //指定されたビットレートの1.5倍は最大ビットレートを確保する
        const int prefered_bitrate_kbps = (int)((int64_t)m_stEncConfig.rcParams.averageBitRate * 3 / 2 / 1000);
        int level = get_level(m_stEncConfig.encodeCodecConfig, inputParam->codec_rgy);
        if (level == codecLevel->level_auto()) {
            level = codecLevel->calc_auto_level(m_uEncWidth, m_uEncHeight, numRefFrames(m_stEncConfig.encodeCodecConfig, inputParam->codec_rgy), is_interlaced(m_stPicStruct),
                m_encFps.n(), m_encFps.d(), codecProfile, hevc_high_tier, prefered_bitrate_kbps, m_stEncConfig.rcParams.vbvBufferSize / 1000,
                av1_tile_col, av1_tile_row);
        }
        if (inputParam->codec_rgy == RGY_CODEC_H264) {
            int64_t max_bitrate_kbps = codecLevel->get_max_bitrate(level, codecProfile);
            if (codecProfile >= 100) {
                //なぜかhigh profileではぎりぎりを指定するとエラー終了するので、すこし減らす
                max_bitrate_kbps = (int)(max_bitrate_kbps * 0.96 + 0.5);
            }
            m_stEncConfig.rcParams.maxBitRate = (uint32_t)(max_bitrate_kbps * 1000);
        } else if (inputParam->codec_rgy == RGY_CODEC_HEVC) {
            //なぜかぎりぎりを指定するとエラー終了するので、すこし減らす
            m_stEncConfig.rcParams.maxBitRate = ((uint32_t)((int64_t)codecLevel->get_max_bitrate(level, codecProfile, hevc_high_tier)) * 960);
        } else if (inputParam->codec_rgy == RGY_CODEC_AV1) {
            //なぜか制限値の2/3を指定する必要がある (そうしないとエラーになる)
            m_stEncConfig.rcParams.maxBitRate = (uint32_t)((int64_t)codecLevel->get_max_bitrate(level, codecProfile) * 1000 * 2 / 3);
        } else {
            m_stEncConfig.rcParams.maxBitRate = DEFAULT_MAX_BITRATE;
        }
    }
    // AV1の4Kでのqvbrはなぜか624230kbpsを超えるとよくわからない挙動を示し、qvbr51でも高ビットレートになってしまうので対策。
    static const int64_t AV1_MAX_QVBR_BITRATE_KBPS = 624230;
    if ((m_stEncConfig.rcParams.targetQuality > 0 || m_stEncConfig.rcParams.targetQualityLSB > 0)
        && m_stEncConfig.rcParams.averageBitRate == 0
        && inputParam->codec_rgy == RGY_CODEC_AV1
        && (int64_t)m_stEncConfig.rcParams.maxBitRate >= AV1_MAX_QVBR_BITRATE_KBPS * 1000) {
        PrintMes(RGY_LOG_INFO, _T("using AV1 qvbr max bitrate %d kbps.\n"), AV1_MAX_QVBR_BITRATE_KBPS);
        m_stEncConfig.rcParams.maxBitRate = AV1_MAX_QVBR_BITRATE_KBPS * 1000;
    }
    if (RGY_CSP_CHROMA_FORMAT[inputParam->outputCsp] != RGY_CHROMAFMT_YUV420) {
        if (inputParam->codec_rgy == RGY_CODEC_H264 || inputParam->codec_rgy == RGY_CODEC_HEVC) {
            set_chromaSampleLocationFlag(m_stEncConfig.encodeCodecConfig, inputParam->codec_rgy, 0);
            set_chromaSampleLocationTop(m_stEncConfig.encodeCodecConfig, inputParam->codec_rgy, 0);
            set_chromaSampleLocationBot(m_stEncConfig.encodeCodecConfig, inputParam->codec_rgy, 0);
        }
    }

    //バッファサイズ
    int extraBufSize = 0;
    if (m_pipelineDepth > 1) {
        if (m_uEncWidth * m_uEncHeight <= 2048 * 1080) {
            extraBufSize = 8;
        } else if (m_uEncWidth * m_uEncHeight <= 4096 * 2160) {
            extraBufSize = 4;
        }
    }
    int requiredBufferFrames = m_stEncConfig.frameIntervalP + 4;
    if (m_stEncConfig.rcParams.enableLookahead) {
        requiredBufferFrames += m_stEncConfig.rcParams.lookaheadDepth;
    }
    //m_pipelineDepth分拡張しないと、バッファ不足でエンコードが止まってしまう
    m_encodeBufferCount = requiredBufferFrames + m_pipelineDepth;
    m_encodeBufferCount = std::max(m_encodeBufferCount, std::min(m_encodeBufferCount + extraBufSize, 32));
    if (m_encodeBufferCount > MAX_ENCODE_QUEUE) {
#if FOR_AUO
        PrintMes(RGY_LOG_ERROR, _T("入力バッファは多すぎます。: %d フレーム\n"), m_encodeBufferCount);
        PrintMes(RGY_LOG_ERROR, _T("%d フレームまでに設定して下さい。\n"), MAX_ENCODE_QUEUE);
#else
        PrintMes(RGY_LOG_ERROR, _T("Input frame of %d exceeds the maximum size allowed (%d).\n"), m_encodeBufferCount, MAX_ENCODE_QUEUE);
#endif
        return RGY_ERR_UNSUPPORTED;
    }

    m_stCreateEncodeParams.encodeConfig        = &m_stEncConfig;
    m_stCreateEncodeParams.encodeHeight        = m_uEncHeight;
    m_stCreateEncodeParams.encodeWidth         = m_uEncWidth;
    m_stCreateEncodeParams.darHeight           = m_uEncHeight;
    m_stCreateEncodeParams.darWidth            = m_uEncWidth;
    get_dar_pixels(&m_stCreateEncodeParams.darWidth, &m_stCreateEncodeParams.darHeight, par.first, par.second);

    m_stCreateEncodeParams.maxEncodeHeight     = m_uEncHeight;
    m_stCreateEncodeParams.maxEncodeWidth      = m_uEncWidth;
    m_stCreateEncodeParams.bufferFormat        = GetEncBufferFormat(inputParam);
    m_stCreateEncodeParams.frameRateNum        = m_encFps.n();
    m_stCreateEncodeParams.frameRateDen        = m_encFps.d();
    if (inputParam->nWeightP.has_value() && inputParam->nWeightP.value() != 0) {
        if (!codecFeature->getCapLimit(NV_ENC_CAPS_SUPPORT_WEIGHTED_PREDICTION)) {
            error_feature_unsupported(RGY_LOG_WARN, _T("weighted prediction"));
        } else if (m_stEncConfig.frameIntervalP - 1 > 0) {
            error_feature_unsupported(RGY_LOG_WARN, _T("weighted prediction with B frames"));
        } else {
            if (inputParam->codec_rgy == RGY_CODEC_HEVC) {
                if (m_dev->cc().first == 6 //Pascal
                    || (m_dev->cc().first == 7 && m_dev->cc().second <= 2) //Volta
                    ) {
                    PrintMes(RGY_LOG_WARN, _T("HEVC encode with weightp is known to be unstable on some environments.\n"));
                    PrintMes(RGY_LOG_WARN, _T("Consider not using weightp with HEVC encode if unstable.\n"));
                }
            }
            m_stCreateEncodeParams.enableWeightedPrediction = 1;
        }
    }
    m_stCreateEncodeParams.tuningInfo = inputParam->tuningInfo;

    if (inputParam->ctrl.lowLatency
        && m_dev->encoder()->checkAPIver(10, 0)) {
        m_stCreateEncodeParams.encodeConfig->rcParams.lowDelayKeyFrameScale = 1;
    }
    m_stCreateEncodeParams.enableEncodeAsync   = ENABLE_ASYNC != 0;
    m_stCreateEncodeParams.enablePTD           = true;
    m_stCreateEncodeParams.encodeGUID          = m_stCodecGUID;
    m_stCreateEncodeParams.splitEncodeMode     = inputParam->splitEncMode;

    // enableUniDirectionalB
    if (inputParam->unidirectB && m_dev->encoder()->checkAPIver(12, 2)) {
        const bool hasBframes = (m_stEncConfig.frameIntervalP - 1) > 0;
        if (!hasBframes) {
            m_stCreateEncodeParams.enableUniDirectionalB = 1;
        } else {
            // regular B frames 使用時はAPI仕様により無視される
            m_stCreateEncodeParams.enableUniDirectionalB = 0;
        }
    }

    //bref-modeの自動設定
    if (m_dev->encoder()->checkAPIver(10, 0)) {
        m_stCreateEncodeParams.presetGUID = get_guid_from_value(inputParam->preset, list_nvenc_preset_names_ver10);
    } else {
        m_stCreateEncodeParams.presetGUID = get_guid_from_value(inputParam->preset, list_nvenc_preset_names_ver9_2);
    }
    //Bフレーム数が3以下では強制的に無効
    if (m_stEncConfig.frameIntervalP - 1 < 3 && inputParam->brefMode != NV_ENC_BFRAME_REF_MODE_DISABLED) {
        const auto loglevel = (inputParam->brefMode != NV_ENC_BFRAME_REF_MODE_AUTO && m_stEncConfig.frameIntervalP - 1 > 0) ? RGY_LOG_WARN : RGY_LOG_DEBUG;
        PrintMes(loglevel, _T("bref-mode will be disabled as B-frames is smaller than 3.\n"));
        inputParam->brefMode = NV_ENC_BFRAME_REF_MODE_DISABLED;
    }
    if (inputParam->brefMode == NV_ENC_BFRAME_REF_MODE_AUTO) {
        if (preset_slower_than_default(inputParam->preset)) {
            //defaultより遅いpresetの場合、可能ならbref-modeを使用しないと映像が破綻するケースがある (#449, #458)
            const auto caps = codecFeature->getCapLimit(NV_ENC_CAPS_SUPPORT_BFRAME_REF_MODE);
            for (auto mode : { NV_ENC_BFRAME_REF_MODE_EACH, NV_ENC_BFRAME_REF_MODE_MIDDLE }) {
                if ((caps & mode) == mode) {
                    inputParam->brefMode = mode;
                }
            }
        }
    } else {
        set_useBFramesAsRef(m_stEncConfig.encodeCodecConfig, inputParam->codec_rgy, (NV_ENC_BFRAME_REF_MODE)inputParam->brefMode);
    }

    //ロスレス出力
    if (inputParam->lossless) {
        if (m_dev->encoder()->checkAPIver(10, 0)) {
            m_stCreateEncodeParams.tuningInfo = NV_ENC_TUNING_INFO_LOSSLESS;
        } else {
            #pragma warning (push)
            #pragma warning (disable: 4996)
            RGY_DISABLE_WARNING_PUSH
            RGY_DISABLE_WARNING_STR("-Wdeprecated-declarations")
            switch (inputParam->preset) {
            case NVENC_PRESET_HP:
            case NVENC_PRESET_LL_HP:
                m_stCreateEncodeParams.presetGUID = NV_ENC_PRESET_LOSSLESS_HP_GUID;
                break;
            default:
                m_stCreateEncodeParams.presetGUID = NV_ENC_PRESET_LOSSLESS_DEFAULT_GUID;
                break;
            }
            RGY_DISABLE_WARNING_POP
            #pragma warning (pop)
        }
        //profileは0にしておかないと正常に動作しない
        if (inputParam->codec_rgy == RGY_CODEC_H264) {
            memset(&m_stCreateEncodeParams.encodeConfig->profileGUID, 0, sizeof(m_stCreateEncodeParams.encodeConfig->profileGUID));
            m_stCreateEncodeParams.encodeConfig->encodeCodecConfig.h264Config.qpPrimeYZeroTransformBypassFlag = 1;
        }
        m_stCreateEncodeParams.encodeConfig->rcParams.rateControlMode = NV_ENC_PARAMS_RC_CONSTQP;
        m_stCreateEncodeParams.encodeConfig->rcParams.averageBitRate = 0;
        m_stCreateEncodeParams.encodeConfig->rcParams.maxBitRate = 0;
        m_stCreateEncodeParams.encodeConfig->rcParams.qpMapMode = NV_ENC_QP_MAP_DISABLED;
        m_stCreateEncodeParams.encodeConfig->rcParams.aqStrength = 0;
        m_stCreateEncodeParams.encodeConfig->rcParams.enableAQ = 0;
        m_stCreateEncodeParams.encodeConfig->rcParams.enableTemporalAQ = 0;
        m_stCreateEncodeParams.encodeConfig->rcParams.targetQuality = 0;
        m_stCreateEncodeParams.encodeConfig->rcParams.targetQualityLSB = 0;
        m_stCreateEncodeParams.encodeConfig->rcParams.temporallayerIdxMask = 0;
        memset(&m_stCreateEncodeParams.encodeConfig->rcParams.temporalLayerQP, 0, sizeof(m_stCreateEncodeParams.encodeConfig->rcParams.temporalLayerQP));
        m_stCreateEncodeParams.encodeConfig->rcParams.vbvBufferSize = 0;
        m_stCreateEncodeParams.encodeConfig->rcParams.vbvInitialDelay = 0;
        m_stCreateEncodeParams.encodeConfig->rcParams.constQP.qpIntra = 0;
        m_stCreateEncodeParams.encodeConfig->rcParams.constQP.qpInterP = 0;
        m_stCreateEncodeParams.encodeConfig->rcParams.constQP.qpInterB = 0;
        m_stCreateEncodeParams.encodeConfig->rcParams.enableMinQP = 0;
        m_stCreateEncodeParams.encodeConfig->rcParams.enableMaxQP = 0;
        m_stCreateEncodeParams.encodeConfig->rcParams.enableInitialRCQP = 0;
        m_stCreateEncodeParams.encodeConfig->rcParams.minQP.qpIntra = 0;
        m_stCreateEncodeParams.encodeConfig->rcParams.minQP.qpInterP = 0;
        m_stCreateEncodeParams.encodeConfig->rcParams.minQP.qpInterB = 0;
        m_stCreateEncodeParams.encodeConfig->rcParams.maxQP.qpIntra = 0;
        m_stCreateEncodeParams.encodeConfig->rcParams.maxQP.qpInterP = 0;
        m_stCreateEncodeParams.encodeConfig->rcParams.maxQP.qpInterB = 0;
        m_stCreateEncodeParams.encodeConfig->rcParams.initialRCQP.qpIntra = 0;
        m_stCreateEncodeParams.encodeConfig->rcParams.initialRCQP.qpInterP = 0;
        m_stCreateEncodeParams.encodeConfig->rcParams.initialRCQP.qpInterB = 0;
    }

    m_stEncConfig.rcParams.crQPIndexOffset = 0;
    m_stEncConfig.rcParams.cbQPIndexOffset = 0;
    if (inputParam->chromaQPOffset != 0) {
        if (!m_dev->encoder()->checkAPIver(11, 1)) {
            error_feature_unsupported(RGY_LOG_WARN, _T("chroma qp offset"));
        } else {
            m_stEncConfig.rcParams.crQPIndexOffset = (decltype(m_stEncConfig.rcParams.crQPIndexOffset))inputParam->chromaQPOffset;
            m_stEncConfig.rcParams.cbQPIndexOffset = (decltype(m_stEncConfig.rcParams.cbQPIndexOffset))inputParam->chromaQPOffset;
        }
    }
    //VUI関係の設定
    set_bitstreamRestrictionFlag(m_stCreateEncodeParams.encodeConfig->encodeCodecConfig, inputParam->codec_rgy, 1);

    set_overscanInfoPresentFlag(m_stCreateEncodeParams.encodeConfig->encodeCodecConfig, inputParam->codec_rgy, 0);
    set_overscanInfo(           m_stCreateEncodeParams.encodeConfig->encodeCodecConfig, inputParam->codec_rgy, 0);

    m_encVUI.setDescriptPreset();
    set_colourDescriptionPresentFlag(m_stCreateEncodeParams.encodeConfig->encodeCodecConfig, inputParam->codec_rgy, m_encVUI.descriptpresent);
    if (m_encVUI.descriptpresent) {
        set_colorprim(  m_stCreateEncodeParams.encodeConfig->encodeCodecConfig, inputParam->codec_rgy, m_encVUI.colorprim);
        set_colormatrix(m_stCreateEncodeParams.encodeConfig->encodeCodecConfig, inputParam->codec_rgy, m_encVUI.matrix);
        set_transfer(   m_stCreateEncodeParams.encodeConfig->encodeCodecConfig, inputParam->codec_rgy, m_encVUI.transfer);
    }

    const int videoSignalTypePresentFlag = (get_cx_value(list_videoformat, _T("undef")) != (int)m_encVUI.format
        || m_encVUI.colorrange == RGY_COLORRANGE_FULL
        || m_encVUI.descriptpresent) ? 1 : 0;
    set_videoSignalTypePresentFlag(m_stCreateEncodeParams.encodeConfig->encodeCodecConfig, inputParam->codec_rgy, videoSignalTypePresentFlag);
    if (videoSignalTypePresentFlag) {
        set_videoFormat(m_stCreateEncodeParams.encodeConfig->encodeCodecConfig, inputParam->codec_rgy, m_encVUI.format);
        set_colorrange(m_stCreateEncodeParams.encodeConfig->encodeCodecConfig, inputParam->codec_rgy, m_encVUI.colorrange == RGY_COLORRANGE_FULL ? 1 : 0);
    }
    if (inputParam->codec_rgy == RGY_CODEC_AV1) {
        switch (m_encVUI.chromaloc) {
        case RGY_CHROMALOC_LEFT:    m_stCreateEncodeParams.encodeConfig->encodeCodecConfig.av1Config.chromaSamplePosition = 1; break;
        case RGY_CHROMALOC_TOPLEFT: m_stCreateEncodeParams.encodeConfig->encodeCodecConfig.av1Config.chromaSamplePosition = 2; break;
        default:                    m_stCreateEncodeParams.encodeConfig->encodeCodecConfig.av1Config.chromaSamplePosition = 0; break;
        }
    } else {
        set_chromaSampleLocationFlag(m_stCreateEncodeParams.encodeConfig->encodeCodecConfig, inputParam->codec_rgy, m_encVUI.chromaloc != RGY_CHROMALOC_UNSPECIFIED);
        if (m_encVUI.chromaloc != RGY_CHROMALOC_UNSPECIFIED) {
            set_chromaSampleLocationTop(m_stCreateEncodeParams.encodeConfig->encodeCodecConfig, inputParam->codec_rgy, m_encVUI.chromaloc - 1);
            set_chromaSampleLocationBot(m_stCreateEncodeParams.encodeConfig->encodeCodecConfig, inputParam->codec_rgy, m_encVUI.chromaloc - 1);
        }
    }

    // 整合性チェック
    // idrPeriod 指定があればそれを優先、なければ gopLength を使用
    set_idrPeriod(m_stCreateEncodeParams.encodeConfig->encodeCodecConfig, inputParam->codec_rgy, m_stEncConfig.gopLength);
    // slice 設定
    if (inputParam->slices > 0) {
        set_sliceMode(m_stCreateEncodeParams.encodeConfig->encodeCodecConfig, inputParam->codec_rgy, 3);
        set_sliceModeData(m_stCreateEncodeParams.encodeConfig->encodeCodecConfig, inputParam->codec_rgy, inputParam->slices);
    } else if (get_sliceMode(m_stCreateEncodeParams.encodeConfig->encodeCodecConfig, inputParam->codec_rgy) != 3) {
        set_sliceMode(m_stCreateEncodeParams.encodeConfig->encodeCodecConfig, inputParam->codec_rgy, 3);
        set_sliceModeData(m_stCreateEncodeParams.encodeConfig->encodeCodecConfig, inputParam->codec_rgy, 1);
    }
    set_bitDepth(m_stCreateEncodeParams.encodeConfig->encodeCodecConfig, inputParam->codec_rgy, m_dev->encoder()->getAPIver(), (NV_ENC_BIT_DEPTH)clamp(inputParam->outputDepth, 8, 10));
    
    if (inputParam->codec_rgy == RGY_CODEC_HEVC || inputParam->codec_rgy == RGY_CODEC_AV1) {
        if (!m_dev->encoder()->checkAPIver(13, 0)) {
            set_enableTemporalSVC(m_stCreateEncodeParams.encodeConfig->encodeCodecConfig, inputParam->codec_rgy, inputParam->temporalSVC ? 1 : 0);
        }
    }
    
    if ((inputParam->codec_rgy == RGY_CODEC_HEVC && m_dev->encoder()->checkAPIver(12, 2))
        || ((inputParam->codec_rgy == RGY_CODEC_AV1 || inputParam->codec_rgy == RGY_CODEC_H264) && m_dev->encoder()->checkAPIver(13, 0))) {
        if (inputParam->temporalFilterLevel.has_value()) {
            set_tfLevel(m_stCreateEncodeParams.encodeConfig->encodeCodecConfig, inputParam->codec_rgy, inputParam->temporalFilterLevel.value());
        }
        if (inputParam->lookaheadLevel.has_value()) {
            m_stEncConfig.rcParams.lookaheadLevel = (NV_ENC_LOOKAHEAD_LEVEL)inputParam->lookaheadLevel.value();
        }
    }

    if (inputParam->temporalLayers.has_value()) {
        set_temporalLayers(m_stCreateEncodeParams.encodeConfig->encodeCodecConfig, inputParam->codec_rgy, inputParam->temporalLayers.value(), m_dev->encoder()->getAPIver());
    }

    auto require_repeat_headers = [this, inputParam]() {
        return m_hdr10plus || m_hdr10plusMetadataCopy || m_dovirpuMetadataCopy || (m_hdrseiOut && m_hdrseiOut->gen_nal().size() > 0) || inputParam->ctrl.parallelEnc.isEnabled();
    };

    if (inputParam->codec_rgy == RGY_CODEC_AV1) {
        // AV1 AnnexB / SeqHdr 周り (新フィールド優先)
        if (inputParam->av1.annexB.has_value()) {
            m_stCreateEncodeParams.encodeConfig->encodeCodecConfig.av1Config.outputAnnexBFormat = inputParam->av1.annexB.value() ? 1 : 0;
        } else {
            m_stCreateEncodeParams.encodeConfig->encodeCodecConfig.av1Config.outputAnnexBFormat = 0;
        }
        if (inputParam->av1.disableSeqHdr.has_value()) {
            m_stCreateEncodeParams.encodeConfig->encodeCodecConfig.av1Config.disableSeqHdr = inputParam->av1.disableSeqHdr.value() ? 1 : 0;
        } else {
            m_stCreateEncodeParams.encodeConfig->encodeCodecConfig.av1Config.disableSeqHdr = 0;
        }
        m_stCreateEncodeParams.encodeConfig->encodeCodecConfig.av1Config.repeatSeqHdr = 1;

        // AV1 参照 (前方/後方)
        if (inputParam->av1.fwdRefs != NV_ENC_NUM_REF_FRAMES_AUTOSELECT) {
            m_stCreateEncodeParams.encodeConfig->encodeCodecConfig.av1Config.numFwdRefs = inputParam->av1.fwdRefs;
        }
        if (inputParam->av1.bwdRefs != NV_ENC_NUM_REF_FRAMES_AUTOSELECT) {
            m_stCreateEncodeParams.encodeConfig->encodeCodecConfig.av1Config.numBwdRefs = inputParam->av1.bwdRefs;
        }

        // パーティションサイズ
        if (inputParam->av1.partMin != NV_ENC_AV1_PART_SIZE_AUTOSELECT) {
            m_stCreateEncodeParams.encodeConfig->encodeCodecConfig.av1Config.minPartSize = inputParam->av1.partMin;
        }
        if (inputParam->av1.partMax != NV_ENC_AV1_PART_SIZE_AUTOSELECT) {
            m_stCreateEncodeParams.encodeConfig->encodeCodecConfig.av1Config.maxPartSize = inputParam->av1.partMax;
        }

        // タイル設定
        if (inputParam->av1.tilesCols > 0 && inputParam->av1.tilesRows > 0) {
            m_stCreateEncodeParams.encodeConfig->encodeCodecConfig.av1Config.enableCustomTileConfig = 0;
            m_stCreateEncodeParams.encodeConfig->encodeCodecConfig.av1Config.numTileColumns = inputParam->av1.tilesCols;
            m_stCreateEncodeParams.encodeConfig->encodeCodecConfig.av1Config.numTileRows    = inputParam->av1.tilesRows;
        }
        m_stCreateEncodeParams.encodeConfig->encodeCodecConfig.av1Config.chromaFormatIDC = 1; // YUV420
        m_stCreateEncodeParams.encodeConfig->encodeCodecConfig.av1Config.enableBitstreamPadding = inputParam->bitstreamPadding ? 1 : 0;
    } else if (inputParam->codec_rgy == RGY_CODEC_HEVC) {
        // HEVC 固有設定の反映
        if (inputParam->hevc.cuMin != NV_ENC_HEVC_CUSIZE_AUTOSELECT) {
            m_stCreateEncodeParams.encodeConfig->encodeCodecConfig.hevcConfig.minCUSize = inputParam->hevc.cuMin;
        }
        if (inputParam->hevc.cuMax != NV_ENC_HEVC_CUSIZE_AUTOSELECT) {
            m_stCreateEncodeParams.encodeConfig->encodeCodecConfig.hevcConfig.maxCUSize = inputParam->hevc.cuMax;
        }
        if (m_dev->encoder()->checkAPIver(12, 2)) {
            if (inputParam->lookaheadLevel.has_value()) {
                m_stEncConfig.rcParams.lookaheadLevel = (NV_ENC_LOOKAHEAD_LEVEL)inputParam->lookaheadLevel.value();
            }
        }

        //整合性チェック (一般, H.265/HEVC)
        if (m_stCreateEncodeParams.encodeConfig->encodeCodecConfig.hevcConfig.outputPictureTimingSEI) {
            m_stCreateEncodeParams.encodeConfig->encodeCodecConfig.hevcConfig.outputBufferingPeriodSEI = 1;
        }
        m_stCreateEncodeParams.encodeConfig->encodeCodecConfig.hevcConfig.chromaFormatIDC = 1; // YUV420
        //YUV422/444出力
        if (RGY_CSP_CHROMA_FORMAT[inputParam->outputCsp] != RGY_CHROMAFMT_YUV420) {
            switch (RGY_CSP_CHROMA_FORMAT[inputParam->outputCsp]) {
            case RGY_CHROMAFMT_YUV444:
            case RGY_CHROMAFMT_RGB:
            case RGY_CHROMAFMT_RGB_PACKED:
                m_stCreateEncodeParams.encodeConfig->encodeCodecConfig.hevcConfig.chromaFormatIDC = 3;
                break;
            case RGY_CHROMAFMT_YUV422:
                m_stCreateEncodeParams.encodeConfig->encodeCodecConfig.hevcConfig.chromaFormatIDC = 2;
                break;
            default:
                break;
            }
            //m_stCreateEncodeParams.encodeConfig->encodeCodecConfig.h264Config.separateColourPlaneFlag = 1;
            m_stCreateEncodeParams.encodeConfig->profileGUID = NV_ENC_HEVC_PROFILE_FREXT_GUID;
        } else if (get_bitDepth(m_stEncConfig.encodeCodecConfig, inputParam->codec_rgy, m_dev->encoder()->getAPIver()) > 8) {
            m_stCreateEncodeParams.encodeConfig->profileGUID = NV_ENC_HEVC_PROFILE_MAIN10_GUID;
        }
        if (rgy_csp_has_alpha(inputParam->outputCsp)) {
            m_stCreateEncodeParams.encodeConfig->encodeCodecConfig.hevcConfig.enableAlphaLayerEncoding = 1;
            m_stCreateEncodeParams.encodeConfig->rcParams.alphaLayerBitrateRatio = inputParam->alphaBitrateRatio;
            if (m_stCreateEncodeParams.enableWeightedPrediction) {
                PrintMes(RGY_LOG_WARN, _T("weighted prediction disabled as not supported with alpha channel encoding.\n"));
                m_stCreateEncodeParams.enableWeightedPrediction = 0;
            }
            if (m_stCreateEncodeParams.splitEncodeMode != NV_ENC_SPLIT_AUTO_MODE) {
                PrintMes(RGY_LOG_WARN, _T("split encode mode disabled as not supported with alpha channel encoding.\n"));
                m_stCreateEncodeParams.splitEncodeMode = NV_ENC_SPLIT_AUTO_MODE;
            }
        }
        // AUD/SEI/Repeat headers (H.26x)
        if (inputParam->aud.has_value()) {
            set_outputAUD(m_stCreateEncodeParams.encodeConfig->encodeCodecConfig, inputParam->codec_rgy, inputParam->aud.value() ? 1 : 0);
        }
        if (inputParam->picTimingSEI.has_value()) {
            set_outputPictureTimingSEI(m_stCreateEncodeParams.encodeConfig->encodeCodecConfig, inputParam->codec_rgy, inputParam->picTimingSEI.value() ? 1 : 0);
        }
        if (inputParam->bufferingPeriodSEI.has_value()) {
            set_outputBufferingPeriodSEI(m_stCreateEncodeParams.encodeConfig->encodeCodecConfig, inputParam->codec_rgy, inputParam->bufferingPeriodSEI.value() ? 1 : 0);
        }
        if (inputParam->repeatHeaders.has_value()) {
            set_repeatSPSPPS(m_stCreateEncodeParams.encodeConfig->encodeCodecConfig, inputParam->codec_rgy, inputParam->repeatHeaders.value() ? 1 : 0);
        } else if (require_repeat_headers()) {
            set_repeatSPSPPS(m_stCreateEncodeParams.encodeConfig->encodeCodecConfig, inputParam->codec_rgy, 1);
        }

        if (auto profile = getDOVIProfile(inputParam->common.doviProfile); profile != nullptr && profile->HRDSEI) {
            //制限解除のためコメントアウト
            //if (std::find(VBR_RC_LIST.begin(), VBR_RC_LIST.end(), m_stEncConfig.rcParams.rateControlMode) == VBR_RC_LIST.end()) {
            //    PrintMes(RGY_LOG_ERROR, _T("Please use VBR mode for dolby vision output.\n"));
            //    return RGY_ERR_UNSUPPORTED;
            //}
            //if (!codecFeature->getCapLimit(NV_ENC_CAPS_SUPPORT_CUSTOM_VBV_BUF_SIZE)) {
            //    error_feature_unsupported(RGY_LOG_ERROR, _T("Custom VBV Bufsize"));
            //    PrintMes(RGY_LOG_ERROR, _T("Therfore you cannot output for dolby vision.\n"));
            //    return RGY_ERR_UNSUPPORTED;
            //}
            if (m_stCreateEncodeParams.encodeConfig->rcParams.vbvBufferSize == 0) {
                m_stCreateEncodeParams.encodeConfig->rcParams.vbvBufferSize = m_stCreateEncodeParams.encodeConfig->rcParams.maxBitRate;
            }
            m_stCreateEncodeParams.encodeConfig->rcParams.vbvInitialDelay = m_stCreateEncodeParams.encodeConfig->rcParams.vbvBufferSize / 2;
        }
    } else if (inputParam->codec_rgy == RGY_CODEC_H264) {
        // H.264 固有設定の反映
        if (inputParam->h264.bdirect != NV_ENC_H264_BDIRECT_MODE_AUTOSELECT) {
            m_stCreateEncodeParams.encodeConfig->encodeCodecConfig.h264Config.bdirectMode = inputParam->h264.bdirect;
        }
        if (inputParam->h264.adaptTrans != NV_ENC_H264_ADAPTIVE_TRANSFORM_AUTOSELECT) {
            m_stCreateEncodeParams.encodeConfig->encodeCodecConfig.h264Config.adaptiveTransformMode = inputParam->h264.adaptTrans;
        }
        if (inputParam->h264.entropy != NV_ENC_H264_ENTROPY_CODING_MODE_AUTOSELECT) {
            m_stCreateEncodeParams.encodeConfig->encodeCodecConfig.h264Config.entropyCodingMode = inputParam->h264.entropy;
        }
        if (inputParam->h264.deblockIDC != 0) {
            m_stCreateEncodeParams.encodeConfig->encodeCodecConfig.h264Config.disableDeblockingFilterIDC = inputParam->h264.deblockIDC;
        }
        if (inputParam->temporalLayers.value_or(0) > 0) {
            m_stCreateEncodeParams.encodeConfig->encodeCodecConfig.h264Config.maxTemporalLayers = inputParam->temporalLayers.value();
        }
        // hierarchical P/B frames
        if (inputParam->h264.hierarchicalPFrames.has_value()) {
            m_stCreateEncodeParams.encodeConfig->encodeCodecConfig.h264Config.hierarchicalPFrames = inputParam->h264.hierarchicalPFrames.value() ? 1 : 0;
        }
        if (inputParam->h264.hierarchicalBFrames.has_value()) {
            m_stCreateEncodeParams.encodeConfig->encodeCodecConfig.h264Config.hierarchicalBFrames = inputParam->h264.hierarchicalBFrames.value() ? 1 : 0;
        }
        if (m_stCreateEncodeParams.encodeConfig->encodeCodecConfig.h264Config.hierarchicalPFrames
            || m_stCreateEncodeParams.encodeConfig->encodeCodecConfig.h264Config.hierarchicalBFrames) {
            // H.264 の階層符号化使用時、最低2層を要求
            if (m_stCreateEncodeParams.encodeConfig->encodeCodecConfig.h264Config.numTemporalLayers == 0) {
                m_stCreateEncodeParams.encodeConfig->encodeCodecConfig.h264Config.numTemporalLayers = 2;
            }
        }
        //Bluray 互換出力
        if (inputParam->bluray) {
            m_stCreateEncodeParams.encodeConfig->encodeCodecConfig.h264Config.outputPictureTimingSEI = 1;
            //これをいれるとシークしづらくなる場合があるし、blurayには不要
            //m_stCreateEncodeParams.encodeConfig->encodeCodecConfig.h264Config.outputRecoveryPointSEI = 1;
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
            const auto maxGOPLen =
                (m_uEncWidth <= 1280 && m_uEncHeight <= 720
                && (int)(m_encFps.n() / (double)m_encFps.d() + 0.9) >= 60)
                ? 60u : 30u;
            const bool overMaxGOPLen = m_stCreateEncodeParams.encodeConfig->gopLength > maxGOPLen;
            m_stCreateEncodeParams.encodeConfig->gopLength = (std::min(m_stCreateEncodeParams.encodeConfig->gopLength, maxGOPLen) / m_stCreateEncodeParams.encodeConfig->frameIntervalP) * m_stCreateEncodeParams.encodeConfig->frameIntervalP;
            if (maxGOPLen == 30u && overMaxGOPLen) {
                m_stCreateEncodeParams.encodeConfig->gopLength = 30u;
            }
        }
        if (m_stCreateEncodeParams.encodeConfig->encodeCodecConfig.h264Config.outputPictureTimingSEI) {
            m_stCreateEncodeParams.encodeConfig->encodeCodecConfig.h264Config.outputBufferingPeriodSEI = 1;
        }
        m_stCreateEncodeParams.encodeConfig->encodeCodecConfig.h264Config.chromaFormatIDC = 1; // YUV420
        //YUV422/444出力
        if (RGY_CSP_CHROMA_FORMAT[inputParam->outputCsp] != RGY_CHROMAFMT_YUV420) {
            switch (RGY_CSP_CHROMA_FORMAT[inputParam->outputCsp]) {
            case RGY_CHROMAFMT_YUV444:
            case RGY_CHROMAFMT_RGB:
            case RGY_CHROMAFMT_RGB_PACKED:
                m_stCreateEncodeParams.encodeConfig->encodeCodecConfig.h264Config.chromaFormatIDC = 3;
                m_stCreateEncodeParams.encodeConfig->profileGUID = NV_ENC_H264_PROFILE_HIGH_444_GUID;
                break;
            case RGY_CHROMAFMT_YUV422:
                m_stCreateEncodeParams.encodeConfig->encodeCodecConfig.h264Config.chromaFormatIDC = 2;
                m_stCreateEncodeParams.encodeConfig->profileGUID = NV_ENC_H264_PROFILE_HIGH_422_GUID;
                break;
            default:
                break;
            }
            //m_stCreateEncodeParams.encodeConfig->encodeCodecConfig.h264Config.separateColourPlaneFlag = 1;
        } else if (get_bitDepth(m_stEncConfig.encodeCodecConfig, inputParam->codec_rgy, m_dev->encoder()->getAPIver()) > 8) {
            m_stCreateEncodeParams.encodeConfig->profileGUID = NV_ENC_H264_PROFILE_HIGH_10_GUID;
        }

        //整合性チェック (一般, H.264/AVC)
        m_stCreateEncodeParams.encodeConfig->frameFieldMode = (m_stPicStruct == NV_ENC_PIC_STRUCT_FRAME) ? NV_ENC_PARAMS_FRAME_FIELD_MODE_FRAME : NV_ENC_PARAMS_FRAME_FIELD_MODE_FIELD;
        //m_stCreateEncodeParams.encodeConfig->encodeCodecConfig.h264Config.entropyCodingMode = (m_stEncoderInput[0].profile > 66) ? NV_ENC_H264_ENTROPY_CODING_MODE_CABAC : NV_ENC_H264_ENTROPY_CODING_MODE_CAVLC;
        if (m_stCreateEncodeParams.encodeConfig->frameIntervalP - 1 <= 0) {
            m_stCreateEncodeParams.encodeConfig->encodeCodecConfig.h264Config.bdirectMode = NV_ENC_H264_BDIRECT_MODE_DISABLE;
            m_stCreateEncodeParams.encodeConfig->encodeCodecConfig.h264Config.hierarchicalBFrames = 0;
        }
        // AUD/SEI/Repeat headers (H.26x)
        if (inputParam->aud.has_value()) {
            set_outputAUD(m_stCreateEncodeParams.encodeConfig->encodeCodecConfig, inputParam->codec_rgy, inputParam->aud.value() ? 1 : 0);
        }
        if (inputParam->picTimingSEI.has_value()) {
            set_outputPictureTimingSEI(m_stCreateEncodeParams.encodeConfig->encodeCodecConfig, inputParam->codec_rgy, inputParam->picTimingSEI.value() ? 1 : 0);
        }
        if (inputParam->bufferingPeriodSEI.has_value()) {
            set_outputBufferingPeriodSEI(m_stCreateEncodeParams.encodeConfig->encodeCodecConfig, inputParam->codec_rgy, inputParam->bufferingPeriodSEI.value() ? 1 : 0);
        }
        if (inputParam->repeatHeaders.has_value()) {
            set_repeatSPSPPS(m_stCreateEncodeParams.encodeConfig->encodeCodecConfig, inputParam->codec_rgy, inputParam->repeatHeaders.value() ? 1 : 0);
        } else if (require_repeat_headers()) {
            set_repeatSPSPPS(m_stCreateEncodeParams.encodeConfig->encodeCodecConfig, inputParam->codec_rgy, 1);
        }
    } else {
        PrintMes(RGY_LOG_ERROR, _T("Unknown codec.\n"));
        return RGY_ERR_UNSUPPORTED;
    }

    return RGY_ERR_NONE;
}

std::vector<VppType> NVEncCore::InitFiltersCreateVppList(const InEncodeVideoParam *inputParam, const bool cspConvRequired, const bool cropRequired, const RGY_VPP_RESIZE_TYPE resizeRequired) {
    std::vector<VppType> filterPipeline;
    filterPipeline.reserve((size_t)VppType::CL_MAX);

    if (cspConvRequired || cropRequired)   filterPipeline.push_back(VppType::CL_CROP);
    if (inputParam->vpp.colorspace.enable) {
#if 0
        bool requireOpenCL = inputParam->vpp.colorspace.hdr2sdr.tonemap != HDR2SDR_DISABLED || inputParam->vpp.colorspace.lut3d.table_file.length() > 0;
        if (!requireOpenCL) {
            auto currentVUI = inputParam->input.vui;
            for (size_t i = 0; i < inputParam->vpp.colorspace.convs.size(); i++) {
                auto conv_from = inputParam->vpp.colorspace.convs[i].from;
                auto conv_to = inputParam->vpp.colorspace.convs[i].to;
                if (conv_from.chromaloc != conv_to.chromaloc
                    || conv_from.colorprim != conv_to.colorprim
                    || conv_from.transfer != conv_to.transfer) {
                    requireOpenCL = true;
                } else if (conv_from.matrix != conv_to.matrix
                    && (conv_from.matrix != RGY_MATRIX_ST170_M && conv_from.matrix != RGY_MATRIX_BT709)
                    && (conv_to.matrix != RGY_MATRIX_ST170_M && conv_to.matrix != RGY_MATRIX_BT709)) {
                    requireOpenCL = true;
                }
            }
        }
        filterPipeline.push_back((requireOpenCL) ? VppType::CL_COLORSPACE : VppType::AMF_COLORSPACE);
#else
        filterPipeline.push_back(VppType::CL_COLORSPACE);
#endif
    }
    if (inputParam->vpp.libplacebo_tonemapping.enable) filterPipeline.push_back(VppType::CL_LIBPLACEBO_TONEMAP);
    if (inputParam->vpp.rff.enable)           filterPipeline.push_back(VppType::CL_RFF);
    if (inputParam->vpp.delogo.enable)        filterPipeline.push_back(VppType::CL_DELOGO);
    if (inputParam->vpp.afs.enable)           filterPipeline.push_back(VppType::CL_AFS);
    if (inputParam->vpp.nnedi.enable)         filterPipeline.push_back(VppType::CL_NNEDI);
    if (inputParam->vpp.yadif.enable)         filterPipeline.push_back(VppType::CL_YADIF);
    if (inputParam->vpp.decomb.enable)        filterPipeline.push_back(VppType::CL_DECOMB);
    if (inputParam->vpp.decimate.enable)      filterPipeline.push_back(VppType::CL_DECIMATE);
    if (inputParam->vpp.mpdecimate.enable)    filterPipeline.push_back(VppType::CL_MPDECIMATE);
    if (inputParam->vpp.selectevery.enable)   filterPipeline.push_back(VppType::CL_SELECT_EVERY);
    if (inputParam->vpp.transform.enable)     filterPipeline.push_back(VppType::CL_TRANSFORM);
    if (inputParam->vpp.convolution3d.enable) filterPipeline.push_back(VppType::CL_CONVOLUTION3D);
    if (inputParam->vppnv.nvvfxDenoise.enable) filterPipeline.push_back(VppType::NVVFX_DENOISE);
    if (inputParam->vppnv.nvvfxArtifactReduction.enable) filterPipeline.push_back(VppType::NVVFX_ARTIFACT_REDUCTION);
    if (inputParam->vpp.smooth.enable)        filterPipeline.push_back(VppType::CL_DENOISE_SMOOTH);
    if (inputParam->vpp.dct.enable)           filterPipeline.push_back(VppType::CL_DENOISE_DCT);
    if (inputParam->vpp.fft3d.enable)         filterPipeline.push_back(VppType::CL_DENOISE_FFT3D);
    if (inputParam->vpp.knn.enable)           filterPipeline.push_back(VppType::CL_DENOISE_KNN);
    if (inputParam->vpp.nlmeans.enable)       filterPipeline.push_back(VppType::CL_DENOISE_NLMEANS);
    if (inputParam->vpp.pmd.enable)           filterPipeline.push_back(VppType::CL_DENOISE_PMD);
    if (inputParam->vppnv.gaussMaskSize>0)    filterPipeline.push_back(VppType::NPP_GAUSS);
    if (inputParam->vpp.subburn.size()>0)     filterPipeline.push_back(VppType::CL_SUBBURN);
    if (inputParam->vpp.libplacebo_shader.size() > 0)  filterPipeline.push_back(VppType::CL_LIBPLACEBO_SHADER);
    if (resizeRequired != RGY_VPP_RESIZE_TYPE_NONE) filterPipeline.push_back(VppType::CL_RESIZE);
    if (inputParam->vpp.unsharp.enable)    filterPipeline.push_back(VppType::CL_UNSHARP);
    if (inputParam->vpp.edgelevel.enable)  filterPipeline.push_back(VppType::CL_EDGELEVEL);
    if (inputParam->vpp.warpsharp.enable)  filterPipeline.push_back(VppType::CL_WARPSHARP);
    if (inputParam->vpp.curves.enable)     filterPipeline.push_back(VppType::CL_CURVES);
    if (inputParam->vpp.tweak.enable)      filterPipeline.push_back(VppType::CL_TWEAK);
    if (inputParam->vpp.deband.enable)     filterPipeline.push_back(VppType::CL_DEBAND);
    if (inputParam->vpp.libplacebo_deband.enable)     filterPipeline.push_back(VppType::CL_LIBPLACEBO_DEBAND);
    if (inputParam->vpp.pad.enable)        filterPipeline.push_back(VppType::CL_PAD);
    if (inputParam->vpp.overlay.size() > 0)  filterPipeline.push_back(VppType::CL_OVERLAY);
    if (inputParam->vppnv.ngxTrueHDR.enable)     filterPipeline.push_back(VppType::NGX_TRUEHDR);
    if (inputParam->vpp.fruc.enable)     filterPipeline.push_back(VppType::CL_FRUC);

    if (filterPipeline.size() == 0) {
        return filterPipeline;
    }
#if ENCODER_VCEENC
    // HQScalerが使用できない場合
    if (!m_dev->getFilterCaps(AMFHQScaler)) {
        //置き換え
        for (auto& filter : filterPipeline) {
            if (filter == VppType::AMF_RESIZE) {
                if (resizeRequired == RGY_VPP_RESIZE_TYPE_AMF) {
                    PrintMes(RGY_LOG_WARN, _T("HQScaler unavailable, switching to OpenCL based resize filter.\n"));
                }
                filter = VppType::CL_RESIZE;
            }
        }
    }
    //OpenCLが使用できない場合
    if (!m_dev->cl()) {
        //置き換え
        for (auto& filter : filterPipeline) {
            if (filter == VppType::CL_RESIZE) filter = VppType::AMF_RESIZE;
        }
        //削除
        decltype(filterPipeline) newPipeline;
        for (auto& filter : filterPipeline) {
            if (getVppFilterType(filter) != VppFilterType::FILTER_OPENCL) {
                newPipeline.push_back(filter);
            }
        }
        if (filterPipeline.size() != newPipeline.size()) {
            PrintMes(RGY_LOG_WARN, _T("OpenCL disabled, OpenCL based vpp filters will be disabled!\n"));
        }
        filterPipeline = newPipeline;
    }
    // cropとresizeはmfxとopencl両方ともあるので、前後のフィルタがどちらもOpenCLだったら、そちらに合わせる
    for (size_t i = 0; i < filterPipeline.size(); i++) {
        const VppFilterType prev = (i >= 1)                        ? getVppFilterType(filterPipeline[i - 1]) : VppFilterType::FILTER_NONE;
        const VppFilterType next = (i + 1 < filterPipeline.size()) ? getVppFilterType(filterPipeline[i + 1]) : VppFilterType::FILTER_NONE;
        if (filterPipeline[i] == VppType::AMF_RESIZE) {
            if (resizeRequired == RGY_VPP_RESIZE_TYPE_AUTO // 自動以外の指定があれば、それに従うので、自動の場合のみ変更
                && m_dev->cl()
                && prev == VppFilterType::FILTER_OPENCL
                && next == VppFilterType::FILTER_OPENCL) {
                filterPipeline[i] = VppType::CL_RESIZE; // OpenCLに挟まれていたら、OpenCLのresizeを優先する
            }
        } else if (filterPipeline[i] == VppType::AMF_CONVERTER) {
            if (m_dev->cl()
                && prev == VppFilterType::FILTER_OPENCL
                && next == VppFilterType::FILTER_OPENCL) {
                filterPipeline[i] = VppType::CL_COLORSPACE; // OpenCLに挟まれていたら、OpenCLのcolorspaceを優先する
            }
        }
    }
#endif
    return filterPipeline;
}

RGY_ERR NVEncCore::InitFilters(const InEncodeVideoParam *inputParam) {
    //cuvidデコーダの場合、cropを入力時に行っていない場合がある
    const bool cropRequired = cropEnabled(inputParam->input.crop)
        && m_pFileReader->getInputCodec() != RGY_CODEC_UNKNOWN
        && CUVID_DISABLE_CROP;

    RGYFrameInfo inputFrame;
    inputFrame.width = inputParam->input.srcWidth;
    inputFrame.height = inputParam->input.srcHeight;
    inputFrame.csp = inputParam->input.csp;
    inputFrame.picstruct = inputParam->input.picstruct;
    const int croppedWidth = inputFrame.width - inputParam->input.crop.e.left - inputParam->input.crop.e.right;
    const int croppedHeight = inputFrame.height - inputParam->input.crop.e.bottom - inputParam->input.crop.e.up;
    if (!cropRequired) {
        //入力時にcrop済み
        inputFrame.width = croppedWidth;
        inputFrame.height = croppedHeight;
    }
    // 読み込み時に同時にGPUに転送されるので、スタートは常にGPU
    inputFrame.mem_type = RGY_MEM_TYPE_GPU;
    m_encFps = rgy_rational<int>(inputParam->input.fpsN, inputParam->input.fpsD);
    if (inputParam->vppnv.deinterlace == cudaVideoDeinterlaceMode_Bob) {
        m_encFps *= 2;
    }

    const bool cspConvRequired = inputFrame.csp != GetEncoderCSP(inputParam);

    //リサイザの出力すべきサイズ
    int resizeWidth  = 0;
    int resizeHeight = 0;

    //指定のリサイズがあればそのサイズに設定する
    if (inputParam->input.dstWidth > 0 && inputParam->input.dstHeight > 0) {
        resizeWidth = inputParam->input.dstWidth;
        resizeHeight = inputParam->input.dstHeight;
        if (inputParam->vpp.pad.enable) {
            resizeWidth -= (inputParam->vpp.pad.right + inputParam->vpp.pad.left);
            resizeHeight -= (inputParam->vpp.pad.bottom + inputParam->vpp.pad.top);
        }
    }
    RGY_VPP_RESIZE_TYPE resizeRequired = RGY_VPP_RESIZE_TYPE_NONE;
    if ((resizeWidth > 0 && resizeHeight > 0) &&
        (croppedWidth != resizeWidth || croppedHeight != resizeHeight)) {
        resizeRequired = getVppResizeType(inputParam->vpp.resize_algo);
        if (resizeRequired == RGY_VPP_RESIZE_TYPE_UNKNOWN) {
            PrintMes(RGY_LOG_ERROR, _T("Unknown resize type.\n"));
            return RGY_ERR_INVALID_VIDEO_PARAM;
        }
    }
    //avhw読みではデコード直後にリサイズが可能
    if (resizeRequired && m_pFileReader->getInputCodec() != RGY_CODEC_UNKNOWN && enableCuvidResize(inputParam)) {
        inputFrame.width  = inputParam->input.dstWidth;
        inputFrame.height = inputParam->input.dstHeight;
        resizeRequired = RGY_VPP_RESIZE_TYPE_NONE;
    }

    //picStructの設定
    m_stPicStruct = picstruct_rgy_to_enc(inputParam->input.picstruct);
    if (inputParam->vppnv.deinterlace != cudaVideoDeinterlaceMode_Weave) {
        m_stPicStruct = NV_ENC_PIC_STRUCT_FRAME;
        inputFrame.picstruct = RGY_PICSTRUCT_FRAME;
    } else if (inputParam->vpp.afs.enable || inputParam->vpp.nnedi.enable || inputParam->vpp.yadif.enable || inputParam->vpp.decomb.enable) {
        m_stPicStruct = NV_ENC_PIC_STRUCT_FRAME;
    }
    //インタレ解除の個数をチェック
    int deinterlacer = 0;
    if (inputParam->vppnv.deinterlace != cudaVideoDeinterlaceMode_Weave) deinterlacer++;
    if (inputParam->vpp.afs.enable) deinterlacer++;
    if (inputParam->vpp.nnedi.enable) deinterlacer++;
    if (inputParam->vpp.yadif.enable) deinterlacer++;
    if (inputParam->vpp.decomb.enable) deinterlacer++;
    if (deinterlacer >= 2) {
        PrintMes(RGY_LOG_ERROR, _T("Activating 2 or more deinterlacer is not supported.\n"));
        return RGY_ERR_UNSUPPORTED;
    }

    //VUI情報
    auto VuiFiltered = inputParam->input.vui;

    //vpp-rffの制約事項
    if (inputParam->vpp.rff.enable) {
        if (inputParam->vppnv.deinterlace != cudaVideoDeinterlaceMode_Weave) {
            PrintMes(RGY_LOG_ERROR, _T("vpp-rff cannot be used with vpp-deinterlace.\n"));
            return RGY_ERR_UNSUPPORTED;
        }
        if (trim_active(&m_trimParam)) {
            PrintMes(RGY_LOG_ERROR, _T("vpp-rff cannot be used with trim.\n"));
            return RGY_ERR_UNSUPPORTED;
        }
    }

    std::vector<VppType> filterPipeline = InitFiltersCreateVppList(inputParam, cspConvRequired, cropRequired, resizeRequired);
    //読み込み時のcrop
    const sInputCrop *inputCrop = (cropRequired) ? &inputParam->input.crop : nullptr;
    const auto resize = std::make_pair(resizeWidth, resizeHeight);

    std::vector<std::unique_ptr<NVEncFilter>> vppCUDAFilters;
    const auto encCsp = GetEncoderCSP(inputParam);
    auto filterCsp = encCsp;
    switch (filterCsp) {
    case RGY_CSP_NV12:  filterCsp = RGY_CSP_YV12; break;
    case RGY_CSP_P010:  filterCsp = RGY_CSP_YV12_16; break;
    case RGY_CSP_NV16:  filterCsp = RGY_CSP_YUV444; break;
    case RGY_CSP_P210:  filterCsp = RGY_CSP_YUV444_16; break;
    case RGY_CSP_NV12A: filterCsp = RGY_CSP_YUVA420; break;
    case RGY_CSP_P010A: filterCsp = RGY_CSP_YUVA420_16; break;
    default: break;
    }
    if (inputParam->vpp.afs.enable && RGY_CSP_CHROMA_FORMAT[inputFrame.csp] == RGY_CHROMAFMT_YUV444) {
        filterCsp = (RGY_CSP_BIT_DEPTH[inputFrame.csp] > 8) ? RGY_CSP_YUV444_16 : RGY_CSP_YUV444;
    }

    size_t ifilter = 0;
    if (filterPipeline.size() > 0 && (inputFrame.csp != filterCsp || filterPipeline.front() == VppType::CL_CROP)) {
        if (filterPipeline.front() == VppType::CL_CROP) {
            ifilter++;
        }
        unique_ptr<NVEncFilter> filterCrop(new NVEncFilterCspCrop());
        shared_ptr<NVEncFilterParamCrop> param(new NVEncFilterParamCrop());
        param->frameIn = inputFrame;
        param->frameOut = inputFrame;
        param->frameOut.csp = filterCsp;
        if (filterPipeline.size() > ifilter && filterPipeline[ifilter] == VppType::CL_COLORSPACE) { // 次のフィルタがcolorpaceの時はなるべくそのまま転送するように
            if (RGY_CSP_CHROMA_FORMAT[inputFrame.csp] == RGY_CHROMAFMT_RGB || RGY_CSP_CHROMA_FORMAT[inputFrame.csp] == RGY_CHROMAFMT_RGB_PACKED) {
                param->frameOut.csp = (rgy_csp_has_alpha(inputFrame.csp))
                    ? ((RGY_CSP_BIT_DEPTH[inputFrame.csp] > 8) ? RGY_CSP_RGBA_16 : RGY_CSP_RGBA)
                    : ((RGY_CSP_BIT_DEPTH[inputFrame.csp] > 8) ? RGY_CSP_RGB_16  : RGY_CSP_RGB);
            } else if (RGY_CSP_CHROMA_FORMAT[inputFrame.csp] == RGY_CHROMAFMT_YUV444 || RGY_CSP_CHROMA_FORMAT[inputFrame.csp] == RGY_CHROMAFMT_YUV422) {
                param->frameOut.csp = (rgy_csp_has_alpha(inputFrame.csp))
                    ? ((RGY_CSP_BIT_DEPTH[inputFrame.csp] > 8) ? RGY_CSP_YUVA444_16 : RGY_CSP_YUVA444)
                    : ((RGY_CSP_BIT_DEPTH[inputFrame.csp] > 8) ? RGY_CSP_YUV444_16  : RGY_CSP_YUV444);
            }
        }
        param->frameOut.bitdepth = RGY_CSP_BIT_DEPTH[param->frameOut.csp];
        param->baseFps = m_encFps;
        if (inputCrop) {
            param->crop = *inputCrop;
            inputCrop = nullptr;
        }
        param->bOutOverwrite = false;
        NVEncCtxAutoLock(cxtlock(m_dev->vidCtxLock()));
        auto sts = filterCrop->init(param, m_pLog);
        if (sts != RGY_ERR_NONE) {
            return sts;
        }
        //パラメータ情報を更新
        m_pLastFilterParam = std::dynamic_pointer_cast<NVEncFilterParam>(param);
        //入力フレーム情報を更新
        inputFrame = param->frameOut;
        m_encFps = param->baseFps;
        vppCUDAFilters.push_back(std::move(filterCrop));
    }
    for (; ifilter < filterPipeline.size(); ifilter++) {
        auto err = AddFilterCUDA(vppCUDAFilters, inputFrame, filterPipeline[ifilter], inputParam, inputCrop, resize, VuiFiltered);
        if (err != RGY_ERR_NONE) {
            PrintMes(RGY_LOG_ERROR, _T("Unsupported vpp filter type.\n"));
            return RGY_ERR_UNSUPPORTED;
        }
    }
    //最後のフィルタ (かならず一つはコピーが必要)
    {
        const auto cropOutCsp = (inputParam->codec_rgy == RGY_CODEC_RAW || inputParam->codec_rgy == RGY_CODEC_AVCODEC) ? GetRawOutCSP(inputParam) : GetEncoderCSP(inputParam);
        //インタレ保持の場合、またエンコーダを行わない場合は、CPU側に戻す必要がある
        const auto outMemType = ((!ENABLE_INTERLACE_FROM_HWMEM && m_stPicStruct != NV_ENC_PIC_STRUCT_FRAME) || !m_dev->encoder()) ? RGY_MEM_TYPE_CPU : RGY_MEM_TYPE_GPU;
        // cspとmemtypeが両方一致しなかったら、まずはcspを変換
        if (inputFrame.csp != cropOutCsp && inputFrame.mem_type != outMemType) {
            unique_ptr<NVEncFilter> filterCrop(new NVEncFilterCspCrop());
            shared_ptr<NVEncFilterParamCrop> param(new NVEncFilterParamCrop());
            param->frameIn = inputFrame;
            param->frameOut = inputFrame;
            param->frameOut.csp = cropOutCsp;
            param->baseFps = m_encFps;
            param->matrix = VuiFiltered.matrix;
            //インタレ保持であれば、CPU側にフレームを戻す必要がある
            //色空間が同じなら、ここでやってしまう
            param->frameOut.mem_type = RGY_MEM_TYPE_GPU;
            param->bOutOverwrite = false;
            NVEncCtxAutoLock(cxtlock(m_dev->vidCtxLock()));
            auto sts = filterCrop->init(param, m_pLog);
            if (sts != RGY_ERR_NONE) {
                return sts;
            }
            vppCUDAFilters.push_back(std::move(filterCrop));
            m_pLastFilterParam = std::dynamic_pointer_cast<NVEncFilterParam>(param);
            //入力フレーム情報を更新
            inputFrame = param->frameOut;
            m_encFps = param->baseFps;
        }

        // cspとmemtypeをこれで一致させる
        unique_ptr<NVEncFilter> filterCrop(new NVEncFilterCspCrop());
        shared_ptr<NVEncFilterParamCrop> param(new NVEncFilterParamCrop());
        param->frameIn = inputFrame;
        param->frameOut = inputFrame;
        param->frameOut.csp = cropOutCsp;
        param->frameOut.mem_type = outMemType;
        param->baseFps = m_encFps;
        param->matrix = VuiFiltered.matrix;
        param->bOutOverwrite = false;
        NVEncCtxAutoLock(cxtlock(m_dev->vidCtxLock()));
        auto sts = filterCrop->init(param, m_pLog);
        if (sts != RGY_ERR_NONE) {
            return sts;
        }
        vppCUDAFilters.push_back(std::move(filterCrop));
        m_pLastFilterParam = std::dynamic_pointer_cast<NVEncFilterParam>(param);
        //入力フレーム情報を更新
        inputFrame = param->frameOut;
        m_encFps = param->baseFps;
    }

    m_vpFilters.push_back(VppVilterBlock(vppCUDAFilters));

    //パフォーマンスチェックを行うかどうか
    if (inputParam->vpp.checkPerformance) {
        NVEncCtxAutoLock(cxtlock(m_dev->vidCtxLock()));
        for (auto& vppblock : m_vpFilters) {
            for (auto& filter : vppblock.vppnv) {
                filter->setCheckPerformance(inputParam->vpp.checkPerformance);
            }
        }
    }
    m_uEncWidth = inputFrame.width;
    m_uEncHeight = inputFrame.height;
    m_stPicStruct = picstruct_rgy_to_enc(inputFrame.picstruct);
    m_encVUI = inputParam->common.out_vui;
    if (m_rgbAsYUV444) {
        m_encVUI.descriptpresent = 1;
        if (m_encVUI.matrix == RGY_MATRIX_UNSPECIFIED) m_encVUI.matrix = RGY_MATRIX_AUTO;
        if (m_encVUI.colorprim == RGY_PRIM_UNSPECIFIED) m_encVUI.colorprim = RGY_PRIM_AUTO;
        if (m_encVUI.transfer == RGY_TRANSFER_UNSPECIFIED) m_encVUI.transfer = RGY_TRANSFER_AUTO;
        if (m_encVUI.colorrange == RGY_COLORRANGE_UNSPECIFIED) m_encVUI.colorrange = RGY_COLORRANGE_AUTO;
        VuiFiltered.matrix = RGY_MATRIX_RGB;
        VuiFiltered.colorprim = RGY_PRIM_BT709;
        VuiFiltered.transfer = RGY_TRANSFER_IEC61966_2_1;
        VuiFiltered.colorrange = RGY_COLORRANGE_FULL;
    }
    m_encVUI.apply_auto(VuiFiltered, m_uEncHeight);
    return RGY_ERR_NONE;
}

RGY_ERR NVEncCore::AddFilterCUDA(std::vector<std::unique_ptr<NVEncFilter>>& cufilters,
    RGYFrameInfo & inputFrame, const VppType vppType, const InEncodeVideoParam *inputParam, [[maybe_unused]] const sInputCrop *crop, const std::pair<int, int> resize, VideoVUIInfo& vuiInfo) {
    //フィルタが必要
    //colorspace
    if (vppType == VppType::CL_COLORSPACE) {
        unique_ptr<NVEncFilterColorspace> filter(new NVEncFilterColorspace());
        shared_ptr<NVEncFilterParamColorspace> param(new NVEncFilterParamColorspace());
        param->colorspace = inputParam->vpp.colorspace;
        param->encCsp = inputFrame.csp;
        param->VuiIn = vuiInfo;
        param->frameIn = inputFrame;
        param->frameOut = inputFrame;
        param->baseFps = m_encFps;
        NVEncCtxAutoLock(cxtlock(m_dev->vidCtxLock()));
        auto sts = filter->init(param, m_pLog);
        if (sts != RGY_ERR_NONE) {
            return sts;
        }
        vuiInfo = filter->VuiOut();
        //フィルタチェーンに追加
        cufilters.push_back(std::move(filter));
        //パラメータ情報を更新
        m_pLastFilterParam = std::dynamic_pointer_cast<NVEncFilterParam>(param);
        //入力フレーム情報を更新
        inputFrame = param->frameOut;
        m_encFps = param->baseFps;
        return RGY_ERR_NONE;
    }
    //libplacebo_tonemapping
    if (vppType == VppType::CL_LIBPLACEBO_TONEMAP) {
        unique_ptr<NVEncFilterLibplaceboToneMapping> filter(new NVEncFilterLibplaceboToneMapping());
        shared_ptr<NVEncFilterParamLibplaceboToneMapping> param(new NVEncFilterParamLibplaceboToneMapping());
        param->toneMapping = inputParam->vpp.libplacebo_tonemapping;
        param->vui = vuiInfo;
        param->dx11 = m_dev->dx11();
        param->vk = m_dev->vulkan();
        param->hdrMetadataIn = m_hdrseiIn.get();
        param->hdrMetadataOut = m_hdrseiOut.get();
        param->frameIn = inputFrame;
        param->frameOut = inputFrame;
        param->baseFps = m_encFps;
        NVEncCtxAutoLock(cxtlock(m_dev->vidCtxLock()));
        auto sts = filter->init(param, m_pLog);
        if (sts != RGY_ERR_NONE) {
            return sts;
        }
    #if ENABLE_LIBPLACEBO
        vuiInfo = filter->VuiOut();
    #endif
        //フィルタチェーンに追加
        cufilters.push_back(std::move(filter));
        //パラメータ情報を更新
        m_pLastFilterParam = std::dynamic_pointer_cast<NVEncFilterParam>(param);
        //入力フレーム情報を更新
        inputFrame = param->frameOut;
        m_encFps = param->baseFps;
        return RGY_ERR_NONE;
    }
    //rff
    if (vppType == VppType::CL_RFF) {
        unique_ptr<NVEncFilter> filter(new NVEncFilterRff());
        shared_ptr<NVEncFilterParamRff> param(new NVEncFilterParamRff());
        param->rff      = inputParam->vpp.rff;
        param->frameIn  = inputFrame;
        param->frameOut = inputFrame;
        param->baseFps  = m_encFps;
        param->inFps    = m_inputFps;
        param->timebase = m_outputTimebase;
        param->outFilename = inputParam->common.outputFilename;
        param->bOutOverwrite = true;
        NVEncCtxAutoLock(cxtlock(m_dev->vidCtxLock()));
        auto sts = filter->init(param, m_pLog);
        if (sts != RGY_ERR_NONE) {
            return sts;
        }
        //フィルタチェーンに追加
        cufilters.push_back(std::move(filter));
        //パラメータ情報を更新
        m_pLastFilterParam = std::dynamic_pointer_cast<NVEncFilterParam>(param);
        //入力フレーム情報を更新
        inputFrame = param->frameOut;
        m_encFps = param->baseFps;
        return RGY_ERR_NONE;
    }
    //delogo
    if (vppType == VppType::CL_DELOGO) {
        unique_ptr<NVEncFilter> filter(new NVEncFilterDelogo());
        shared_ptr<NVEncFilterParamDelogo> param(new NVEncFilterParamDelogo());
        param->inputFileName = inputParam->common.inputFilename.c_str();
        param->outputFileName = inputParam->common.outputFilename.c_str();
        param->cudaSchedule  = m_cudaSchedule;
        param->delogo        = inputParam->vpp.delogo;
        param->frameIn = inputFrame;
        param->frameOut = inputFrame;
        param->baseFps = m_encFps;
        param->bOutOverwrite = true;
        NVEncCtxAutoLock(cxtlock(m_dev->vidCtxLock()));
        auto sts = filter->init(param, m_pLog);
        if (sts != RGY_ERR_NONE) {
            return sts;
        }
        //フィルタチェーンに追加
        cufilters.push_back(std::move(filter));
        //パラメータ情報を更新
        m_pLastFilterParam = std::dynamic_pointer_cast<NVEncFilterParam>(param);
        //入力フレーム情報を更新
        inputFrame = param->frameOut;
        m_encFps = param->baseFps;
        return RGY_ERR_NONE;
    }
    //afs
    if (vppType == VppType::CL_AFS) {
        unique_ptr<NVEncFilter> filter(new NVEncFilterAfs());
        shared_ptr<NVEncFilterParamAfs> param(new NVEncFilterParamAfs());
        param->afs = inputParam->vpp.afs;
        param->afs.tb_order = (inputParam->input.picstruct & RGY_PICSTRUCT_TFF) != 0;
        if (inputParam->common.timecode && param->afs.timecode) {
            param->afs.timecode = 2;
        }
        param->frameIn = inputFrame;
        param->frameOut = inputFrame;
        param->inFps = m_inputFps;
        param->inTimebase = m_outputTimebase;
        param->outTimebase = m_outputTimebase;
        param->baseFps = m_encFps;
        param->outFilename = inputParam->common.outputFilename;
        param->cudaSchedule = m_cudaSchedule;
        param->bOutOverwrite = false;
        NVEncCtxAutoLock(cxtlock(m_dev->vidCtxLock()));
        auto sts = filter->init(param, m_pLog);
        if (sts != RGY_ERR_NONE) {
            return sts;
        }
        //フィルタチェーンに追加
        cufilters.push_back(std::move(filter));
        //パラメータ情報を更新
        m_pLastFilterParam = std::dynamic_pointer_cast<NVEncFilterParam>(param);
        //入力フレーム情報を更新
        inputFrame = param->frameOut;
        m_encFps = param->baseFps;
        return RGY_ERR_NONE;
    }
    //nnedi
    if (vppType == VppType::CL_NNEDI) {
        unique_ptr<NVEncFilter> filter(new NVEncFilterNnedi());
        shared_ptr<NVEncFilterParamNnedi> param(new NVEncFilterParamNnedi());
        param->nnedi = inputParam->vpp.nnedi;
        param->compute_capability = m_dev->cc();
        param->frameIn = inputFrame;
        param->frameOut = inputFrame;
        param->baseFps = m_encFps;
        param->timebase = m_outputTimebase;
        param->bOutOverwrite = false;
        NVEncCtxAutoLock(cxtlock(m_dev->vidCtxLock()));
        auto sts = filter->init(param, m_pLog);
        if (sts != RGY_ERR_NONE) {
            return sts;
        }
        //フィルタチェーンに追加
        cufilters.push_back(std::move(filter));
        //パラメータ情報を更新
        m_pLastFilterParam = std::dynamic_pointer_cast<NVEncFilterParam>(param);
        //入力フレーム情報を更新
        inputFrame = param->frameOut;
        m_encFps = param->baseFps;
        return RGY_ERR_NONE;
    }
    //yadif
    if (vppType == VppType::CL_YADIF) {
        unique_ptr<NVEncFilter> filter(new NVEncFilterYadif());
        shared_ptr<NVEncFilterParamYadif> param(new NVEncFilterParamYadif());
        param->yadif = inputParam->vpp.yadif;
        param->frameIn = inputFrame;
        param->frameOut = inputFrame;
        param->baseFps = m_encFps;
        param->timebase = m_outputTimebase;
        param->bOutOverwrite = false;
        NVEncCtxAutoLock(cxtlock(m_dev->vidCtxLock()));
        auto sts = filter->init(param, m_pLog);
        if (sts != RGY_ERR_NONE) {
            return sts;
        }
        //フィルタチェーンに追加
        cufilters.push_back(std::move(filter));
        //パラメータ情報を更新
        m_pLastFilterParam = std::dynamic_pointer_cast<NVEncFilterParam>(param);
        //入力フレーム情報を更新
        inputFrame = param->frameOut;
        m_encFps = param->baseFps;
        return RGY_ERR_NONE;
    }
    //decomb
    if (vppType == VppType::CL_DECOMB) {
        unique_ptr<NVEncFilter> filter(new NVEncFilterDecomb());
        shared_ptr<NVEncFilterParamDecomb> param(new NVEncFilterParamDecomb());
        param->decomb = inputParam->vpp.decomb;
        param->frameIn = inputFrame;
        param->frameOut = inputFrame;
        param->baseFps = m_encFps;
        param->bOutOverwrite = false;
        NVEncCtxAutoLock(cxtlock(m_dev->vidCtxLock()));
        auto sts = filter->init(param, m_pLog);
        if (sts != RGY_ERR_NONE) {
            return sts;
        }
        //フィルタチェーンに追加
        cufilters.push_back(std::move(filter));
        //パラメータ情報を更新
        m_pLastFilterParam = std::dynamic_pointer_cast<NVEncFilterParam>(param);
        //入力フレーム情報を更新
        inputFrame = param->frameOut;
        m_encFps = param->baseFps;
        return RGY_ERR_NONE;
    }
    //decimate
    if (vppType == VppType::CL_DECIMATE) {
        unique_ptr<NVEncFilter> filter(new NVEncFilterDecimate());
        shared_ptr<NVEncFilterParamDecimate> param(new NVEncFilterParamDecimate());
        param->frameIn = inputFrame;
        param->frameOut = inputFrame;
        param->baseFps = m_encFps;
        param->decimate = inputParam->vpp.decimate;
        param->outfilename = inputParam->common.outputFilename;
        param->bOutOverwrite = false;
        NVEncCtxAutoLock(cxtlock(m_dev->vidCtxLock()));
        auto sts = filter->init(param, m_pLog);
        if (sts != RGY_ERR_NONE) {
            return sts;
        }
        //フィルタチェーンに追加
        cufilters.push_back(std::move(filter));
        //パラメータ情報を更新
        m_pLastFilterParam = std::dynamic_pointer_cast<NVEncFilterParam>(param);
        //入力フレーム情報を更新
        inputFrame = param->frameOut;
        m_encFps = param->baseFps;
        return RGY_ERR_NONE;
    }
    //mpdecimate
    if (vppType == VppType::CL_MPDECIMATE) {
        unique_ptr<NVEncFilter> filter(new NVEncFilterMpdecimate());
        shared_ptr<NVEncFilterParamMpdecimate> param(new NVEncFilterParamMpdecimate());
        param->frameIn = inputFrame;
        param->frameOut = inputFrame;
        param->baseFps = m_encFps;
        param->mpdecimate = inputParam->vpp.mpdecimate;
        param->outfilename = inputParam->common.outputFilename;
        param->bOutOverwrite = false;
        NVEncCtxAutoLock(cxtlock(m_dev->vidCtxLock()));
        auto sts = filter->init(param, m_pLog);
        if (sts != RGY_ERR_NONE) {
            return sts;
        }
        //フィルタチェーンに追加
        cufilters.push_back(std::move(filter));
        //パラメータ情報を更新
        m_pLastFilterParam = std::dynamic_pointer_cast<NVEncFilterParam>(param);
        //入力フレーム情報を更新
        inputFrame = param->frameOut;
        m_encFps = param->baseFps;
        return RGY_ERR_NONE;
    }
    //select-every
    if (vppType == VppType::CL_SELECT_EVERY) {
        unique_ptr<NVEncFilter> filter(new NVEncFilterSelectEvery());
        shared_ptr<NVEncFilterParamSelectEvery> param(new NVEncFilterParamSelectEvery());
        param->frameIn  = inputFrame;
        param->frameOut = inputFrame;
        param->baseFps  = m_encFps;
        param->selectevery = inputParam->vpp.selectevery;
        param->bOutOverwrite = false;
        NVEncCtxAutoLock(cxtlock(m_dev->vidCtxLock()));
        auto sts = filter->init(param, m_pLog);
        if (sts != RGY_ERR_NONE) {
            return sts;
        }
        //フィルタチェーンに追加
        cufilters.push_back(std::move(filter));
        //パラメータ情報を更新
        m_pLastFilterParam = std::dynamic_pointer_cast<NVEncFilterParam>(param);
        //入力フレーム情報を更新
        inputFrame = param->frameOut;
        m_encFps = param->baseFps;
        return RGY_ERR_NONE;
    }
    //回転
    if (vppType == VppType::CL_TRANSFORM) {
        unique_ptr<NVEncFilter> filter(new NVEncFilterTransform());
        shared_ptr<NVEncFilterParamTransform> param(new NVEncFilterParamTransform());
        param->trans = inputParam->vpp.transform;
        param->frameIn = inputFrame;
        param->frameOut = inputFrame;
        param->baseFps = m_encFps;
        param->bOutOverwrite = false;
        NVEncCtxAutoLock(cxtlock(m_dev->vidCtxLock()));
        auto sts = filter->init(param, m_pLog);
        if (sts != RGY_ERR_NONE) {
            return sts;
        }
        //フィルタチェーンに追加
        cufilters.push_back(std::move(filter));
        //パラメータ情報を更新
        m_pLastFilterParam = std::dynamic_pointer_cast<NVEncFilterParam>(param);
        //入力フレーム情報を更新
        inputFrame = param->frameOut;
        m_encFps = param->baseFps;
        return RGY_ERR_NONE;
    }
    //ノイズ除去 (convolution3d)
    if (vppType == VppType::CL_CONVOLUTION3D) {
        unique_ptr<NVEncFilter> filter(new NVEncFilterConvolution3d());
        shared_ptr<NVEncFilterParamConvolution3d> param(new NVEncFilterParamConvolution3d());
        param->convolution3d = inputParam->vpp.convolution3d;
        param->frameIn = inputFrame;
        param->frameOut = inputFrame;
        param->baseFps = m_encFps;
        param->bOutOverwrite = false;
        NVEncCtxAutoLock(cxtlock(m_dev->vidCtxLock()));
        auto sts = filter->init(param, m_pLog);
        if (sts != RGY_ERR_NONE) {
            return sts;
        }
        //フィルタチェーンに追加
        cufilters.push_back(std::move(filter));
        //パラメータ情報を更新
        m_pLastFilterParam = std::dynamic_pointer_cast<NVEncFilterParam>(param);
        //入力フレーム情報を更新
        inputFrame = param->frameOut;
        m_encFps = param->baseFps;
        return RGY_ERR_NONE;
    }
    //ノイズ除去 (nvvfx-denoise)
    if (vppType == VppType::NVVFX_DENOISE) {
        unique_ptr<NVEncFilter> filter(new NVEncFilterNvvfxDenoise());
        shared_ptr<NVEncFilterParamNvvfxDenoise> param(new NVEncFilterParamNvvfxDenoise());
        param->nvvfxDenoise = inputParam->vppnv.nvvfxDenoise;
        param->compute_capability = m_dev->cc();
        param->modelDir = inputParam->vppnv.nvvfxModelDir;
        param->vuiInfo = vuiInfo;
        param->frameIn = inputFrame;
        param->frameOut = inputFrame;
        param->baseFps = m_encFps;
        param->bOutOverwrite = false;
        NVEncCtxAutoLock(cxtlock(m_dev->vidCtxLock()));
        auto sts = filter->init(param, m_pLog);
        if (sts != RGY_ERR_NONE) {
            return sts;
        }
        //フィルタチェーンに追加
        cufilters.push_back(std::move(filter));
        //パラメータ情報を更新
        m_pLastFilterParam = std::dynamic_pointer_cast<NVEncFilterParam>(param);
        //入力フレーム情報を更新
        inputFrame = param->frameOut;
        m_encFps = param->baseFps;
        return RGY_ERR_NONE;
    }
    //ノイズ除去 (nvvfx-artifact-reduction)
    if (vppType == VppType::NVVFX_ARTIFACT_REDUCTION) {
        unique_ptr<NVEncFilter> filter(new NVEncFilterNvvfxArtifactReduction());
        shared_ptr<NVEncFilterParamNvvfxArtifactReduction> param(new NVEncFilterParamNvvfxArtifactReduction());
        param->nvvfxArtifactReduction = inputParam->vppnv.nvvfxArtifactReduction;
        param->compute_capability = m_dev->cc();
        param->modelDir = inputParam->vppnv.nvvfxModelDir;
        param->vuiInfo = vuiInfo;
        param->frameIn = inputFrame;
        param->frameOut = inputFrame;
        param->baseFps = m_encFps;
        param->bOutOverwrite = false;
        NVEncCtxAutoLock(cxtlock(m_dev->vidCtxLock()));
        auto sts = filter->init(param, m_pLog);
        if (sts != RGY_ERR_NONE) {
            return sts;
        }
        //フィルタチェーンに追加
        cufilters.push_back(std::move(filter));
        //パラメータ情報を更新
        m_pLastFilterParam = std::dynamic_pointer_cast<NVEncFilterParam>(param);
        //入力フレーム情報を更新
        inputFrame = param->frameOut;
        m_encFps = param->baseFps;
        return RGY_ERR_NONE;
    }
    //ノイズ除去 (smooth)
    if (vppType == VppType::CL_DENOISE_SMOOTH) {
        unique_ptr<NVEncFilter> filter(new NVEncFilterSmooth());
        shared_ptr<NVEncFilterParamSmooth> param(new NVEncFilterParamSmooth());
        param->smooth = inputParam->vpp.smooth;
        param->qpTableRef = m_qpTable.get();
        param->compute_capability = m_dev->cc();
        param->frameIn = inputFrame;
        param->frameOut = inputFrame;
        param->baseFps = m_encFps;
        param->bOutOverwrite = false;
        NVEncCtxAutoLock(cxtlock(m_dev->vidCtxLock()));
        auto sts = filter->init(param, m_pLog);
        if (sts != RGY_ERR_NONE) {
            return sts;
        }
        //フィルタチェーンに追加
        cufilters.push_back(std::move(filter));
        //パラメータ情報を更新
        m_pLastFilterParam = std::dynamic_pointer_cast<NVEncFilterParam>(param);
        if (param->smooth.qp > 0) {
            m_qpTable.reset();
        }
        //入力フレーム情報を更新
        inputFrame = param->frameOut;
        m_encFps = param->baseFps;
        return RGY_ERR_NONE;
    }
    //ノイズ除去 (denoise-dct)
    if (vppType == VppType::CL_DENOISE_DCT) {
        unique_ptr<NVEncFilter> filter(new NVEncFilterDenoiseDct());
        shared_ptr<NVEncFilterParamDenoiseDct> param(new NVEncFilterParamDenoiseDct());
        param->dct = inputParam->vpp.dct;
        param->frameIn = inputFrame;
        param->frameOut = inputFrame;
        param->baseFps = m_encFps;
        param->bOutOverwrite = false;
        NVEncCtxAutoLock(cxtlock(m_dev->vidCtxLock()));
        auto sts = filter->init(param, m_pLog);
        if (sts != RGY_ERR_NONE) {
            return sts;
        }
        //フィルタチェーンに追加
        cufilters.push_back(std::move(filter));
        //パラメータ情報を更新
        m_pLastFilterParam = std::dynamic_pointer_cast<NVEncFilterParam>(param);
        //入力フレーム情報を更新
        inputFrame = param->frameOut;
        m_encFps = param->baseFps;
        return RGY_ERR_NONE;
    }
    //ノイズ除去 (denoise-fft3d)
    if (vppType == VppType::CL_DENOISE_FFT3D) {
        unique_ptr<NVEncFilter> filter(new NVEncFilterDenoiseFFT3D());
        shared_ptr<NVEncFilterParamDenoiseFFT3D> param(new NVEncFilterParamDenoiseFFT3D());
        param->fft3d = inputParam->vpp.fft3d;
        param->compute_capability = m_dev->cc();
        param->frameIn = inputFrame;
        param->frameOut = inputFrame;
        param->baseFps = m_encFps;
        param->bOutOverwrite = false;
        NVEncCtxAutoLock(cxtlock(m_dev->vidCtxLock()));
        auto sts = filter->init(param, m_pLog);
        if (sts != RGY_ERR_NONE) {
            return sts;
        }
        //フィルタチェーンに追加
        cufilters.push_back(std::move(filter));
        //パラメータ情報を更新
        m_pLastFilterParam = std::dynamic_pointer_cast<NVEncFilterParam>(param);
        //入力フレーム情報を更新
        inputFrame = param->frameOut;
        m_encFps = param->baseFps;
        return RGY_ERR_NONE;
    }
    //ノイズ除去 (knn)
    if (vppType == VppType::CL_DENOISE_KNN) {
        unique_ptr<NVEncFilter> filter(new NVEncFilterDenoiseKnn());
        shared_ptr<NVEncFilterParamDenoiseKnn> param(new NVEncFilterParamDenoiseKnn());
        param->knn = inputParam->vpp.knn;
        param->frameIn = inputFrame;
        param->frameOut = inputFrame;
        param->baseFps = m_encFps;
        param->bOutOverwrite = false;
        NVEncCtxAutoLock(cxtlock(m_dev->vidCtxLock()));
        auto sts = filter->init(param, m_pLog);
        if (sts != RGY_ERR_NONE) {
            return sts;
        }
        //フィルタチェーンに追加
        cufilters.push_back(std::move(filter));
        //パラメータ情報を更新
        m_pLastFilterParam = std::dynamic_pointer_cast<NVEncFilterParam>(param);
        //入力フレーム情報を更新
        inputFrame = param->frameOut;
        m_encFps = param->baseFps;
        return RGY_ERR_NONE;
    }
    //ノイズ除去 (nlmeans)
    if (vppType == VppType::CL_DENOISE_NLMEANS) {
        unique_ptr<NVEncFilter> filter(new NVEncFilterDenoiseNLMeans());
        shared_ptr<NVEncFilterParamDenoiseNLMeans> param(new NVEncFilterParamDenoiseNLMeans());
        param->nlmeans = inputParam->vpp.nlmeans;
        param->compute_capability = m_dev->cc();
        param->frameIn = inputFrame;
        param->frameOut = inputFrame;
        param->baseFps = m_encFps;
        param->bOutOverwrite = false;
        NVEncCtxAutoLock(cxtlock(m_dev->vidCtxLock()));
        auto sts = filter->init(param, m_pLog);
        if (sts != RGY_ERR_NONE) {
            return sts;
        }
        //フィルタチェーンに追加
        cufilters.push_back(std::move(filter));
        //パラメータ情報を更新
        m_pLastFilterParam = std::dynamic_pointer_cast<NVEncFilterParam>(param);
        //入力フレーム情報を更新
        inputFrame = param->frameOut;
        m_encFps = param->baseFps;
        return RGY_ERR_NONE;
    }
    //ノイズ除去 (pmd)
    if (vppType == VppType::CL_DENOISE_PMD) {
        unique_ptr<NVEncFilter> filter(new NVEncFilterDenoisePmd());
        shared_ptr<NVEncFilterParamDenoisePmd> param(new NVEncFilterParamDenoisePmd());
        param->pmd = inputParam->vpp.pmd;
        param->frameIn = inputFrame;
        param->frameOut = inputFrame;
        param->baseFps = m_encFps;
        param->bOutOverwrite = false;
        NVEncCtxAutoLock(cxtlock(m_dev->vidCtxLock()));
        auto sts = filter->init(param, m_pLog);
        if (sts != RGY_ERR_NONE) {
            return sts;
        }
        //フィルタチェーンに追加
        cufilters.push_back(std::move(filter));
        //パラメータ情報を更新
        m_pLastFilterParam = std::dynamic_pointer_cast<NVEncFilterParam>(param);
        //入力フレーム情報を更新
        inputFrame = param->frameOut;
        m_encFps = param->baseFps;
        return RGY_ERR_NONE;
    }
    //ノイズ除去
    if (vppType == VppType::NPP_GAUSS) {
    #if _M_IX86
        PrintMes(RGY_LOG_ERROR, _T("gauss denoise filter not supported in x86.\n"));
        return RGY_ERR_UNSUPPORTED;
    #else
        unique_ptr<NVEncFilter> filterGauss(new NVEncFilterDenoiseGauss());
        shared_ptr<NVEncFilterParamGaussDenoise> param(new NVEncFilterParamGaussDenoise());
        param->masksize = inputParam->vppnv.gaussMaskSize;
        param->frameIn = inputFrame;
        param->frameOut = inputFrame;
        param->baseFps = m_encFps;
        param->bOutOverwrite = false;
        NVEncCtxAutoLock(cxtlock(m_dev->vidCtxLock()));
        auto sts = filterGauss->init(param, m_pLog);
        if (sts != RGY_ERR_NONE) {
            return sts;
        }
        //フィルタチェーンに追加
        cufilters.push_back(std::move(filterGauss));
        //パラメータ情報を更新
        m_pLastFilterParam = std::dynamic_pointer_cast<NVEncFilterParam>(param);
        //入力フレーム情報を更新
        inputFrame = param->frameOut;
        m_encFps = param->baseFps;
    #endif
        return RGY_ERR_NONE;
    }
    //字幕焼きこみ
    if (vppType == VppType::CL_SUBBURN) {
        for (const auto& subburn : inputParam->vpp.subburn) {
            if (!subburn.enable)
        #if ENABLE_AVSW_READER
            if (subburn.filename.length() > 0
                && m_trimParam.list.size() > 0) {
                PrintMes(RGY_LOG_ERROR, _T("--vpp-subburn with input as file cannot be used with --trim.\n"));
                return RGY_ERR_UNSUPPORTED;
            }
            unique_ptr<NVEncFilter> filter(new NVEncFilterSubburn());
            shared_ptr<NVEncFilterParamSubburn> param(new NVEncFilterParamSubburn());
            param->subburn = subburn;
            if (m_timestampPassThrough) {
                param->subburn.vid_ts_offset = false;
            }

            auto pAVCodecReader = std::dynamic_pointer_cast<RGYInputAvcodec>(m_pFileReader);
            if (pAVCodecReader != nullptr) {
                param->videoInputStream = pAVCodecReader->GetInputVideoStream();
                param->videoInputFirstKeyPts = pAVCodecReader->GetVideoFirstKeyPts();
                for (const auto &stream : pAVCodecReader->GetInputStreamInfo()) {
                    if (stream.trackId == trackFullID(AVMEDIA_TYPE_SUBTITLE, param->subburn.trackId)) {
                        param->streamIn = stream;
                        break;
                    }
                }
                param->attachmentStreams = pAVCodecReader->GetInputAttachmentStreams();
            }
            param->videoInfo = m_pFileReader->GetInputFrameInfo();
            if (param->subburn.trackId != 0 && param->streamIn.stream == nullptr) {
                PrintMes(RGY_LOG_WARN, _T("Could not find subtitle track #%d, vpp-subburn for track #%d will be disabled.\n"),
                    param->subburn.trackId, param->subburn.trackId);
            } else {
                param->bOutOverwrite = true;
                param->videoOutTimebase = av_make_q(m_outputTimebase);
                param->frameIn = inputFrame;
                param->frameOut = inputFrame;
                param->baseFps = m_encFps;
                param->crop = inputParam->input.crop;
                param->poolPkt = m_poolPkt.get();
                NVEncCtxAutoLock(cxtlock(m_dev->vidCtxLock()));
                auto sts = filter->init(param, m_pLog);
                if (sts != RGY_ERR_NONE) {
                    return sts;
                }
                //フィルタチェーンに追加
                cufilters.push_back(std::move(filter));
                //パラメータ情報を更新
                m_pLastFilterParam = std::dynamic_pointer_cast<NVEncFilterParam>(param);
                //入力フレーム情報を更新
                inputFrame = param->frameOut;
                m_encFps = param->baseFps;
            }
        #else
            PrintMes(RGY_LOG_ERROR, _T("--vpp-subburn not supported in this build.\n"));
            return RGY_ERR_UNSUPPORTED;
        #endif
        }
        return RGY_ERR_NONE;
    }
    //libplacebo-shader
    if (vppType == VppType::CL_LIBPLACEBO_SHADER) {
        for (const auto& shader : inputParam->vpp.libplacebo_shader) {
            unique_ptr<NVEncFilter> filter(new NVEncFilterLibplaceboShader());
            shared_ptr<NVEncFilterParamLibplaceboShader> param(new NVEncFilterParamLibplaceboShader());
            param->shader = shader;
            param->frameIn = inputFrame;
            param->frameOut = inputFrame;
            if (shader.width > 0) {
                param->frameOut.width = shader.width;
            }
            if (shader.height > 0) {
                param->frameOut.height = shader.height;
            }
            param->dx11 = m_dev->dx11();
            param->vk = m_dev->vulkan();
            param->baseFps = m_encFps;
            param->bOutOverwrite = false;
            NVEncCtxAutoLock(cxtlock(m_dev->vidCtxLock()));
            auto sts = filter->init(param, m_pLog);
            if (sts != RGY_ERR_NONE) {
                return sts;
            }
            //フィルタチェーンに追加
            cufilters.push_back(std::move(filter));
            //パラメータ情報を更新
            m_pLastFilterParam = std::dynamic_pointer_cast<NVEncFilterParam>(param);
            //入力フレーム情報を更新
            inputFrame = param->frameOut;
            m_encFps = param->baseFps;
        }
        return RGY_ERR_NONE;
    }

    //リサイズ
    if (vppType == VppType::CL_RESIZE) {
        unique_ptr<NVEncFilter> filterCrop(new NVEncFilterResize());
        shared_ptr<NVEncFilterParamResize> param(new NVEncFilterParamResize());
        if (inputParam->vpp.resize_algo == RGY_VPP_RESIZE_AUTO) {
            param->interp = (resize.first < inputFrame.width && resize.second < inputFrame.height)
                ? RGY_VPP_RESIZE_BICUBIC   // 縮小時
                : RGY_VPP_RESIZE_SPLINE36; // 拡大時
        } else if (inputParam->vpp.resize_algo <= RGY_VPP_RESIZE_OPENCL_CUDA_MAX) {
            param->interp = inputParam->vpp.resize_algo;
        } else {
            param->interp = inputParam->vpp.resize_algo;
        }
        if (isNvvfxResizeFiter(inputParam->vpp.resize_algo)) {
            param->nvvfxSuperRes = std::make_shared<NVEncFilterParamNvvfxSuperRes>();
            param->nvvfxSuperRes->nvvfxSuperRes = inputParam->vppnv.nvvfxSuperRes;
            param->nvvfxSuperRes->compute_capability = m_dev->cc();
            param->nvvfxSuperRes->modelDir = inputParam->vppnv.nvvfxModelDir;
            param->nvvfxSuperRes->vuiInfo = vuiInfo;
        } else if (isNgxResizeFiter(inputParam->vpp.resize_algo)) {
            param->ngxvsr = std::make_shared<NVEncFilterParamNGXVSR>();
            param->ngxvsr->ngxvsr = inputParam->vppnv.ngxVSR;
            param->ngxvsr->compute_capability = m_dev->cc();
            param->ngxvsr->dx11 = m_dev->dx11();
            param->ngxvsr->vui = vuiInfo;
        } else if (isLibplaceboResizeFiter(inputParam->vpp.resize_algo)) {
            param->libplaceboResample = std::make_shared<NVEncFilterParamLibplaceboResample>();
            param->libplaceboResample->resample = inputParam->vpp.resize_libplacebo;
            param->libplaceboResample->vui = vuiInfo;
            param->libplaceboResample->dx11 = m_dev->dx11();
            param->libplaceboResample->vk = m_dev->vulkan();
            param->libplaceboResample->resize_algo = inputParam->vpp.resize_algo;
        }
        param->frameIn = inputFrame;
        param->frameOut = inputFrame;
        param->frameOut.width = resize.first;
        param->frameOut.height = resize.second;
        param->baseFps = m_encFps;
        param->bOutOverwrite = false;
        NVEncCtxAutoLock(cxtlock(m_dev->vidCtxLock()));
        auto sts = filterCrop->init(param, m_pLog);
        if (sts != RGY_ERR_NONE) {
            return sts;
        }
        //フィルタチェーンに追加
        cufilters.push_back(std::move(filterCrop));
        //パラメータ情報を更新
        m_pLastFilterParam = std::dynamic_pointer_cast<NVEncFilterParam>(param);
        //入力フレーム情報を更新
        inputFrame = param->frameOut;
        m_encFps = param->baseFps;
        return RGY_ERR_NONE;
    }
    //unsharp
    if (vppType == VppType::CL_UNSHARP) {
        unique_ptr<NVEncFilter> filterUnsharp(new NVEncFilterUnsharp());
        shared_ptr<NVEncFilterParamUnsharp> param(new NVEncFilterParamUnsharp());
        param->unsharp = inputParam->vpp.unsharp;
        param->frameIn = inputFrame;
        param->frameOut = inputFrame;
        param->baseFps = m_encFps;
        param->bOutOverwrite = false;
        NVEncCtxAutoLock(cxtlock(m_dev->vidCtxLock()));
        auto sts = filterUnsharp->init(param, m_pLog);
        if (sts != RGY_ERR_NONE) {
            return sts;
        }
        //フィルタチェーンに追加
        cufilters.push_back(std::move(filterUnsharp));
        //パラメータ情報を更新
        m_pLastFilterParam = std::dynamic_pointer_cast<NVEncFilterParam>(param);
        //入力フレーム情報を更新
        inputFrame = param->frameOut;
        m_encFps = param->baseFps;
        return RGY_ERR_NONE;
    }
    //edgelevel
    if (vppType == VppType::CL_EDGELEVEL) {
        unique_ptr<NVEncFilter> filterEdgelevel(new NVEncFilterEdgelevel());
        shared_ptr<NVEncFilterParamEdgelevel> param(new NVEncFilterParamEdgelevel());
        param->edgelevel = inputParam->vpp.edgelevel;
        param->frameIn = inputFrame;
        param->frameOut = inputFrame;
        param->baseFps = m_encFps;
        param->bOutOverwrite = false;
        NVEncCtxAutoLock(cxtlock(m_dev->vidCtxLock()));
        auto sts = filterEdgelevel->init(param, m_pLog);
        if (sts != RGY_ERR_NONE) {
            return sts;
        }
        //フィルタチェーンに追加
        cufilters.push_back(std::move(filterEdgelevel));
        //パラメータ情報を更新
        m_pLastFilterParam = std::dynamic_pointer_cast<NVEncFilterParam>(param);
        //入力フレーム情報を更新
        inputFrame = param->frameOut;
        m_encFps = param->baseFps;
        return RGY_ERR_NONE;
    }
    //warpsharp
    if (vppType == VppType::CL_WARPSHARP) {
        unique_ptr<NVEncFilter> filterWarpsharp(new NVEncFilterWarpsharp());
        shared_ptr<NVEncFilterParamWarpsharp> param(new NVEncFilterParamWarpsharp());
        param->warpsharp = inputParam->vpp.warpsharp;
        param->frameIn = inputFrame;
        param->frameOut = inputFrame;
        param->baseFps = m_encFps;
        param->bOutOverwrite = false;
        NVEncCtxAutoLock(cxtlock(m_dev->vidCtxLock()));
        auto sts = filterWarpsharp->init(param, m_pLog);
        if (sts != RGY_ERR_NONE) {
            return sts;
        }
        //フィルタチェーンに追加
        cufilters.push_back(std::move(filterWarpsharp));
        //パラメータ情報を更新
        m_pLastFilterParam = std::dynamic_pointer_cast<NVEncFilterParam>(param);
        //入力フレーム情報を更新
        inputFrame = param->frameOut;
        m_encFps = param->baseFps;
        return RGY_ERR_NONE;
    }
    //curves
    if (vppType == VppType::CL_CURVES) {
        unique_ptr<NVEncFilter> filter(new NVEncFilterCurves());
        shared_ptr<NVEncFilterParamCurves> param(new NVEncFilterParamCurves());
        param->curves = inputParam->vpp.curves;
        param->frameIn = inputFrame;
        param->frameOut = inputFrame;
        param->baseFps = m_encFps;
        param->bOutOverwrite = true;
        NVEncCtxAutoLock(cxtlock(m_dev->vidCtxLock()));
        auto sts = filter->init(param, m_pLog);
        if (sts != RGY_ERR_NONE) {
            return sts;
        }
        //フィルタチェーンに追加
        cufilters.push_back(std::move(filter));
        //パラメータ情報を更新
        m_pLastFilterParam = std::dynamic_pointer_cast<NVEncFilterParam>(param);
        //入力フレーム情報を更新
        inputFrame = param->frameOut;
        m_encFps = param->baseFps;
        return RGY_ERR_NONE;
    }
    //tweak
    if (vppType == VppType::CL_TWEAK) {
        unique_ptr<NVEncFilter> filterEq(new NVEncFilterTweak());
        shared_ptr<NVEncFilterParamTweak> param(new NVEncFilterParamTweak());
        param->tweak = inputParam->vpp.tweak;
        param->frameIn = inputFrame;
        param->frameOut = inputFrame;
        param->vui = vuiInfo;
        param->baseFps = m_encFps;
        param->bOutOverwrite = true;
        NVEncCtxAutoLock(cxtlock(m_dev->vidCtxLock()));
        auto sts = filterEq->init(param, m_pLog);
        if (sts != RGY_ERR_NONE) {
            return sts;
        }
        //フィルタチェーンに追加
        cufilters.push_back(std::move(filterEq));
        //パラメータ情報を更新
        m_pLastFilterParam = std::dynamic_pointer_cast<NVEncFilterParam>(param);
        //入力フレーム情報を更新
        inputFrame = param->frameOut;
        m_encFps = param->baseFps;
        return RGY_ERR_NONE;
    }
    //deband
    if (vppType == VppType::CL_DEBAND) {
        unique_ptr<NVEncFilter> filter(new NVEncFilterDeband());
        shared_ptr<NVEncFilterParamDeband> param(new NVEncFilterParamDeband());
        param->deband = inputParam->vpp.deband;
        param->frameIn = inputFrame;
        param->frameOut = inputFrame;
        param->baseFps = m_encFps;
        param->bOutOverwrite = false;
        NVEncCtxAutoLock(cxtlock(m_dev->vidCtxLock()));
        auto sts = filter->init(param, m_pLog);
        if (sts != RGY_ERR_NONE) {
            return sts;
        }
        //フィルタチェーンに追加
        cufilters.push_back(std::move(filter));
        //パラメータ情報を更新
        m_pLastFilterParam = std::dynamic_pointer_cast<NVEncFilterParam>(param);
        //入力フレーム情報を更新
        inputFrame = param->frameOut;
        m_encFps = param->baseFps;
        return RGY_ERR_NONE;
    }
    // libplacebo deband
    if (vppType == VppType::CL_LIBPLACEBO_DEBAND) {
        unique_ptr<NVEncFilter> filter(new NVEncFilterLibplaceboDeband());
        shared_ptr<NVEncFilterParamLibplaceboDeband> param(new NVEncFilterParamLibplaceboDeband());
        param->deband = inputParam->vpp.libplacebo_deband;
        param->frameIn = inputFrame;
        param->frameOut = inputFrame;
        param->dx11 = m_dev->dx11();
        param->vk = m_dev->vulkan();
        param->baseFps = m_encFps;
        param->bOutOverwrite = false;
        NVEncCtxAutoLock(cxtlock(m_dev->vidCtxLock()));
        auto sts = filter->init(param, m_pLog);
        if (sts != RGY_ERR_NONE) {
            return sts;
        }
        //フィルタチェーンに追加
        cufilters.push_back(std::move(filter));
        //パラメータ情報を更新
        m_pLastFilterParam = std::dynamic_pointer_cast<NVEncFilterParam>(param);
        //入力フレーム情報を更新
        inputFrame = param->frameOut;
        m_encFps = param->baseFps;
        return RGY_ERR_NONE;
    }
    //padding
    if (vppType == VppType::CL_PAD) {
        unique_ptr<NVEncFilter> filter(new NVEncFilterPad());
        shared_ptr<NVEncFilterParamPad> param(new NVEncFilterParamPad());
        param->pad = inputParam->vpp.pad;
        param->frameIn = inputFrame;
        param->frameOut = inputFrame;
        param->frameOut.width += (param->pad.left + param->pad.right);
        param->frameOut.height += (param->pad.top + param->pad.bottom);
        param->encoderCsp = GetEncoderCSP(inputParam);
        param->baseFps = m_encFps;
        param->bOutOverwrite = false;
        NVEncCtxAutoLock(cxtlock(m_dev->vidCtxLock()));
        auto sts = filter->init(param, m_pLog);
        if (sts != RGY_ERR_NONE) {
            return sts;
        }
        //フィルタチェーンに追加
        cufilters.push_back(std::move(filter));
        //パラメータ情報を更新
        m_pLastFilterParam = std::dynamic_pointer_cast<NVEncFilterParam>(param);
        //入力フレーム情報を更新
        inputFrame = param->frameOut;
        m_encFps = param->baseFps;
        return RGY_ERR_NONE;
    }
    //overlay
    if (vppType == VppType::CL_OVERLAY) {
        for (const auto& overlay : inputParam->vpp.overlay) {
            if (overlay.enable) {
                unique_ptr<NVEncFilter> filter(new NVEncFilterOverlay());
                shared_ptr<NVEncFilterParamOverlay> param(new NVEncFilterParamOverlay());
                param->overlay = overlay;
                param->threadPrm = inputParam->ctrl.threadParams.get(RGYThreadType::CSP);
                param->frameIn = inputFrame;
                param->frameOut = inputFrame;
                param->baseFps = m_encFps;
                param->bOutOverwrite = false;
                NVEncCtxAutoLock(cxtlock(m_dev->vidCtxLock()));
                auto sts = filter->init(param, m_pLog);
                if (sts != RGY_ERR_NONE) {
                    return sts;
                }
                //フィルタチェーンに追加
                cufilters.push_back(std::move(filter));
                //パラメータ情報を更新
                m_pLastFilterParam = std::dynamic_pointer_cast<NVEncFilterParam>(param);
                //入力フレーム情報を更新
                inputFrame = param->frameOut;
                m_encFps = param->baseFps;
            }
        }
        return RGY_ERR_NONE;
    }
    //truehdr
    if (vppType == VppType::NGX_TRUEHDR) {
        unique_ptr<NVEncFilterNGXTrueHDR> filter(new NVEncFilterNGXTrueHDR());
        shared_ptr<NVEncFilterParamNGXTrueHDR> param(new NVEncFilterParamNGXTrueHDR());
        param->trueHDR = inputParam->vppnv.ngxTrueHDR;
        param->compute_capability = m_dev->cc();
        param->dx11 = m_dev->dx11();
        param->vui = vuiInfo;
        param->frameIn = inputFrame;
        param->frameOut = inputFrame;
        param->baseFps = m_encFps;
        param->bOutOverwrite = false;
        NVEncCtxAutoLock(cxtlock(m_dev->vidCtxLock()));
        auto sts = filter->init(param, m_pLog);
        if (sts != RGY_ERR_NONE) {
            return sts;
        }
        vuiInfo = filter->VuiOut();
        //フィルタチェーンに追加
        cufilters.push_back(std::move(filter));
        //パラメータ情報を更新
        m_pLastFilterParam = std::dynamic_pointer_cast<NVEncFilterParam>(param);
        //入力フレーム情報を更新
        inputFrame = param->frameOut;
        m_encFps = param->baseFps;
        return RGY_ERR_NONE;
    }
    // fruc
    if (vppType == VppType::CL_FRUC) {
        unique_ptr<NVEncFilter> filter(new NVEncFilterNVOFFRUC());
        shared_ptr<NVEncFilterParamNVOFFRUC> param(new NVEncFilterParamNVOFFRUC());
        param->fruc = inputParam->vpp.fruc;
        param->compute_capability = m_dev->cc();
        param->frameIn = inputFrame;
        param->frameOut = inputFrame;
        param->baseFps = m_encFps;
        param->bOutOverwrite = false;
        param->timebase = m_outputTimebase;
        NVEncCtxAutoLock(cxtlock(m_dev->vidCtxLock()));
        auto sts = filter->init(param, m_pLog);
        if (sts != RGY_ERR_NONE) {
            return sts;
        }
        //フィルタチェーンに追加
        cufilters.push_back(std::move(filter));
        //パラメータ情報を更新
        m_pLastFilterParam = std::dynamic_pointer_cast<NVEncFilterParam>(param);
        //入力フレーム情報を更新
        inputFrame = param->frameOut;
        m_encFps = param->baseFps;
        return RGY_ERR_NONE;
    }
    return RGY_ERR_UNSUPPORTED;
}

bool NVEncCore::VppRffEnabled() {
    //vpp-rffが使用されているか
    for (auto& vppblock : m_vpFilters) {
        for (auto& filter : vppblock.vppnv) {
            if (typeid(*filter) == typeid(NVEncFilterRff)) {
                return true;
            }
        }
    }
    return false;
}

bool NVEncCore::VppAfsRffAware() {
    //vpp-afsのrffが使用されているか
    for (auto& vppblock : m_vpFilters) {
        for (auto& filter : vppblock.vppnv) {
            if (typeid(*filter) == typeid(NVEncFilterAfs)) {
                auto afs_prm = reinterpret_cast<const NVEncFilterParamAfs *>(filter->GetFilterParam());
                if (afs_prm != nullptr) {
                    return afs_prm->afs.rff;
                }
            }
        }
    }
    return false;
}

RGY_ERR NVEncCore::InitSsimFilter(const InEncodeVideoParam *inputParam) {
    if (inputParam->common.metric.enabled() && m_dev->encoder()) {
        //デコードのほうもチェックしてあげないといけない
        const auto& cuvid_csp = m_dev->cuvid_csp();
        if (cuvid_csp.count(inputParam->codec_rgy) == 0) {
            PrintMes(RGY_LOG_ERROR, _T("GPU #%d (%s) does not support %s decoding required for ssim/psnr/vmaf calculation.\n"), m_dev->id(), m_dev->name().c_str(), CodecToStr(inputParam->codec_rgy).c_str());
            return RGY_ERR_DEVICE_NOT_FOUND;
        }
        const auto targetInfo = videooutputinfo(m_stCodecGUID, GetEncBufferFormat(inputParam), false,
            m_uEncWidth, m_uEncHeight,
            &m_stEncConfig, m_stPicStruct,
            std::make_pair(m_sar.n(), m_sar.d()),
            m_encFps);
        const auto& cuvid_codec_csp = cuvid_csp.at(inputParam->codec_rgy);
        if (std::find(cuvid_codec_csp.begin(), cuvid_codec_csp.end(), targetInfo.csp) == cuvid_codec_csp.end()) {
            PrintMes(RGY_LOG_ERROR, _T("GPU #%d (%s) does not support %s %s decoding required for ssim/psnr/vmaf calculation.\n"), m_dev->id(), m_dev->name().c_str(), CodecToStr(inputParam->codec_rgy).c_str(), RGY_CSP_NAMES[targetInfo.csp]);
            return RGY_ERR_DEVICE_NOT_FOUND;
        }

        unique_ptr<NVEncFilterSsim> filterSsim(new NVEncFilterSsim());
        shared_ptr<NVEncFilterParamSsim> param(new NVEncFilterParamSsim());
        param->input = targetInfo;
        param->input.srcWidth = m_uEncWidth;
        param->input.srcHeight = m_uEncHeight;
        param->frameIn = m_pLastFilterParam->frameOut;
        param->frameOut = param->frameIn;
        param->frameOut.csp = param->input.csp;
        param->frameIn.mem_type = RGY_MEM_TYPE_GPU;
        param->frameOut.mem_type = RGY_MEM_TYPE_GPU;
        param->bOutOverwrite = false;
        param->streamtimebase = m_outputTimebase;
        param->vidctxlock = m_dev->vidCtxLock();
        param->threadParamCompare = inputParam->ctrl.threadParams.get(RGYThreadType::VIDEO_QUALITY);
        param->ssim = inputParam->common.metric.ssim;
        param->psnr = inputParam->common.metric.psnr;
        param->vmaf = inputParam->common.metric.vmaf;
        param->deviceId = m_nDeviceId;
        auto sts = filterSsim->init(param, m_pLog);
        if (sts != RGY_ERR_NONE) {
            return sts;
        }
        m_videoQualityMetric = std::move(filterSsim);
    }
    return RGY_ERR_NONE;
}

RGY_ERR NVEncCore::CheckDynamicRCParams(std::vector<NVEncRCParam>& dynamicRC) {
    if (dynamicRC.size() == 0) {
        return RGY_ERR_NONE;
    }
    std::sort(dynamicRC.begin(), dynamicRC.end(), [](const NVEncRCParam& a, const NVEncRCParam& b) {
        return (a.start == b.start) ? a.end < b.end : a.start < b.start;
    });
    std::for_each(dynamicRC.begin(), dynamicRC.end(), [](NVEncRCParam &a) {
        if (a.end <= 0) {
            a.end = TRIM_MAX;
        }
    });
    int id = 0;
    for (auto a : dynamicRC) {
        if (a.start < id) {
            PrintMes(RGY_LOG_ERROR, _T("Invalid sequence of frame ID in --dynamic-rc.\n"));
            PrintMes(RGY_LOG_ERROR, _T("%s\n"), printParams(dynamicRC).c_str());
            return RGY_ERR_INVALID_PARAM;
        }
        id = a.start;
        if (a.end > 0 && a.end < id) {
            PrintMes(RGY_LOG_ERROR, _T("Invalid sequence of frame ID in --dynamic-rc.\n"));
            PrintMes(RGY_LOG_ERROR, _T("%s\n"), printParams(dynamicRC).c_str());
            return RGY_ERR_INVALID_PARAM;
        }
    }
    PrintMes(RGY_LOG_DEBUG, _T("%s\n"), printParams(dynamicRC).c_str());
    m_dynamicRC = dynamicRC;
    m_appliedDynamicRC = DYNAMIC_PARAM_NOT_SELECTED;
    return RGY_ERR_NONE;
}

int NVEncCore::GetEncoderBitDepth(const InEncodeVideoParam *inputParam) const {
    if (inputParam->codec_rgy == RGY_CODEC_H264) {
        return 8;
    }
    return inputParam->outputDepth;
}

bool NVEncCore::encodeIsHighBitDepth(const InEncodeVideoParam *inputParam) const {
    return GetEncoderBitDepth(inputParam) > 8;
}

NV_ENC_BUFFER_FORMAT NVEncCore::GetEncBufferFormat(const InEncodeVideoParam *inputParam) const {
    NV_ENC_BUFFER_FORMAT encBufferFormat;
    if (encodeIsHighBitDepth(inputParam)) {
        switch (RGY_CSP_CHROMA_FORMAT[inputParam->outputCsp]) {
        case RGY_CHROMAFMT_YUV444:
        case RGY_CHROMAFMT_RGB:
        case RGY_CHROMAFMT_RGB_PACKED:
            encBufferFormat = NV_ENC_BUFFER_FORMAT_YUV444_10BIT;
            break;
        case RGY_CHROMAFMT_YUV422:
            encBufferFormat = NV_ENC_BUFFER_FORMAT_P210;
            break;
        default:
            encBufferFormat = NV_ENC_BUFFER_FORMAT_YUV420_10BIT;
            break;
        }
    } else {
        switch (RGY_CSP_CHROMA_FORMAT[inputParam->outputCsp]) {
        case RGY_CHROMAFMT_YUV444:
        case RGY_CHROMAFMT_RGB:
        case RGY_CHROMAFMT_RGB_PACKED:
            encBufferFormat = NV_ENC_BUFFER_FORMAT_YUV444_PL;
            break;
        case RGY_CHROMAFMT_YUV422:
            encBufferFormat = NV_ENC_BUFFER_FORMAT_NV16;
            break;
        default:
            encBufferFormat = NV_ENC_BUFFER_FORMAT_NV12_PL;
            break;
        }
    }
    return encBufferFormat;
}

DeviceCodecCsp NVEncCore::GetHWDecCodecCsp(const bool skipHWDecodeCheck, std::vector<std::unique_ptr<NVGPUInfo>>& gpuList) {
    DeviceCodecCsp HWDecCodecCsp;
    for (const auto &gpu : gpuList) {
        HWDecCodecCsp.push_back(std::make_pair(gpu->id(), (skipHWDecodeCheck) ? getHWDecCodecCsp(skipHWDecodeCheck) : gpu->cuvid_csp()));
    }
    return HWDecCodecCsp;
}

RGY_ERR NVEncCore::Init(InEncodeVideoParam *inputParam) {
    auto sts = InitLog(inputParam);
    if (sts != RGY_ERR_NONE) {
        return sts;
    }

#if ENABLE_VULKAN
    if (inputParam->ctrl.enableVulkan == RGYParamInitVulkan::TargetVendor) {
        setenv("VK_LOADER_DRIVERS_SELECT", "*nvidia*", 1);
    }
#endif // ENABLE_VULKAN

    //m_pDeviceを初期化
    if (!check_if_nvcuda_dll_available()) {
        PrintMes(RGY_LOG_ERROR,
            FOR_AUO ? _T("CUDAが使用できないため、NVEncによるエンコードが行えません。(check_if_nvcuda_dll_available)\n") : _T("CUDA not available.\n"));
        return err_to_rgy(NV_ENC_ERR_UNSUPPORTED_DEVICE);
    }
    m_nDeviceId = inputParam->deviceID;
    m_cudaSchedule = (CUctx_flags)(inputParam->cudaSchedule & CU_CTX_SCHED_MASK);

    if ((sts = InitCuda()) != RGY_ERR_NONE) {
        PrintMes(RGY_LOG_ERROR, FOR_AUO ? _T("Cudaの初期化に失敗しました。\n") : _T("Failed to initialize CUDA.\n"));
        return sts;
    }
    PrintMes(RGY_LOG_DEBUG, _T("InitCuda: Success.\n"));

    //入力などにも渡すため、まずはインスタンスを作っておく必要がある
    m_pPerfMonitor = std::make_unique<CPerfMonitor>();
 #if ENABLE_PERF_COUNTER
    if (inputParam->ctrl.parallelEnc.isChild()) {
        // 子スレッドの場合はパフォーマンスカウンタは親と共有するので初期化不要
        m_pPerfMonitor->setCounter(inputParam->ctrl.parallelEnc.sendData->perfCounter);
    } else {
        m_pPerfMonitor->runCounterThread();
    }
 #endif

    if (const auto affinity = inputParam->ctrl.threadParams.get(RGYThreadType::PROCESS).affinity; affinity.mode != RGYThreadAffinityMode::ALL) {
        SetProcessAffinityMask(GetCurrentProcess(), affinity.getMask());
        PrintMes(RGY_LOG_DEBUG, _T("Set Process Affinity Mask: %s (0x%llx).\n"), affinity.to_string().c_str(), affinity.getMask());
    }
    if (const auto priority = inputParam->ctrl.threadParams.get(RGYThreadType::PROCESS).priority; priority != RGYThreadPriority::Normal) {
        SetPriorityClass(GetCurrentProcess(), inputParam->ctrl.threadParams.get(RGYThreadType::PROCESS).getPriorityCalss());
        PrintMes(RGY_LOG_DEBUG, _T("Set Process priority: %s.\n"), rgy_thread_priority_mode_to_str(priority));
    }

    m_nAVSyncMode = inputParam->common.AVSyncMode;
    m_nProcSpeedLimit = inputParam->ctrl.procSpeedLimit;
    m_taskPerfMonitor = inputParam->ctrl.taskPerfMonitor;
    m_videoIgnoreTimestampError = inputParam->common.videoIgnoreTimestampError;
    if (inputParam->ctrl.lowLatency) {
        m_pipelineDepth = 1;
    }
    if (inputParam->codec_rgy == RGY_CODEC_RAW || inputParam->codec_rgy == RGY_CODEC_AVCODEC) {
        if (invalid_with_raw_out(inputParam->common, m_pLog)) {
            return err_to_rgy(NV_ENC_ERR_INVALID_CALL);
        }
    }
    
    DeviceCodecCsp HWDecCodecCsp;
    auto deviceInfoCache = std::make_shared<RGYDeviceInfoCache>();
    if (sts = deviceInfoCache->loadCacheFile(); sts != RGY_ERR_NONE) {
        if (sts == RGY_ERR_FILE_OPEN) { // ファイルは存在するが開けない
            deviceInfoCache.reset(); // キャッシュの存在を無視して進める
        }
    } else {
        HWDecCodecCsp = deviceInfoCache->getDeviceDecCodecCsp();
        PrintMes(RGY_LOG_DEBUG, _T("HW dec codec csp support read from cache file.\n"));
    }
    std::vector<std::unique_ptr<NVGPUInfo>> gpuList;
    auto getDevIdName = [&gpuList]() {
        std::map<int, std::string> devIdName;
        for (const auto& gpu : gpuList) {
            devIdName[gpu->id()] = tchar_to_string(gpu->name());
        }
        return devIdName;
    };
    if (deviceInfoCache
        && (deviceInfoCache->getDeviceIds().size() == 0
            ||deviceInfoCache->getDeviceIds().size() != HWDecCodecCsp.size())) {
        if (RGY_ERR_NONE != (sts = InitDeviceList(gpuList, m_cudaSchedule, !inputParam->disableDX11, inputParam->ctrl.enableVulkan, inputParam->ctrl.skipHWDecodeCheck, inputParam->disableNVML))) {
            PrintMes(RGY_LOG_ERROR, _T("Failed to initialize devices.\n"));
            return sts;
        }
        PrintMes(RGY_LOG_DEBUG, _T("InitDeviceList: Success.\n"));
        HWDecCodecCsp = GetHWDecCodecCsp(inputParam->ctrl.skipHWDecodeCheck, gpuList);
        deviceInfoCache->setDecCodecCsp(getDevIdName(), HWDecCodecCsp);
        deviceInfoCache->saveCacheFile();
        PrintMes(RGY_LOG_DEBUG, _T("HW dec codec csp support saved to cache file.\n"));
    }

    //入力ファイルを開き、入力情報も取得
    //デコーダが使用できるか確認する必要があるので、先にGPU関係の情報を取得しておく必要がある
    auto input_ret = std::async(std::launch::async, [&] {
        auto sts = InitInput(inputParam, HWDecCodecCsp);
        if (sts == RGY_ERR_NONE) {
            inputParam->applyDOVIProfile(m_pFileReader->getInputDOVIProfile());
        }
        return sts;
    });

    if (gpuList.size() == 0) {
        if (RGY_ERR_NONE != (sts = InitDeviceList(gpuList, m_cudaSchedule, !inputParam->disableDX11, inputParam->ctrl.enableVulkan, inputParam->ctrl.skipHWDecodeCheck, inputParam->disableNVML))) {
            PrintMes(RGY_LOG_ERROR, _T("Failed to initialize devices.\n"));
            return sts;
        }
        if (deviceInfoCache) {
            deviceInfoCache->setDeviceIds(getDevIdName());
        }
    }

    if ((sts = input_ret.get()) < RGY_ERR_NONE) return sts;
    PrintMes(RGY_LOG_DEBUG, _T("InitInput: Success.\n"));

    // 並列動作の子は読み込みが終了したらすぐに並列動作を呼び出し
    // ただし、親-子間のデータやり取りを少し遅らせる場合(delayChildSync)は親と同じタイミングで処理する
    if (inputParam->ctrl.parallelEnc.isChild() && !inputParam->ctrl.parallelEnc.delayChildSync) {
        sts = InitParallelEncode(inputParam, gpuList);
        if (sts < RGY_ERR_NONE) return sts;
    }

    std::unique_ptr<RGYDeviceUsageLockManager> devUsageLock;
    if (gpuList.size() > 1) {
        m_deviceUsage = std::make_unique<RGYDeviceUsage>();
        devUsageLock = m_deviceUsage->lock(); // ロックは親プロセス側でとる
    }

    if (inputParam->tuningInfo == NV_ENC_TUNING_INFO_LOSSLESS) {
        inputParam->lossless = true;
    }
    bool bOutputHighBitDepth = encodeIsHighBitDepth(inputParam);
    if (inputParam->lossless && inputParam->losslessIgnoreInputCsp == 0) {
        const auto inputFrameInfo = m_pFileReader->GetInputFrameInfo();
        //入力ファイルの情報をもとに修正
        //なるべくオリジナルに沿ったものにエンコードする
        switch (RGY_CSP_CHROMA_FORMAT[inputFrameInfo.csp]) {
        case RGY_CHROMAFMT_RGB:
        case RGY_CHROMAFMT_RGB_PACKED:
            inputParam->outputCsp = RGY_CSP_RGB;
            break;
        case RGY_CHROMAFMT_YUV444:
            inputParam->outputCsp = RGY_CSP_YUV444;
            break;
        case RGY_CHROMAFMT_YUV422:
            inputParam->outputCsp = RGY_CSP_YUV422;
            break;
        default:
            inputParam->outputCsp = RGY_CSP_YV12;
            break;
        }

        if (RGY_CSP_BIT_DEPTH[inputFrameInfo.csp] > 8) {
            if (inputParam->codec_rgy != RGY_CODEC_H264) {
                inputParam->outputDepth = 10; // 8bitより上のときはとりあえず10bitで出力
                PrintMes(RGY_LOG_DEBUG, _T("Set bitdepth to %d for lossless encoding.\n"), inputParam->outputDepth);
                bOutputHighBitDepth = encodeIsHighBitDepth(inputParam); // 更新
            }
        }
    }
    if (rgy_csp_has_alpha(inputParam->outputCsp)) {
        if (rgy_csp_alpha_base(inputParam->input.csp) == RGY_CSP_NA) {
            PrintMes(RGY_LOG_ERROR, _T("Input file does not have alpha channel.\n"));
            return RGY_ERR_UNSUPPORTED;
        }
        if (inputParam->codec_rgy != RGY_CODEC_HEVC) {
            PrintMes(RGY_LOG_ERROR, _T("alpha channel encoding only supported with HEVC encoding.\n"));
            return RGY_ERR_UNSUPPORTED;
        }
        if (bOutputHighBitDepth || RGY_CSP_CHROMA_FORMAT[inputParam->outputCsp] != RGY_CHROMAFMT_YUV420) {
            PrintMes(RGY_LOG_ERROR, _T("alpha channel encoding only supported with 8bit YUVA420 HEVC encoding.\n"));
            return RGY_ERR_UNSUPPORTED;
        }
    }
    m_rgbAsYUV444 = RGY_CSP_CHROMA_FORMAT[inputParam->outputCsp] == RGY_CHROMAFMT_RGB || RGY_CSP_CHROMA_FORMAT[inputParam->outputCsp] == RGY_CHROMAFMT_RGB_PACKED;

    //リスト中のGPUのうち、まずは指定されたHWエンコードが可能なもののみを選択
    if ((sts = CheckGPUListByEncoder(gpuList, inputParam)) != RGY_ERR_NONE) {
        PrintMes(RGY_LOG_ERROR, _T("Unknown error occurred during checking GPU.\n"));
        return sts;
    }
    PrintMes(RGY_LOG_DEBUG, _T("CheckGPUListByEncoder: Success, found %d GPU.\n"), (int)gpuList.size());

    m_devNames.clear();
    for (const auto& dev : gpuList) {
        m_devNames.push_back(dev->infostr());
    }

    //使用するGPUの優先順位を決定
    if ((sts = GPUAutoSelect(gpuList, inputParam, devUsageLock.get())) != RGY_ERR_NONE) {
        PrintMes(RGY_LOG_ERROR, FOR_AUO ? _T("GPUの自動選択に失敗しました。\n") : _T("Failed to select gpu.\n"));
        return sts;
    }
    PrintMes(RGY_LOG_DEBUG, _T("GPUAutoSelect: Success.\n"));

    if ((sts = CheckDynamicRCParams(inputParam->dynamicRC)) != RGY_ERR_NONE) {
        PrintMes(RGY_LOG_DEBUG, _T("Error in dynamic rate control params.\n"));
        return sts;
    }

    if (gpuList.size() > 1 && m_nDeviceId < 0) {
        m_nDeviceId = gpuList.front()->id();
        PrintMes(RGY_LOG_DEBUG, _T("device #%d (%s) selected.\n"), gpuList.front()->id(), gpuList.front()->name().c_str());
    }
    if (m_deviceUsage) {
        const int devID = gpuList.front()->id();
        // 登録を解除するプロセスを起動
        const auto [err_run_proc, child_pid] = m_deviceUsage->startProcessMonitor(devID);
        if (err_run_proc == RGY_ERR_NONE) {
            // プロセスが起動できたら、その子プロセスのIDを登録する
            m_deviceUsage->add(devID, child_pid, devUsageLock.get());
        }
    }
    devUsageLock.reset();

    if ((sts = InitDevice(gpuList, inputParam)) != RGY_ERR_NONE) {
        PrintMes(RGY_LOG_ERROR, FOR_AUO ? _T("NVENCのインスタンス作成に失敗しました。\n") : _T("Failed to create NVENC instance.\n"));
        return sts;
    }
    PrintMes(RGY_LOG_DEBUG, _T("InitNVEncInstance: Success.\n"));
    if (deviceInfoCache) {
        deviceInfoCache->updateCacheFile();
    }

    { //出力解像度の自動設定 decoderの初期化前に実施
        if (inputParam->input.dstWidth < 0 && inputParam->input.dstHeight < 0) {
            PrintMes(RGY_LOG_ERROR, _T("Either one of output resolution must be positive value.\n"));
            return RGY_ERR_UNSUPPORTED;
        }
        auto outpar = std::make_pair(inputParam->par[0], inputParam->par[1]);
        if ((!inputParam->par[0] || !inputParam->par[1]) //SAR比の指定がない
            && inputParam->input.sar[0] && inputParam->input.sar[1] //入力側からSAR比を取得ずみ
            && (inputParam->input.dstWidth == inputParam->input.srcWidth && inputParam->input.dstHeight == inputParam->input.srcHeight)) {//リサイズは行われない
            outpar = std::make_pair(inputParam->input.sar[0], inputParam->input.sar[1]);
        }
        set_auto_resolution(inputParam->input.dstWidth, inputParam->input.dstHeight, outpar.first, outpar.second,
            inputParam->input.srcWidth, inputParam->input.srcHeight, inputParam->input.sar[0], inputParam->input.sar[1], 2, 2, inputParam->inprm.resizeResMode, inputParam->inprm.ignoreSAR, inputParam->input.crop);
    }

    //必要ならデコーダを作成
    if ((sts = InitDecoder(inputParam)) != RGY_ERR_NONE) {
        return sts;
    }
    PrintMes(RGY_LOG_DEBUG, _T("InitDecoder: Success.\n"));

    //必要ならフィルターを作成
    if (InitFilters(inputParam) != RGY_ERR_NONE) {
        return err_to_rgy(NV_ENC_ERR_INVALID_PARAM);
    }
    PrintMes(RGY_LOG_DEBUG, _T("InitFilters: Success.\n"));

    if (inputParam->ctrl.lowLatency) {
        if (!m_dev->encoder()->checkAPIver(10, 0)) {
            if (inputParam->preset == NVENC_PRESET_DEFAULT) {
                inputParam->preset = NVENC_PRESET_LL;
            } else if (inputParam->preset == NVENC_PRESET_HP) {
                inputParam->preset = NVENC_PRESET_LL_HP;
            } else if (inputParam->preset == NVENC_PRESET_HQ) {
                inputParam->preset = NVENC_PRESET_LL_HQ;
            }
        }
    }
    if ((sts = SetInputParam(inputParam)) != RGY_ERR_NONE) {
        return sts;
    }
    PrintMes(RGY_LOG_DEBUG, _T("SetInputParam: Success.\n"));

    //エンコーダにパラメータを渡し、初期化
    if (m_dev->encoder()) {
        if (RGY_ERR_NONE != (sts = err_to_rgy(m_dev->encoder()->CreateEncoder(&m_stCreateEncodeParams)))) {
            PrintMes(RGY_LOG_ERROR, _T("Failed to create encoder\n%s.\n"), GetEncoderParamsInfo(RGY_LOG_ERROR, false).c_str());
            return sts;
        }
        if (inputParam->cudaStreamOpt) {
            NVEncCtxAutoLock(cxtlock(m_dev->vidCtxLock()));
            if (RGY_ERR_NONE != (sts = err_to_rgy(m_dev->encoder()->NvEncSetIOCudaStreams(m_encRunCtx->streamIn(), m_encRunCtx->streamOut())))) {
                return sts;
            }
        }
    }
    PrintMes(RGY_LOG_DEBUG, _T("CreateEncoder: Success.\n"));

    if ((sts = InitPowerThrottoling(inputParam)) != RGY_ERR_NONE) {
        return sts;
    }
    NV_ENC_BUFFER_FORMAT encBufferFormat = GetEncBufferFormat(inputParam);
#if 0
    //入出力用メモリ確保
    if ((sts = AllocateBufferInputHost(&inputParam->input)) != RGY_ERR_NONE) {
        return sts;
    }

    if ((sts = AllocateBufferEncoder(m_uEncWidth, m_uEncHeight, encBufferFormat, inputParam->alphaChannel)) != RGY_ERR_NONE) {
        return sts;
    }

    if ((sts = AllocateBufferRawOutput(m_uEncWidth, m_uEncHeight, GetRawOutCSP(inputParam))) != RGY_ERR_NONE) {
        return sts;
    }

    PrintMes(RGY_LOG_DEBUG, _T("AllocateIOBuffers: Success.\n"));
#endif
    //エンコーダにパラメータを渡し、初期化
    if ((sts = InitChapters(inputParam)) != RGY_ERR_NONE) {
        return sts;
    }
    PrintMes(RGY_LOG_DEBUG, _T("InitChapters: Success.\n"));

    if (inputParam->common.keyFile.length() > 0) {
        if (m_trimParam.list.size() > 0) {
            PrintMes(RGY_LOG_WARN, _T("--keyfile could not be used with --trim, disabled.\n"));
        } else {
            m_keyFile = read_keyfile(inputParam->common.keyFile);
            if (m_keyFile.size() == 0) {
                PrintMes(RGY_LOG_ERROR, _T("Failed to read keyFile \"%s\".\n"), inputParam->common.keyFile.c_str());
                return RGY_ERR_UNSUPPORTED;
            }
        }
    }

    if ((sts = InitPerfMonitor(inputParam)) != RGY_ERR_NONE) {
        PrintMes(RGY_LOG_ERROR, _T("Faield to initialize performance monitor.\n"));
        return sts;
    }
    PrintMes(RGY_LOG_DEBUG, _T("InitPerfMonitor: Success.\n"));

    // 親はエンコード設定が完了してから並列動作を呼び出し
    if (inputParam->ctrl.parallelEnc.isParent() || (inputParam->ctrl.parallelEnc.isChild() && inputParam->ctrl.parallelEnc.delayChildSync)) {
        sts = InitParallelEncode(inputParam, gpuList);
        if (sts < RGY_ERR_NONE) return sts;
    }

    m_encTimestamp = std::make_unique<RGYTimestamp>(inputParam->common.timestampPassThrough, inputParam->ctrl.parallelEnc.isParent() /*durationは子エンコーダで修正済み*/);
    m_encodeFrameID = 0;

    //出力ファイルを開く
    if ((sts = InitOutput(inputParam, encBufferFormat)) != RGY_ERR_NONE) {
        PrintMes(RGY_LOG_ERROR, FOR_AUO ? _T("出力ファイルのオープンに失敗しました。: \"%s\"\n") : _T("Failed to open output file: \"%s\"\n"), inputParam->common.outputFilename.c_str());
        return sts;
    }
    PrintMes(RGY_LOG_DEBUG, _T("InitOutput: Success.\n"), inputParam->common.outputFilename.c_str());

    if ((sts = InitSsimFilter(inputParam)) != RGY_ERR_NONE) {
        return sts;
    }

    if (RGY_ERR_NONE != (sts = initPipeline(inputParam))) {
        return sts;
    }

    if (RGY_ERR_NONE != (sts = allocatePiplelineFrames(inputParam))) {
        return sts;
    }

    {
        const auto& threadParam = inputParam->ctrl.threadParams.get(RGYThreadType::MAIN);
        threadParam.apply(GetCurrentThread());
        PrintMes(RGY_LOG_DEBUG, _T("Set main thread param: %s.\n"), threadParam.desc().c_str());
    }
    return RGY_ERR_NONE;
}

RGY_ERR NVEncCore::initPipeline(const InEncodeVideoParam *prm) {
    m_pipelineTasks.clear();

    if (m_parallelEnc && m_parallelEnc->id() < 0) {
        // 親プロセスの子プロセスのデータ回収用
        std::unique_ptr<PipelineTaskAudio> taskAudio;
        if (m_pFileWriterListAudio.size() > 0) {
            taskAudio = std::make_unique<PipelineTaskAudio>(m_dev.get(), m_pFileReader.get(), m_AudioReaders, m_pFileWriterListAudio, m_vpFilters, 0, prm->ctrl.threadParams.get(RGYThreadType::AUDIO), m_pLog);
        }
        const auto encOutputTimebase = (ENCODER_QSV) ? to_rgy(HW_NATIVE_TIMEBASE) : m_outputTimebase;
        m_pipelineTasks.push_back(std::make_unique<PipelineTaskParallelEncBitstream>(m_dev.get(), m_pFileReader.get(), m_encTimestamp.get(), m_timecode.get(), m_parallelEnc.get(),
            m_pStatus.get(), m_encFps, encOutputTimebase, taskAudio, 0, prm->ctrl.threadParams.get(RGYThreadType::MAIN), m_pLog));
        return RGY_ERR_NONE;
    }

    // 並列処理時用の終了時刻 (この時刻は含まないようにする) -1の場合は制限なし(最後まで)
    const auto parallelEncEndPts = (m_parallelEnc) ? m_parallelEnc->getVideoEndKeyPts() : -1ll;
    PipelineTaskNVDecode *taskNVDec = nullptr;
    if (m_pDecoder) {
        m_pipelineTasks.push_back(std::make_unique<PipelineTaskNVDecode>(m_dev.get(), m_pDecoder.get(), 0, m_pFileReader.get(), parallelEncEndPts, prm->ctrl.threadParams.get(RGYThreadType::DEC), m_pLog));
        taskNVDec = dynamic_cast<PipelineTaskNVDecode *>(m_pipelineTasks.back().get());
    } else {
        m_pipelineTasks.push_back(std::make_unique<PipelineTaskInput>(m_dev.get(), 1, m_pFileReader.get(), parallelEncEndPts, prm->ctrl.threadParams.get(RGYThreadType::INPUT), m_pLog));
    }
    if (m_pFileWriterListAudio.size() > 0) {
        m_pipelineTasks.push_back(std::make_unique<PipelineTaskAudio>(m_dev.get(), m_pFileReader.get(), m_AudioReaders, m_pFileWriterListAudio, m_vpFilters, 0, prm->ctrl.threadParams.get(RGYThreadType::AUDIO), m_pLog));
    }
    { // checkpts
        const PipelineTaskTrim *taskTrim = nullptr;
        RGYInputAvcodec *pReader = dynamic_cast<RGYInputAvcodec *>(m_pFileReader.get());
        const int64_t outFrameDuration = std::max<int64_t>(1, rational_rescale(1, m_inputFps.inv(), m_outputTimebase)); //固定fpsを仮定した時の1フレームのduration (スケール: m_outputTimebase)
        const auto inputFrameInfo = m_pFileReader->GetInputFrameInfo();
        const auto inputFpsTimebase = rgy_rational<int>((int)inputFrameInfo.fpsD, (int)inputFrameInfo.fpsN);
        const auto srcTimebase = (m_pFileReader->getInputTimebase().n() > 0 && m_pFileReader->getInputTimebase().is_valid()) ? m_pFileReader->getInputTimebase() : inputFpsTimebase;
        if (m_trimParam.list.size() > 0 || prm->common.seekToSec > 0.0f) {
            m_pipelineTasks.push_back(std::make_unique<PipelineTaskTrim>(m_dev.get(), m_trimParam, m_pFileReader.get(), m_parallelEnc.get(),
                srcTimebase, m_outputTimebase, 0, prm->ctrl.threadParams.get(RGYThreadType::MAIN), m_pLog));
            taskTrim = dynamic_cast<PipelineTaskTrim *>(m_pipelineTasks.back().get());
        }
        const bool interlaceAutoDetect = pReader && pReader->GetInputFrameInfo().picstruct == RGY_PICSTRUCT_AUTO;
        m_pipelineTasks.push_back(std::make_unique<PipelineTaskCheckPTS>(m_dev.get(), m_pDecoder.get(), taskNVDec, taskTrim,
            srcTimebase, srcTimebase, m_outputTimebase, outFrameDuration, m_nAVSyncMode, (m_pDecoder) ? m_pDecoder->getDeinterlaceMode() : cudaVideoDeinterlaceMode_Weave,
            m_timestampPassThrough, VppRffEnabled() && m_pFileReader->rffAware(), VppAfsRffAware() && m_pFileReader->rffAware(),
            interlaceAutoDetect, (pReader) ? pReader->GetFramePosList() : nullptr, prm->ctrl.threadParams.get(RGYThreadType::MAIN), m_pLog));
    }

    PipelineTaskCUDAVpp *taskFirstCudaVpp = nullptr;
    PipelineTaskCUDAVpp *taskLastCudaVpp = nullptr;
    for (auto& vppblock : m_vpFilters) {
        m_pipelineTasks.push_back(std::make_unique<PipelineTaskCUDAVpp>(m_dev.get(), vppblock.vppnv, m_videoQualityMetric.get(),
            m_encRunCtx->qEncodeBufferFree(), m_rgbAsYUV444, prm->cudaStreamOpt, prm->cudaMT, 1, prm->ctrl.threadParams.get(RGYThreadType::FILTER), m_pLog));
        auto taskCudaVpp = dynamic_cast<PipelineTaskCUDAVpp *>(m_pipelineTasks.back().get());
        taskLastCudaVpp = taskCudaVpp;
        if (taskFirstCudaVpp == nullptr) {
            taskFirstCudaVpp = taskCudaVpp;
        }
    }
    if (m_pipelineTasks.size() > 0 && taskNVDec) {
        taskNVDec->setVppFrameReleaseData(taskFirstCudaVpp->cuvidFrameReleaseData());
    }
    tstring vppFilterMes;
    for (const auto& vppblock : m_vpFilters) {
        for (const auto& filter : vppblock.vppnv) {
            vppFilterMes += strsprintf(_T("%s%s\n"), (vppFilterMes.length()) ? _T("               ") : _T("Vpp Filters    "), filter->GetInputMessage().c_str());
        }
    }
    PrintMes(RGY_LOG_DEBUG, _T("Filter list\n%s.\n"), vppFilterMes.c_str());
#if 0
    if (m_videoQualityMetric) {
        int prevtask = -1;
        for (int itask = (int)m_pipelineTasks.size() - 1; itask >= 0; itask--) {
            if (!m_pipelineTasks[itask]->isPassThrough()) {
                prevtask = itask;
                break;
            }
        }
        if (m_pipelineTasks[prevtask]->taskType() == PipelineTaskType::INPUT) {
            //inputと直接つながる場合はうまく処理できなくなる(うまく同期がとれない)
            //そこで、CopyのOpenCLフィルタを挟んでその中で処理する
            auto err = createOpenCLCopyFilterForPreVideoMetric(prm);
            if (err != RGY_ERR_NONE) {
                PrintMes(RGY_LOG_ERROR, _T("Failed to join mfx vpp session: %s.\n"), get_err_mes(err));
                return err;
            } else if (m_vpFilters.size() != 1) {
                PrintMes(RGY_LOG_ERROR, _T("m_vpFilters.size() != 1.\n"));
                return RGY_ERR_UNDEFINED_BEHAVIOR;
            }
            m_pipelineTasks.push_back(std::make_unique<PipelineTaskOpenCL>(m_dev.get(), m_vpFilters.front().vppcl, m_videoQualityMetric.get(), m_dev->cl(), 1, m_dev->dx11interlop(), m_pLog));
        } else if (m_pipelineTasks[prevtask]->taskType() == PipelineTaskType::OPENCL) {
            auto taskOpenCL = dynamic_cast<PipelineTaskOpenCL*>(m_pipelineTasks[prevtask].get());
            if (taskOpenCL == nullptr) {
                PrintMes(RGY_LOG_ERROR, _T("taskOpenCL == nullptr.\n"));
                return RGY_ERR_UNDEFINED_BEHAVIOR;
            }
            taskOpenCL->setVideoQualityMetricFilter(m_videoQualityMetric.get());
        } else {
            m_pipelineTasks.push_back(std::make_unique<PipelineTaskVideoQualityMetric>(m_dev.get(), m_videoQualityMetric.get(), 0, m_pLog));
        }
    }
#endif
    if (m_dev->encoder()) {
        m_pipelineTasks.push_back(std::make_unique<PipelineTaskNVEncode>(m_dev.get(), m_encRunCtx.get(),
            codec_guid_enc_to_rgy(m_stCodecGUID), m_uEncWidth, m_uEncHeight, GetEncoderCSP(prm), GetEncoderBitDepth(prm), picstruct_enc_to_rgy(m_stPicStruct),
            m_stEncConfig, m_stCreateEncodeParams, m_timecode.get(), m_encTimestamp.get(), m_outputTimebase, m_hdr10plus.get(), m_dovirpu.get(),
            m_dynamicRC, m_keyFile, m_keyOnChapter, m_Chapters, 1, prm->ctrl.threadParams.get(RGYThreadType::ENC), m_pLog));
        auto taskEnc = dynamic_cast<PipelineTaskNVEncode *>(m_pipelineTasks.back().get());
        if (taskLastCudaVpp) {
            taskLastCudaVpp->setEncodeTask(taskEnc);
        }
    } else {
        m_pipelineTasks.push_back(std::make_unique<PipelineTaskOutputRaw>(m_dev.get(), m_pFileWriter.get(), 1, prm->ctrl.threadParams.get(RGYThreadType::MAIN), m_pLog));
    }

    if (m_pipelineTasks.size() == 0) {
        PrintMes(RGY_LOG_DEBUG, _T("Failed to create pipeline: size = 0.\n"));
        return RGY_ERR_INVALID_OPERATION;
    }

    PrintMes(RGY_LOG_DEBUG, _T("Created pipeline.\n"));
    for (auto& p : m_pipelineTasks) {
        PrintMes(RGY_LOG_DEBUG, _T("  %s\n"), p->print().c_str());
    }
    PrintMes(RGY_LOG_DEBUG, _T("\n"));
    return RGY_ERR_NONE;
}

RGY_ERR NVEncCore::allocatePiplelineFrames(const InEncodeVideoParam *prm) {
    if (m_pipelineTasks.size() == 0) {
        PrintMes(RGY_LOG_ERROR, _T("allocFrames: pipeline not defined!\n"));
        return RGY_ERR_INVALID_CALL;
    }

    const int asyncdepth = 3;
    PrintMes(RGY_LOG_DEBUG, _T("allocFrames: m_nAsyncDepth - %d frames\n"), asyncdepth);

    PipelineTask *t0 = m_pipelineTasks[0].get();
    for (size_t ip = 1; ip < m_pipelineTasks.size(); ip++) {
        if (t0->isPassThrough()) {
            PrintMes(RGY_LOG_ERROR, _T("allocFrames: t0 cannot be path through task!\n"));
            return RGY_ERR_UNSUPPORTED;
        }
        // 次のtaskを見つける
        PipelineTask *t1 = nullptr;
        for (; ip < m_pipelineTasks.size(); ip++) {
            if (!m_pipelineTasks[ip]->isPassThrough()) { // isPassThroughがtrueなtaskはスキップ
                t1 = m_pipelineTasks[ip].get();
                break;
            }
        }
        if (t1 == nullptr) {
            PrintMes(RGY_LOG_ERROR, _T("AllocFrames: invalid pipeline, t1 not found!\n"));
            return RGY_ERR_UNSUPPORTED;
        }
        PrintMes(RGY_LOG_DEBUG, _T("AllocFrames: %s-%s\n"), t0->print().c_str(), t1->print().c_str());

        const auto t0Alloc = t0->requiredSurfOut();
        const auto t1Alloc = t1->requiredSurfIn();
        int t0RequestNumFrame = 0;
        int t1RequestNumFrame = 0;
        RGYFrameInfo allocateFrameInfo;
        if (t0Alloc.has_value() && t1Alloc.has_value()) {
            t0RequestNumFrame = t0Alloc.value().second;
            t1RequestNumFrame = t1Alloc.value().second;
            allocateFrameInfo = (t0->workSurfacesAllocPriority() >= t1->workSurfacesAllocPriority()) ? t0Alloc.value().first : t1Alloc.value().first;
            allocateFrameInfo.width = std::max(t0Alloc.value().first.width, t1Alloc.value().first.width);
            allocateFrameInfo.height = std::max(t0Alloc.value().first.height, t1Alloc.value().first.height);
        } else if (t0Alloc.has_value()) {
            allocateFrameInfo = t0Alloc.value().first;
            t0RequestNumFrame = t0Alloc.value().second;
        } else if (t1Alloc.has_value()) {
            allocateFrameInfo = t1Alloc.value().first;
            t1RequestNumFrame = t1Alloc.value().second;
        } else {
            PrintMes(RGY_LOG_ERROR, _T("AllocFrames: invalid pipeline: cannot get request from either t0 or t1!\n"));
            return RGY_ERR_UNSUPPORTED;
        }
        if (t1->taskType() == PipelineTaskType::NVENC) {
            auto sts = m_encRunCtx->allocEncodeBuffer(m_uEncWidth, m_uEncHeight, GetEncBufferFormat(prm), m_stPicStruct, rgy_csp_has_alpha(prm->outputCsp), m_encodeBufferCount + t0RequestNumFrame + t1RequestNumFrame + asyncdepth + 1);
            if (sts != RGY_ERR_NONE) {
                PrintMes(RGY_LOG_ERROR, _T("AllocFrames:   Failed to allocate frames for %s-%s: %s."), t0->print().c_str(), t1->print().c_str(), get_err_mes(sts));
                return sts;
            }
            t0->setWorkSurfaces(m_encRunCtx->stEncodeBuffer(), m_encRunCtx->qEncodeBufferFree(), m_dev->encoder(), m_rgbAsYUV444);
        } else if (t0->taskType() != PipelineTaskType::NVDEC) {
            const int requestNumFrames = std::max(1, t0RequestNumFrame + t1RequestNumFrame + asyncdepth + 1);
            PrintMes(RGY_LOG_DEBUG, _T("AllocFrames: %s-%s, type: CL, %s %dx%d, request %d frames\n"),
                t0->print().c_str(), t1->print().c_str(), RGY_CSP_NAMES[allocateFrameInfo.csp],
                allocateFrameInfo.width, allocateFrameInfo.height, requestNumFrames);
            auto sts = t0->workSurfacesAllocCUBuf(requestNumFrames, allocateFrameInfo);
            if (sts != RGY_ERR_NONE) {
                PrintMes(RGY_LOG_ERROR, _T("AllocFrames:   Failed to allocate frames for %s-%s: %s."), t0->print().c_str(), t1->print().c_str(), get_err_mes(sts));
                return sts;
            }
            CUDA_DEBUG_SYNC_ERR;
        }
        t0 = t1;
    }
    // 最後がエンコーダでない場合の特例
    if (   m_pipelineTasks.back()->taskType() != PipelineTaskType::NVENC
        && m_pipelineTasks.back()->taskType() != PipelineTaskType::PECOLLECT) {
        const auto t1Alloc = m_pipelineTasks.back()->requiredSurfOut();
        if (t1Alloc.has_value()) {
            RGYFrameInfo allocateFrameInfo = t1Alloc.value().first;
            const int requestNumFrames = asyncdepth + 1;
            PrintMes(RGY_LOG_DEBUG, _T("AllocFrames: %s, type: CL, %s %dx%d, request %d frames\n"),
                m_pipelineTasks.back()->print().c_str(), RGY_CSP_NAMES[allocateFrameInfo.csp],
                allocateFrameInfo.width, allocateFrameInfo.height, requestNumFrames);
            auto sts = m_pipelineTasks.back()->workSurfacesAllocCUBuf(requestNumFrames, allocateFrameInfo);
            if (sts != RGY_ERR_NONE) {
                PrintMes(RGY_LOG_ERROR, _T("AllocFrames:   Failed to allocate frames for %s: %s."), m_pipelineTasks.back()->print().c_str(), get_err_mes(sts));
                return sts;
            }
            CUDA_DEBUG_SYNC_ERR;
        } else if (m_pipelineTasks.back()->taskType() != PipelineTaskType::OUTPUTRAW) {
            PrintMes(RGY_LOG_ERROR, _T("AllocFrames: invalid pipeline: cannot get request from last element!\n"));
            return RGY_ERR_UNSUPPORTED;
        }
    }
    return RGY_ERR_NONE;
}

RGY_ERR NVEncCore::Encode() {
    PrintMes(RGY_LOG_DEBUG, _T("Encode Thread: RunEncode2...\n"));
    if (m_pipelineTasks.size() == 0) {
        PrintMes(RGY_LOG_DEBUG, _T("Failed to create pipeline: size = 0.\n"));
        return RGY_ERR_INVALID_OPERATION;
    }

#if defined(_WIN32) || defined(_WIN64)
    TCHAR handleEvent[256];
    _stprintf_s(handleEvent, NVENCC_ABORT_EVENT, GetCurrentProcessId());
    auto heAbort = std::unique_ptr<std::remove_pointer<HANDLE>::type, handle_deleter>((HANDLE)CreateEvent(nullptr, TRUE, FALSE, handleEvent));
    auto checkAbort = [pabort = m_pAbortByUser, &heAbort]() { return ((pabort != nullptr && *pabort) || WaitForSingleObject(heAbort.get(), 0) == WAIT_OBJECT_0) ? true : false; };
#else
    auto checkAbort = [pabort = m_pAbortByUser]() { return  (pabort != nullptr && *pabort); };
#endif
    m_pStatus->SetStart();

    CProcSpeedControl speedCtrl(m_nProcSpeedLimit);
    for (auto& task : m_pipelineTasks) {
        if (m_taskPerfMonitor) {
            task->setStopWatch();
        }
    }
    std::unique_ptr<PipelineTaskStopWatch> stopwatchOutput;
    if (m_taskPerfMonitor) {
        stopwatchOutput = std::make_unique<PipelineTaskStopWatch>(
            std::vector<tstring>{ _T("") },
            std::vector<tstring>{_T("")}
        );
    }

    auto requireSync = [this]([[maybe_unused]] const size_t itask) {
#if ENCODER_NVENC
        return false;
#else
        if (itask + 1 >= m_pipelineTasks.size()) return true; // 次が最後のタスクの時

        size_t srctask = itask;
        if (m_pipelineTasks[srctask]->isPassThrough()) {
            for (size_t prevtask = srctask - 1; prevtask >= 0; prevtask--) {
                if (!m_pipelineTasks[prevtask]->isPassThrough()) {
                    srctask = prevtask;
                    break;
                }
            }
        }
        for (size_t nexttask = itask + 1; nexttask < m_pipelineTasks.size(); nexttask++) {
            if (!m_pipelineTasks[nexttask]->isPassThrough()) {
                return m_pipelineTasks[srctask]->requireSync(m_pipelineTasks[nexttask]->taskType());
            }
        }
        return true;
#endif
    };

    RGY_ERR err = RGY_ERR_NONE;
    auto setloglevel = [](RGY_ERR err) {
        if (err == RGY_ERR_NONE || err == RGY_ERR_MORE_DATA || err == RGY_ERR_MORE_SURFACE || err == RGY_ERR_MORE_BITSTREAM) return RGY_LOG_DEBUG;
        if (err > RGY_ERR_NONE) return RGY_LOG_WARN;
        return RGY_LOG_ERROR;
    };
    struct PipelineTaskData {
        size_t task;
        std::unique_ptr<PipelineTaskOutput> data;
        PipelineTaskData(size_t t) : task(t), data() {};
        PipelineTaskData(size_t t, std::unique_ptr<PipelineTaskOutput>& d) : task(t), data(std::move(d)) {};
    };
    std::deque<PipelineTaskData> dataqueue;
    {
        auto checkContinue = [&checkAbort](RGY_ERR& err) {
            if (checkAbort() || stdInAbort()) { err = RGY_ERR_ABORTED; return false; }
            return err >= RGY_ERR_NONE || err == RGY_ERR_MORE_DATA || err == RGY_ERR_MORE_SURFACE;
        };
        while (checkContinue(err)) {
            if (dataqueue.empty()) {
                speedCtrl.wait(m_pipelineTasks.front()->outputFrames());
                dataqueue.push_back(PipelineTaskData(0)); // デコード実行用
            }
            while (!dataqueue.empty()) {
                auto d = std::move(dataqueue.front());
                dataqueue.pop_front();
                if (d.task < m_pipelineTasks.size()) {
                    err = RGY_ERR_NONE;
                    auto& task = m_pipelineTasks[d.task];
                    err = task->sendFrame(d.data);
                    if (!checkContinue(err)) {
                        PrintMes(setloglevel(err), _T("Break in task %s: %s.\n"), task->print().c_str(), get_err_mes(err));
                        break;
                    }
                    if (err == RGY_ERR_NONE) {
                        auto output = task->getOutput(requireSync(d.task));
                        if (output.size() == 0) break;
                        //出てきたものは先頭に追加していく
                        std::for_each(output.rbegin(), output.rend(), [itask = d.task, &dataqueue](auto&& o) {
                            dataqueue.push_front(PipelineTaskData(itask + 1, o));
                            });
                    }
                } else { // pipelineの最終的なデータを出力
                    if (stopwatchOutput) stopwatchOutput->set(0);
                    if ((err = d.data->write(m_pFileWriter.get(), m_videoQualityMetric.get())) != RGY_ERR_NONE) {
                        PrintMes(RGY_LOG_ERROR, _T("failed to write output: %s.\n"), get_err_mes(err));
                        break;
                    }
                    if (stopwatchOutput) stopwatchOutput->add(0, 0);
                }
            }
            if (dataqueue.empty()) {
                // taskを前方からひとつづつ出力が残っていないかチェック(主にcheckptsの処理のため)
                for (size_t itask = 0; itask < m_pipelineTasks.size(); itask++) {
                    auto& task = m_pipelineTasks[itask];
                    auto output = task->getOutput(requireSync(itask));
                    if (output.size() > 0) {
                        //出てきたものは先頭に追加していく
                        std::for_each(output.rbegin(), output.rend(), [itask, &dataqueue](auto&& o) {
                            dataqueue.push_front(PipelineTaskData(itask + 1, o));
                            });
                        //checkptsの処理上、でてきたフレームはすぐに後続処理に渡したいのでbreak
                        break;
                    }
                }
            }
        }
    }
    // flush
    PrintMes(RGY_LOG_DEBUG, _T("Flushing start....\n"));
    if (err == RGY_ERR_MORE_BITSTREAM) { // 読み込みの完了を示すフラグ
        err = RGY_ERR_NONE;
        for (auto& task : m_pipelineTasks) {
            task->setOutputMaxQueueSize(0); //flushのため
        }
        auto checkContinue = [&checkAbort](RGY_ERR& err) {
            if (checkAbort()) { err = RGY_ERR_ABORTED; return false; }
            return err >= RGY_ERR_NONE || err == RGY_ERR_MORE_SURFACE;
        };
        for (size_t flushedTaskSend = 0, flushedTaskGet = 0; flushedTaskGet < m_pipelineTasks.size(); ) { // taskを前方からひとつづつflushしていく
            err = RGY_ERR_NONE;
            if (flushedTaskSend == flushedTaskGet) {
                dataqueue.push_back(PipelineTaskData(flushedTaskSend)); //flush用
            }
            while (!dataqueue.empty() && checkContinue(err)) {
                auto d = std::move(dataqueue.front());
                dataqueue.pop_front();
                if (d.task < m_pipelineTasks.size()) {
                    err = RGY_ERR_NONE;
                    auto& task = m_pipelineTasks[d.task];
                    err = task->sendFrame(d.data);
                    if (!checkContinue(err)) {
                        if (d.task == flushedTaskSend) flushedTaskSend++;
                        break;
                    }
                    auto output = task->getOutput(requireSync(d.task));
                    if (output.size() == 0) break;
                    //出てきたものは先頭に追加していく
                    std::for_each(output.rbegin(), output.rend(), [itask = d.task, &dataqueue](auto&& o) {
                        dataqueue.push_front(PipelineTaskData(itask + 1, o));
                        });
                    if (err == RGY_ERR_MORE_DATA) err = RGY_ERR_NONE; //VPPなどでsendFrameがRGY_ERR_MORE_DATAだったが、フレームが出てくる場合がある
                } else { // pipelineの最終的なデータを出力
                    if (stopwatchOutput) stopwatchOutput->set(0);
                    if ((err = d.data->write(m_pFileWriter.get(), m_videoQualityMetric.get())) != RGY_ERR_NONE) {
                        PrintMes(RGY_LOG_ERROR, _T("failed to write output: %s.\n"), get_err_mes(err));
                        break;
                    }
                    if (stopwatchOutput) stopwatchOutput->add(0, 0);
                }
            }
            if (dataqueue.empty()) {
                // taskを前方からひとつづつ出力が残っていないかチェック(主にcheckptsの処理のため)
                for (size_t itask = flushedTaskGet; itask < m_pipelineTasks.size(); itask++) {
                    auto& task = m_pipelineTasks[itask];
                    auto output = task->getOutput(requireSync(itask));
                    if (output.size() > 0) {
                        //出てきたものは先頭に追加していく
                        std::for_each(output.rbegin(), output.rend(), [itask, &dataqueue](auto&& o) {
                            dataqueue.push_front(PipelineTaskData(itask + 1, o));
                            });
                        //checkptsの処理上、でてきたフレームはすぐに後続処理に渡したいのでbreak
                        break;
                    } else if (itask == flushedTaskGet && flushedTaskGet < flushedTaskSend) {
                        flushedTaskGet++;
                    }
                }
            }
        }
    }
    if (!(err == RGY_ERR_NONE || err == RGY_ERR_MORE_DATA || err == RGY_ERR_MORE_SURFACE || err == RGY_ERR_MORE_BITSTREAM || err == RGY_ERR_ABORTED)) {
        for (auto& p : m_pipelineTasks) {
            p->printStatus();
        }
    }
    // エラー終了の場合も含めキューをすべて開放する (m_pipelineTasksを解放する前に行う)
    dataqueue.clear();

    if (m_videoQualityMetric) {
        PrintMes(RGY_LOG_DEBUG, _T("Flushing video quality metric calc.\n"));
        m_videoQualityMetric->addBitstream(nullptr);
    }

    //vpp-perf-monitor
    std::vector<std::pair<tstring, double>> filter_result;
    for (auto& vppblock : m_vpFilters) {
        for (auto& filter : vppblock.vppnv) {
            auto avgtime = filter->GetAvgTimeElapsed();
            if (avgtime > 0.0) {
                filter_result.push_back({ filter->name(), avgtime });
            }
        }
    }
    // MFXのコンポーネントをm_pipelineTasksの解放(フレームの解放)前に実施する
    PrintMes(RGY_LOG_DEBUG, _T("Clear vpp filters...\n"));
    m_vpFilters.clear();
    PrintMes(RGY_LOG_DEBUG, _T("Closing m_pmfxDEC/ENC/VPP...\n"));
    // taskの集計結果を表示
    if (m_taskPerfMonitor) {
        PrintMes(RGY_LOG_INFO, _T("\nTask Performance\n"));
        static const TCHAR *TASK_OUTPUT = _T("OUTPUT");
        const int64_t totalTicks = std::accumulate(m_pipelineTasks.begin(), m_pipelineTasks.end(), 0LL, [](int64_t total, const std::unique_ptr<PipelineTask>& task) {
            return total + task->getStopWatchTotal();
        }) + stopwatchOutput->totalTicks();
        if (totalTicks > 0) {
            const size_t maxWorkStrLenLen = std::max(std::accumulate(m_pipelineTasks.begin(), m_pipelineTasks.end(), (size_t)0, [](size_t maxStrLength, const std::unique_ptr<PipelineTask>& task) {
                return std::max(maxStrLength, task->getStopWatchMaxWorkStrLen());
            }), stopwatchOutput->maxWorkStrLen());
            const size_t maxTaskStrLen = std::max(std::accumulate(m_pipelineTasks.begin(), m_pipelineTasks.end(), (size_t)0, [](size_t maxStrLength, const std::unique_ptr<PipelineTask>& task) {
                return std::max(maxStrLength, _tcslen(getPipelineTaskTypeName(task->taskType())));
            }), _tcslen(TASK_OUTPUT));
            for (auto& task : m_pipelineTasks) {
                task->printStopWatch(totalTicks, maxWorkStrLenLen + maxTaskStrLen - _tcslen(getPipelineTaskTypeName(task->taskType())));
            }
            const auto strlines = split(stopwatchOutput->print(totalTicks, maxWorkStrLenLen + maxTaskStrLen - _tcslen(TASK_OUTPUT)), _T("\n"));
            for (auto& str : strlines) {
                if (str.length() > 0) {
                    PrintMes(RGY_LOG_INFO, _T("%s: %s\n"), TASK_OUTPUT, str.c_str());
                }
            }
        }
    }

    //この中でフレームの解放がなされる
    PrintMes(RGY_LOG_DEBUG, _T("Clear pipeline tasks and allocated frames...\n"));
    // 依存関係があるため、m_pipelineTasksを後ろから解放する
    for (auto it = m_pipelineTasks.rbegin(); it != m_pipelineTasks.rend(); ++it) {
        it->reset();
    }
    m_pipelineTasks.clear();
    PrintMes(RGY_LOG_DEBUG, _T("Waiting for writer to finish...\n"));
    m_pFileWriter->WaitFin();
    PrintMes(RGY_LOG_DEBUG, _T("Close reader...\n"));
    m_pFileReader->Close();
    PrintMes(RGY_LOG_DEBUG, _T("Write results...\n"));
    m_pStatus->WriteResults();
    if (m_videoQualityMetric) {
        PrintMes(RGY_LOG_DEBUG, _T("Write video quality metric results...\n"));
        m_videoQualityMetric->showResult();
    }
    if (m_encRunCtx) {
        PrintMes(RGY_LOG_DEBUG, _T("Release encode buffer....\n"));
        m_encRunCtx->releaseEncodeBuffer();
    }
    if (m_deviceUsage) {
        m_deviceUsage->close();
    }
    if (filter_result.size()) {
        PrintMes(RGY_LOG_INFO, _T("\nVpp Filter Performance\n"));
        const auto max_len = std::accumulate(filter_result.begin(), filter_result.end(), 0u, [](uint32_t max_length, std::pair<tstring, double> info) {
            return std::max(max_length, (uint32_t)info.first.length());
        });
        for (const auto& info : filter_result) {
            tstring str = info.first + _T(":");
            for (uint32_t i = (uint32_t)info.first.length(); i < max_len; i++) {
                str += _T(" ");
            }
            PrintMes(RGY_LOG_INFO, _T("%s %7.1f us\n"), str.c_str(), info.second * 1000.0);
        }
    }
    PrintMes(RGY_LOG_DEBUG, _T("RunEncode2: finished.\n"));
    err = (err == RGY_ERR_NONE || err == RGY_ERR_MORE_DATA || err == RGY_ERR_MORE_SURFACE || err == RGY_ERR_MORE_BITSTREAM || err > RGY_ERR_NONE) ? RGY_ERR_NONE : err;
    if (m_parallelEnc) {
        m_parallelEnc->close(err == RGY_ERR_NONE);
    }
    return err;
}

#if 0
NVENCSTATUS NVEncCore::Encode() {
    NVENCSTATUS nvStatus = NV_ENC_SUCCESS;
    m_pStatus->SetStart();

    const int nEventCount = m_pipelineDepth + CHECK_PTS_MAX_INSERT_FRAMES + 1 + MAX_FILTER_OUTPUT;

    const int cudaEventFlags = (m_cudaSchedule & CU_CTX_SCHED_BLOCKING_SYNC) ? cudaEventBlockingSync : cudaEventDefault;

    //vpp-afsのrffが使用されているか
    const bool vpp_afs_rff_aware = VppAfsRffAware() && m_pFileReader->rffAware();

    //vpp-rffが使用されているか
    const bool vpp_rff = VppRffEnabled() && m_pFileReader->rffAware();

    //エンコードを開始してもよいかを示すcueventの入れ物
    //FrameBufferDataEncに関連付けて使用する
    vector<unique_ptr<cudaEvent_t, cudaevent_deleter>> vEncStartEvents(nEventCount);
    for (uint32_t i = 0; i < vEncStartEvents.size(); i++) {
        //ctxlockした状態でcudaEventCreateを行わないと、イベントは正常に動作しない
        NVEncCtxAutoLock(ctxlock(m_dev->vidCtxLock()));
        vEncStartEvents[i] = std::unique_ptr<cudaEvent_t, cudaevent_deleter>(new cudaEvent_t(), cudaevent_deleter());
        auto cudaret = cudaEventCreateWithFlags(vEncStartEvents[i].get(), cudaEventFlags | cudaEventDisableTiming);
        if (cudaret != cudaSuccess) {
            PrintMes(RGY_LOG_ERROR, _T("Error cudaEventCreate: %d (%s).\n"), cudaret, char_to_tstring(_cudaGetErrorEnum(cudaret)).c_str());
            return NV_ENC_ERR_GENERIC;
        }
    }

    //入力フレームの転送管理用のイベント FrameTransferDataで使用する
    vector<unique_ptr<cudaEvent_t, cudaevent_deleter>> vInFrameTransferFin(std::max(m_pipelineDepth, (int)m_inputHostBuffer.size()));
    for (uint32_t i = 0; i < vInFrameTransferFin.size(); i++) {
        //ctxlockした状態でcudaEventCreateを行わないと、イベントは正常に動作しない
        NVEncCtxAutoLock(ctxlock(m_dev->vidCtxLock()));
        vInFrameTransferFin[i] = std::unique_ptr<cudaEvent_t, cudaevent_deleter>(new cudaEvent_t(), cudaevent_deleter());
        auto cudaret = cudaEventCreateWithFlags(vInFrameTransferFin[i].get(), cudaEventFlags | cudaEventDisableTiming);
        if (cudaret != cudaSuccess) {
            PrintMes(RGY_LOG_ERROR, _T("Error cudaEventCreate: %d (%s).\n"), cudaret, char_to_tstring(_cudaGetErrorEnum(cudaret)).c_str());
            return NV_ENC_ERR_GENERIC;
        }
    }

    //入力フレームの最初の転送に関連するリソースを管理するデータ構造
    struct FrameTransferData {
        cudaEvent_t *eventFin; //転送終了後セットされるイベント (vInFrameTransferFinのものを使用)
        unique_ptr<FrameBufferDataIn> frameData; //入力フレームデータ
        shared_ptr<void> deviceFrame; //入力フレーム(cuvidのmap->unmap用)
    };
    //入力フレームの最初の転送に関連するリソースを管理するキュー
    //転送中のフレームに関するデータを積んでおく
    //転送を行うたびに、キューにeventとともにデータが追加される
    //転送が終了するとeventFinがセットされるので、キューからデータを削除する
    //shared_ptrなどの参照が0になればデストラクタが呼ばれ、リソースが自動的に解放される
    deque<FrameTransferData> dqFrameTransferData;

    //eventFinのセットを待って、キューからデータを削除する
    //nPipelineDepth以上キューに積まれていたら、eventのセットを強制的に待機する
    auto check_inframe_transfer = [&dqFrameTransferData, ctxLock = this->m_dev->vidCtxLock()](const uint32_t nPipelineDepth) {
        const auto queueLength = dqFrameTransferData.size();
        cudaError_t cuerr = cudaSuccess;
        if (queueLength > 0) {
            NVEncCtxAutoLock(ctxlock(ctxLock));
            auto cuevent = *dqFrameTransferData.front().eventFin;
            if (cudaSuccess == (cuerr = (queueLength >= nPipelineDepth) ? cudaEventSynchronize(cuevent) : cudaEventQuery(cuevent))) {
                dqFrameTransferData.pop_front();
            }
            if (cuerr == cudaErrorNotReady) {
                //queueLength < nPipelineDepthならcudaErrorNotReadyがあり得る
                cuerr = cudaSuccess;
            }
        }
        return cuerr;
    };

    bool interlaceAutoDetect = false;
#if ENABLE_AVSW_READER
    const AVStream *streamIn = nullptr;
    RGYInputAvcodec *pReader = dynamic_cast<RGYInputAvcodec *>(m_pFileReader.get());
    if (pReader != nullptr) {
        streamIn = pReader->GetInputVideoStream();
        interlaceAutoDetect = pReader->GetInputFrameInfo().picstruct == RGY_PICSTRUCT_AUTO;
    }
    //cuvidデコード時は、timebaseの分子はかならず1
    const auto srcTimebase = (streamIn) ? rgy_rational<int>((m_pDecoder) ? 1 : streamIn->time_base.num, streamIn->time_base.den) : m_pFileReader->getInputTimebase();

    //streamのindexから必要なwriteへのポインタを返すテーブルを作成
    std::map<int, shared_ptr<RGYOutputAvcodec>> pWriterForAudioStreams;
    for (auto pWriter : m_pFileWriterListAudio) {
        auto pAVCodecWriter = std::dynamic_pointer_cast<RGYOutputAvcodec>(pWriter);
        if (pAVCodecWriter) {
            auto trackIdList = pAVCodecWriter->GetStreamTrackIdList();
            for (auto trackID : trackIdList) {
                pWriterForAudioStreams[trackID] = pAVCodecWriter;
            }
        }
    }
    //streamのtrackIdからパケットを送信するvppフィルタへのポインタを返すテーブルを作成
    std::map<int, NVEncFilter*> pFilterForStreams;
    for (uint32_t ifilter = 0; ifilter < m_vpFilters.size(); ifilter++) {
        const auto targetTrackId = m_vpFilters[ifilter]->targetTrackIdx();
        if (targetTrackId != 0) {
            pFilterForStreams[targetTrackId] = m_vpFilters[ifilter].get();
        }
    }

    auto extract_audio = [&](int inputFrames) {
        auto sts = RGY_ERR_NONE;
        if ((m_pFileWriterListAudio.size() + pFilterForStreams.size()) > 0) {
#if ENABLE_SM_READER
            RGYInputSM *pReaderSM = dynamic_cast<RGYInputSM *>(m_pFileReader.get());
            const int droppedInAviutl = (pReaderSM != nullptr) ? pReaderSM->droppedFrames() : 0;
#else
            const int droppedInAviutl = 0;
#endif
            vector<AVPacket*> packetList = m_pFileReader->GetStreamDataPackets(inputFrames + droppedInAviutl);

            //音声ファイルリーダーからのトラックを結合する
            for (const auto& reader : m_AudioReaders) {
                vector_cat(packetList, reader->GetStreamDataPackets(inputFrames + droppedInAviutl));
            }
            //パケットを各Writerに分配する
            for (uint32_t i = 0; i < packetList.size(); i++) {
                const int nTrackId = pktFlagGetTrackID(packetList[i]);
                const bool sendToFilter = pFilterForStreams.count(nTrackId) > 0;
                const bool sendToWriter = pWriterForAudioStreams.count(nTrackId) > 0;
                AVPacket *pkt = packetList[i];
                if (sendToFilter) {
                    AVPacket *pktToFilter = nullptr;
                    if (sendToWriter) {
                        pktToFilter = av_packet_clone(pkt);
                    } else {
                        std::swap(pktToFilter, pkt);
                    }
                    if ((sts = pFilterForStreams[nTrackId]->addStreamPacket(pktToFilter)) != RGY_ERR_NONE) {
                        return sts;
                    }
                }
                if (sendToWriter) {
                    auto pWriter = pWriterForAudioStreams[nTrackId];
                    if (pWriter == nullptr) {
                        PrintMes(RGY_LOG_ERROR, _T("Invalid writer found for %s track #%d\n"), char_to_tstring(trackMediaTypeStr(nTrackId)).c_str(), trackID(nTrackId));
                        return RGY_ERR_NOT_FOUND;
                    }
                    if ((sts = pWriter->WriteNextPacket(pkt)) != RGY_ERR_NONE) {
                        return sts;
                    }
                    pkt = nullptr;
                }
                if (pkt != nullptr) {
                    PrintMes(RGY_LOG_ERROR, _T("Failed to find writer for %s track #%d\n"), char_to_tstring(trackMediaTypeStr(nTrackId)).c_str(), trackID(nTrackId));
                    return RGY_ERR_NOT_FOUND;
                }
            }
        }
        return RGY_ERR_NONE;
    };

    int64_t hwDecFirstPts = AV_NOPTS_VALUE;
    RGYQueueMPMP<RGYFrameDataMetadata*> queueMetadata;
    queueMetadata.init(256);
    std::thread th_input;
    if (m_pDecoder) {
        th_input = std::thread([this, streamIn, &hwDecFirstPts, &queueMetadata, &nvStatus]() {
            CUresult curesult = CUDA_SUCCESS;
            RGYBitstream bitstream = RGYBitstreamInit();
            RGY_ERR sts = RGY_ERR_NONE;
            for (int i = 0; sts == RGY_ERR_NONE && nvStatus == NV_ENC_SUCCESS && !m_pDecoder->GetError(); i++) {
                if ((  (sts = m_pFileReader->LoadNextFrame(nullptr)) != RGY_ERR_NONE //進捗表示のため
                    || (sts = m_pFileReader->GetNextBitstream(&bitstream)) != RGY_ERR_NONE)
                    && sts != RGY_ERR_MORE_DATA) {
                    nvStatus = NV_ENC_ERR_GENERIC;
                    break;
                }

                for (auto& frameData : bitstream.getFrameDataList()) {
                    if (frameData->dataType() == RGY_FRAME_DATA_HDR10PLUS) {
                        auto ptr = dynamic_cast<RGYFrameDataHDR10plus*>(frameData);
                        if (ptr) {
                            queueMetadata.push(new RGYFrameDataHDR10plus(*ptr));
                        }
                    } else if (frameData->dataType() == RGY_FRAME_DATA_DOVIRPU) {
                        auto ptr = dynamic_cast<RGYFrameDataDOVIRpu*>(frameData);
                        if (ptr) {
                            queueMetadata.push(new RGYFrameDataDOVIRpu(*ptr));
                        }
                    }
                }
                if (hwDecFirstPts == AV_NOPTS_VALUE) {
                    hwDecFirstPts = bitstream.pts();
                }
                PrintMes(RGY_LOG_TRACE, _T("Set packet #%d, size %zu, pts %lld (%s)\n"), i, bitstream.size(),
                    (long long int)bitstream.pts(), getTimestampString(bitstream.pts(), streamIn->time_base).c_str());
                if (CUDA_SUCCESS != (curesult = m_pDecoder->DecodePacket(bitstream.bufptr() + bitstream.offset(), bitstream.size(), bitstream.pts(), streamIn->time_base))) {
                    PrintMes(RGY_LOG_ERROR, _T("Error in DecodePacket: %d (%s).\n"), curesult, char_to_tstring(_cudaGetErrorEnum(curesult)).c_str());
                    return curesult;
                }
                bitstream.setSize(0);
                bitstream.setOffset(0);
                bitstream.clearFrameDataList();
            }
            if (CUDA_SUCCESS != (curesult = m_pDecoder->DecodePacket(nullptr, 0, AV_NOPTS_VALUE, streamIn->time_base))) {
                PrintMes(RGY_LOG_ERROR, _T("Error in DecodePacketFin: %d (%s).\n"), curesult, char_to_tstring(_cudaGetErrorEnum(curesult)).c_str());
            }
            return curesult;
        });
        PrintMes(RGY_LOG_DEBUG, _T("Started Encode thread\n"));
    }
    auto getMetadata = [&queueMetadata](int64_t timestamp) {
        std::vector<std::shared_ptr<RGYFrameData>> frameData;
        RGYFrameDataMetadata *frameDataPtr = nullptr;
        while (queueMetadata.front_copy_no_lock(&frameDataPtr)) {
            if (frameDataPtr->timestamp() < timestamp) {
                queueMetadata.pop();
                delete frameDataPtr;
            } else {
                break;
            }
        }
        size_t queueSize = queueMetadata.size();
        for (uint32_t i = 0; i < queueSize; i++) {
            if (queueMetadata.copy(&frameDataPtr, i, &queueSize)) {
                if (frameDataPtr->timestamp() == timestamp) {
                    if (frameDataPtr->dataType() == RGY_FRAME_DATA_HDR10PLUS) {
                        auto ptr = dynamic_cast<RGYFrameDataHDR10plus*>(frameDataPtr);
                        if (ptr) {
                            frameData.emplace_back(std::make_shared<RGYFrameDataHDR10plus>(*ptr));
                        }
                    } else if (frameDataPtr->dataType() == RGY_FRAME_DATA_DOVIRPU) {
                        auto ptr = dynamic_cast<RGYFrameDataDOVIRpu*>(frameDataPtr);
                        if (ptr) {
                            frameData.emplace_back(std::make_shared<RGYFrameDataDOVIRpu>(*ptr));
                        }
                    }
                }
            }
        }
        return frameData;
    };

    if (m_pPerfMonitor) {
        HANDLE thOutput = NULL;
        HANDLE thInput = NULL;
        HANDLE thAudProc = NULL;
        HANDLE thAudEnc = NULL;
        auto pAVCodecReader = std::dynamic_pointer_cast<RGYInputAvcodec>(m_pFileReader);
        if (pAVCodecReader != nullptr) {
            thInput = pAVCodecReader->getThreadHandleInput();
        }
        auto pAVCodecWriter = std::dynamic_pointer_cast<RGYOutputAvcodec>(m_pFileWriter);
        if (pAVCodecWriter != nullptr) {
            thOutput = pAVCodecWriter->getThreadHandleOutput();
            thAudProc = pAVCodecWriter->getThreadHandleAudProcess();
            thAudEnc = pAVCodecWriter->getThreadHandleAudEncode();
        }
        m_pPerfMonitor->SetThreadHandles((HANDLE)(th_input.native_handle()), thInput, thOutput, thAudProc, thAudEnc);
    }
    int64_t nOutFirstPts = AV_NOPTS_VALUE; //入力のptsに対する補正 (スケール: m_outputTimebase)
#endif //#if ENABLE_AVSW_READER
    int64_t lastTrimFramePts = AV_NOPTS_VALUE; //直前のtrimで落とされたフレームのpts, trimで落とされてない場合はAV_NOPTS_VALUE (スケール: m_outputTimebase)
    int64_t nOutEstimatedPts = 0; //固定fpsを仮定した時のfps (スケール: m_outputTimebase)
    const int64_t nOutFrameDuration = std::max<int64_t>(1, rational_rescale(1, m_inputFps.inv(), m_outputTimebase)); //固定fpsを仮定した時の1フレームのduration (スケール: m_outputTimebase)
    int64_t nLastPts = AV_NOPTS_VALUE;

    auto add_dec_vpp_param = [&](FrameBufferDataIn *pInputFrame, vector<unique_ptr<FrameBufferDataIn>>& vppParams, int64_t outPts, int64_t outDuration) {
        if (pInputFrame->inputIsHost()) {
            pInputFrame->setTimeStamp(outPts);
            pInputFrame->setDuration(outDuration);
            PrintMes(RGY_LOG_TRACE, _T("add_dec_vpp_param[host](%d): outPtsSource %lld, outDuration %d\n"), pInputFrame->getFrameInfo().inputFrameId, outPts, outDuration);
            vppParams.push_back(unique_ptr<FrameBufferDataIn>(new FrameBufferDataIn(*pInputFrame)));
        }
#if ENABLE_AVSW_READER
        else {
            auto deint = m_pDecoder->getDeinterlaceMode();
            auto frameinfo = pInputFrame->getFrameInfo();
            frameinfo.timestamp = outPts;
            frameinfo.duration = outDuration;
            CUVIDPROCPARAMS oVPP = { 0 };
            oVPP.top_field_first = pInputFrame->getCuvidInfo()->top_field_first;
            switch (deint) {
            case cudaVideoDeinterlaceMode_Weave:
                oVPP.progressive_frame = pInputFrame->getCuvidInfo()->progressive_frame;
                oVPP.unpaired_field = 0;// oVPP.progressive_frame;
                vppParams.push_back(unique_ptr<FrameBufferDataIn>(new FrameBufferDataIn(pInputFrame->getCuvidInfo(), oVPP, frameinfo)));
                //PrintMes(RGY_LOG_INFO, _T("pts: %lld, duration %lld, progressive:%d, rff:%d\n"), (lls)frameinfo.timestamp, (lls)frameinfo.duration, oVPP.progressive_frame, (frameinfo.flags & RGY_FRAME_FLAG_RFF) ? 1 : 0);
                PrintMes(RGY_LOG_TRACE, _T("add_dec_vpp_param[dev](%d): outPtsSource %lld, outDuration %d, progressive %d\n"), pInputFrame->getFrameInfo().inputFrameId, frameinfo.timestamp, frameinfo.duration, oVPP.progressive_frame);
                break;
            case cudaVideoDeinterlaceMode_Bob:
                //RFFに関するフラグを念のためクリア
                frameinfo.flags &= (~(RGY_FRAME_FLAG_RFF | RGY_FRAME_FLAG_RFF_COPY | RGY_FRAME_FLAG_RFF_TFF | RGY_FRAME_FLAG_RFF_BFF));
                frameinfo.picstruct = RGY_PICSTRUCT_FRAME;
                pInputFrame->setInterlaceFlag(RGY_PICSTRUCT_FRAME);
                oVPP.progressive_frame = (interlaceAutoDetect) ? pInputFrame->getCuvidInfo()->progressive_frame : 0;
                oVPP.second_field = 0;
                frameinfo.duration >>= 1;
                vppParams.push_back(unique_ptr<FrameBufferDataIn>(new FrameBufferDataIn(pInputFrame->getCuvidInfo(), oVPP, frameinfo)));
                PrintMes(RGY_LOG_TRACE, _T("add_dec_vpp_param[bob](%d): outPtsSource %lld, outDuration %d, progressive %d\n"), pInputFrame->getFrameInfo().inputFrameId, frameinfo.timestamp, frameinfo.duration, oVPP.progressive_frame);
                oVPP.second_field = 1;
                frameinfo.timestamp += frameinfo.duration;
                vppParams.push_back(unique_ptr<FrameBufferDataIn>(new FrameBufferDataIn(pInputFrame->getCuvidInfo(), oVPP, frameinfo)));
                PrintMes(RGY_LOG_TRACE, _T("add_dec_vpp_param[bob](%d): outPtsSource %lld, outDuration %d, progressive %d\n"), pInputFrame->getFrameInfo().inputFrameId, frameinfo.timestamp, frameinfo.duration, oVPP.progressive_frame);
                break;
            case cudaVideoDeinterlaceMode_Adaptive:
                //RFFに関するフラグを念のためクリア
                frameinfo.flags &= (~(RGY_FRAME_FLAG_RFF | RGY_FRAME_FLAG_RFF_COPY | RGY_FRAME_FLAG_RFF_TFF | RGY_FRAME_FLAG_RFF_BFF));
                frameinfo.picstruct = RGY_PICSTRUCT_FRAME;
                pInputFrame->setInterlaceFlag(RGY_PICSTRUCT_FRAME);
                oVPP.progressive_frame = (interlaceAutoDetect) ? pInputFrame->getCuvidInfo()->progressive_frame : 0;
                vppParams.push_back(unique_ptr<FrameBufferDataIn>(new FrameBufferDataIn(pInputFrame->getCuvidInfo(), oVPP, frameinfo)));
                PrintMes(RGY_LOG_TRACE, _T("add_dec_vpp_param[adp](%d): outPtsSource %lld, outDuration %d, progressive %d\n"), pInputFrame->getFrameInfo().inputFrameId, frameinfo.timestamp, frameinfo.duration, oVPP.progressive_frame);
                break;
            default:
                PrintMes(RGY_LOG_ERROR, _T("Unknown Deinterlace mode\n"));
                break;
            }
        }
#endif //#if ENABLE_AVSW_READER
        return;
    };

    int ignoreVideoTimestampErrorCount = 0;
    uint32_t nInputFramePosIdx = UINT32_MAX;
    auto check_pts = [&](FrameBufferDataIn *pInputFrame) {
        vector<unique_ptr<FrameBufferDataIn>> decFrames;
        int64_t outPtsSource = nOutEstimatedPts;
        int64_t outDuration = nOutFrameDuration; //入力fpsに従ったduration
#if ENABLE_AVSW_READER
        if ((srcTimebase.n() > 0 && srcTimebase.is_valid())
            && ((m_nAVSyncMode & (RGY_AVSYNC_VFR | RGY_AVSYNC_FORCE_CFR)) || vpp_rff || vpp_afs_rff_aware || m_timestampPassThrough)) {
            if (pInputFrame->getTimeStamp() < 0) {
                // timestampを修正
                outPtsSource = nOutEstimatedPts;
                pInputFrame->setTimeStamp(rational_rescale(nOutEstimatedPts, m_outputTimebase, srcTimebase));
                pInputFrame->setDuration(rational_rescale(nOutFrameDuration, m_outputTimebase, srcTimebase));
                PrintMes(RGY_LOG_WARN, _T("check_pts: Invalid timestamp from input frame #%d: timestamp %lld, timebase %d/%d, duration %lld.\n"),
                         pInputFrame->getFrameInfo().inputFrameId, pInputFrame->getTimeStamp(), srcTimebase.n(), srcTimebase.d(), pInputFrame->getDuration());
                PrintMes(RGY_LOG_WARN, _T("           use estimated timestamp: timestamp %lld, timebase %d/%d, duration %lld.\n"),
                    outPtsSource, m_outputTimebase.n(), m_outputTimebase.d(), nOutFrameDuration);
            } else {
                //CFR仮定ではなく、オリジナルの時間を見る
                outPtsSource = rational_rescale(pInputFrame->getTimeStamp(), srcTimebase, m_outputTimebase);
                if (pInputFrame->getDuration() > 0) {
                    pInputFrame->setDuration(rational_rescale(pInputFrame->getDuration(), srcTimebase, m_outputTimebase));
                }
            }
        }
        PrintMes(RGY_LOG_TRACE, _T("check_pts(%d): nOutEstimatedPts %lld, outPtsSource %lld, outDuration %d\n"), pInputFrame->getFrameInfo().inputFrameId, nOutEstimatedPts, outPtsSource, outDuration);
        if (nOutFirstPts == AV_NOPTS_VALUE) {
            nOutFirstPts = outPtsSource; //最初のpts
            PrintMes(RGY_LOG_TRACE, _T("check_pts: nOutFirstPts %lld\n"), outPtsSource);
        }
        //最初のptsを0に修正
        if (!m_timestampPassThrough) {
            outPtsSource -= nOutFirstPts;
        }
        if (outPtsSource < 0) {
            PrintMes(RGY_LOG_WARN, _T("check_pts: Invalid timestamp calculated from input frame #%d: timestamp %lld (-%lld), timebase %d/%d.\n"),
                     pInputFrame->getFrameInfo().inputFrameId, outPtsSource, nOutFirstPts, m_outputTimebase.n(), m_outputTimebase.d());
        }

        if ((m_nAVSyncMode & RGY_AVSYNC_VFR) || vpp_rff || vpp_afs_rff_aware) {
            if (vpp_rff || vpp_afs_rff_aware) {
                if (std::abs(outPtsSource - nOutEstimatedPts) >= 32 * nOutFrameDuration) {
                    PrintMes(RGY_LOG_TRACE, _T("check_pts: detected gap %lld, changing offset.\n"), outPtsSource, std::abs(outPtsSource - nOutEstimatedPts));
                    //timestampに一定以上の差があればそれを無視する
                    nOutFirstPts += (outPtsSource - nOutEstimatedPts); //今後の位置合わせのための補正
                    outPtsSource = nOutEstimatedPts;
                    PrintMes(RGY_LOG_TRACE, _T("check_pts:   changed to nOutFirstPts %lld, outPtsSource %lld.\n"), nOutFirstPts, outPtsSource);
                }
                auto ptsDiff = outPtsSource - nOutEstimatedPts;
                if (ptsDiff <= std::min<int64_t>(-1, -1 * nOutFrameDuration * 7 / 8)) {
                    //間引きが必要
                    PrintMes(RGY_LOG_TRACE, _T("check_pts(%d):   skipping frame (vfr)\n"), pInputFrame->getFrameInfo().inputFrameId);
                    return decFrames;
                }
                // 少しのずれはrffによるものとみなし、基準値を修正する
                nOutEstimatedPts = outPtsSource;
            }
            if (streamIn) {
                //cuvidデコード時は、timebaseの分子はかならず1なので、streamIn->time_baseとズレているかもしれないのでオリジナルを計算
                const auto orig_pts = rational_rescale(pInputFrame->getTimeStamp(), srcTimebase, to_rgy(streamIn->time_base));
                //ptsからフレーム情報を取得する
                const auto framePos = pReader->GetFramePosList()->findpts(orig_pts, &nInputFramePosIdx);
                PrintMes(RGY_LOG_TRACE, _T("check_pts(%d):   estimetaed orig_pts %lld, framePos %d\n"), pInputFrame->getFrameInfo().inputFrameId, orig_pts, framePos.poc);
                if (framePos.poc != FRAMEPOS_POC_INVALID && framePos.duration > 0) {
                    //有効な値ならオリジナルのdurationを使用する
                    outDuration = rational_rescale(framePos.duration, to_rgy(streamIn->time_base), m_outputTimebase);
                    PrintMes(RGY_LOG_TRACE, _T("check_pts(%d):   changing duration to original: %d\n"), pInputFrame->getFrameInfo().inputFrameId, outDuration);
                }
            }
        }
        if (m_nAVSyncMode & RGY_AVSYNC_FORCE_CFR) {
            if (std::abs(outPtsSource - nOutEstimatedPts) >= CHECK_PTS_MAX_INSERT_FRAMES * nOutFrameDuration) {
                //timestampに一定以上の差があればそれを無視する
                nOutFirstPts += (outPtsSource - nOutEstimatedPts); //今後の位置合わせのための補正
                outPtsSource = nOutEstimatedPts;
                PrintMes(RGY_LOG_WARN, _T("Big Gap was found between 2 frames, avsync might be corrupted.\n"));
                PrintMes(RGY_LOG_TRACE, _T("check_pts:   changed to nOutFirstPts %lld, outPtsSource %lld.\n"), nOutFirstPts, outPtsSource);
            }
            auto ptsDiff = outPtsSource - nOutEstimatedPts;
            if (ptsDiff <= std::min<int64_t>(-1, -1 * nOutFrameDuration * 7 / 8)) {
                //間引きが必要
                PrintMes(RGY_LOG_TRACE, _T("check_pts(%d):   skipping frame (assume_cfr)\n"), pInputFrame->getFrameInfo().inputFrameId);
                return decFrames;
            }
            while (ptsDiff >= std::max<int64_t>(1, nOutFrameDuration * 7 / 8)) {
                //水増しが必要
                add_dec_vpp_param(pInputFrame, decFrames, nOutEstimatedPts, outDuration);
                nOutEstimatedPts += nOutFrameDuration;
                ptsDiff = outPtsSource - nOutEstimatedPts;
            }
            outPtsSource = nOutEstimatedPts;
        }
        if (nLastPts >= outPtsSource) {
            if (nLastPts - outPtsSource >= 32 * nOutFrameDuration) {
                PrintMes(RGY_LOG_DEBUG, _T("check_pts: previous pts %lld, current pts %lld, estimated pts %lld, nOutFirstPts %lld, changing offset.\n"), nLastPts, outPtsSource, nOutEstimatedPts, nOutFirstPts);
                nOutFirstPts += (outPtsSource - nOutEstimatedPts); //今後の位置合わせのための補正
                outPtsSource = nOutEstimatedPts;
                PrintMes(RGY_LOG_DEBUG, _T("check_pts:   changed to nOutFirstPts %lld, outPtsSource %lld.\n"), nOutFirstPts, outPtsSource);
                ignoreVideoTimestampErrorCount = 0;
            } else {
                if (m_nAVSyncMode & RGY_AVSYNC_FORCE_CFR) {
                    //間引きが必要
                    PrintMes(RGY_LOG_WARN, _T("check_pts(%d): timestamp of video frame is smaller than previous frame, skipping frame: previous pts %lld, current pts %lld.\n"), pInputFrame->getFrameInfo().inputFrameId, nLastPts, outPtsSource);
                    return decFrames;
                } else if (ignoreVideoTimestampErrorCount < m_videoIgnoreTimestampError) {
                    //間引き
                    PrintMes(RGY_LOG_WARN, _T("check_pts(%d): timestamp of video frame is smaller than previous frame, skipping frame: previous pts %lld, current pts %lld.\n"), pInputFrame->getFrameInfo().inputFrameId, nLastPts, outPtsSource);
                    return decFrames;
                } else {
                    const auto origPts = outPtsSource;
                    outPtsSource = nLastPts + std::max<int64_t>(1, nOutFrameDuration / 8);
                    PrintMes(RGY_LOG_WARN, _T("check_pts(%d): timestamp of video frame is smaller than previous frame, changing pts: %lld -> %lld (previous pts %lld).\n"),
                        pInputFrame->getFrameInfo().inputFrameId, origPts, outPtsSource, nLastPts);
                }
                ignoreVideoTimestampErrorCount++;
            }
        } else {
            ignoreVideoTimestampErrorCount = 0;
        }
#endif //#if ENABLE_AVSW_READER
        //次のフレームのptsの予想
        nOutEstimatedPts += outDuration;
        nLastPts = outPtsSource;
        add_dec_vpp_param(pInputFrame, decFrames, outPtsSource, outDuration);
        if (outPtsSource < 0) {
            PrintMes(RGY_LOG_WARN, _T("check_pts: Invalid timestamp set to frame #%d: timestamp %lld (-%lld), timebase %d/%d.\n"),
                     pInputFrame->getFrameInfo().inputFrameId, outPtsSource, nOutFirstPts, m_outputTimebase.n(), m_outputTimebase.d());
        }
        return decFrames;
    };

    auto add_frame_transfer_data = [&](cudaEvent_t *pCudaEvent, unique_ptr<FrameBufferDataIn>& inframe, shared_ptr<void>& deviceFrame) {
        //最初のフィルタ(転送)が終了したことを示すイベント
        //ctxlockのロック外で行う必要があるよう
        auto cudaret = cudaEventRecord(*pCudaEvent);
        if (cudaret != cudaSuccess) {
            PrintMes(RGY_LOG_ERROR, _T("Error cudaEventRecord [add_frame_transfer_data]: %d (%s).\n"), cudaret, char_to_tstring(_cudaGetErrorEnum(cudaret)).c_str());
            return NV_ENC_ERR_GENERIC;
        }

        //関連するリソースに関する参照を移して、キューに積む
        FrameTransferData transferData;
        transferData.eventFin = pCudaEvent; //転送終了のイベント
        transferData.frameData = std::move(inframe); //入力バッファへの参照
        transferData.deviceFrame = std::move(deviceFrame); //cuvidのmap->unmap
        dqFrameTransferData.push_back(std::move(transferData));
        return NV_ENC_SUCCESS;
    };

    auto filter_frame = [&](int& nFilterFrame, unique_ptr<FrameBufferDataIn>& inframe, deque<unique_ptr<FrameBufferDataEnc>>& dqEncFrames, bool& bDrain) {
        cudaMemcpyKind memcpyKind = cudaMemcpyDeviceToDevice;
        RGYFrameInfo frameInfo;
        shared_ptr<void> deviceFrame;
        if (!bDrain) {
            if (inframe->inputIsHost()) {
                memcpyKind = cudaMemcpyHostToDevice;
                frameInfo = inframe->getFrameInfo();
                deviceFrame = shared_ptr<void>(frameInfo.ptr[0], [&](void *ptr) {
                    UNREFERENCED_PARAMETER(ptr);
                    //このメモリはm_inputHostBufferのメモリであり、使いまわすため、解放しない
                });
            }
#if ENABLE_AVSW_READER
            else {
                //ここで前のフレームの転送を必ず待機する
                //ここで待機しないと、vpp-deinterlace bobのときに同一フレームに対する
                //cuvidMapVideoFrameが2連続で発生してしまい、エラーになってしまう
                auto cuerr = check_inframe_transfer(1);
                if (cuerr != cudaSuccess) {
                    PrintMes(RGY_LOG_ERROR, _T("Error cudaEventSynchronize: %d (%s).\n"), cuerr, char_to_tstring(_cudaGetErrorEnum(cuerr)).c_str());
                    return NV_ENC_ERR_GENERIC;
                }
                CUresult curesult = CUDA_SUCCESS;
                CUdeviceptr dMappedFrame = 0;
                memcpyKind = cudaMemcpyDeviceToDevice;
                auto vppinfo = inframe->getVppInfo();
                uint32_t pitch = 0;
                NVEncCtxAutoLock(ctxlock(m_dev->vidCtxLock()));
                if (CUDA_SUCCESS != (curesult = cuvidMapVideoFrame(m_pDecoder->GetDecoder(), inframe->getCuvidInfo()->picture_index, &dMappedFrame, &pitch, &vppinfo))) {
                    PrintMes(RGY_LOG_ERROR, _T("Error cuvidMapVideoFrame: %d (%s).\n"), curesult, char_to_tstring(_cudaGetErrorEnum(curesult)).c_str());
                    return NV_ENC_ERR_GENERIC;
                }
                frameInfo = inframe->getFrameInfo();
                frameInfo.singleAlloc = true;
                frameInfo.pitch[0] = pitch;
                frameInfo.ptr[0] = (uint8_t *)dMappedFrame;
                deviceFrame = shared_ptr<void>(frameInfo.ptr[0], [&](void *ptr) {
                    //ロック内で解放されるので、ここでのさらなるロックは不要
                    //NVEncCtxAutoLock(ctxlock(m_dev->vidCtxLock()));
                    cuvidUnmapVideoFrame(m_pDecoder->GetDecoder(), (CUdeviceptr)ptr);
                });
                PrintMes(RGY_LOG_TRACE, _T("filter_frame(%d): mapped video frame.\n"), nFilterFrame);
            }
#endif //#if ENABLE_AVSW_READER
        }

        deque<std::pair<RGYFrameInfo, uint32_t>> filterframes;
        filterframes.push_back(std::make_pair(frameInfo, 0u));

        while (filterframes.size() > 0 || bDrain) {
            //フィルタリングするならここ
            for (uint32_t ifilter = filterframes.front().second; ifilter < m_vpFilters.size() - 1; ifilter++) {
                // コピーを作ってそれをfilter関数に渡す
                // vpp-rffなどoverwirteするフィルタのときに、filterframes.pop_front -> push がうまく動作しない
                RGYFrameInfo input = filterframes.front().first;

                NVEncCtxAutoLock(ctxlock(m_dev->vidCtxLock()));
                int nOutFrames = 0;
                RGYFrameInfo *outInfo[16] = { 0 };
                auto sts_filter = m_vpFilters[ifilter]->filter(&input, (RGYFrameInfo **)&outInfo, &nOutFrames, cudaStreamDefault);
                if (sts_filter != RGY_ERR_NONE) {
                    PrintMes(RGY_LOG_ERROR, _T("Error while running filter \"%s\".\n"), m_vpFilters[ifilter]->name().c_str());
                    return NV_ENC_ERR_GENERIC;
                }
                if (nOutFrames == 0) {
                    if (bDrain) {
                        filterframes.front().second++;
                        continue;
                    }
                    return NV_ENC_SUCCESS;
                } 
                if (ifilter == 0) { //最初のフィルタなら転送なので、イベントをここでセットする
                    auto pCudaEvent = vInFrameTransferFin[nFilterFrame % vInFrameTransferFin.size()].get();
                    add_frame_transfer_data(pCudaEvent, inframe, deviceFrame);
                }
                bDrain = false; //途中でフレームが出てきたら、drain完了していない

                filterframes.pop_front();
                //最初に出てきたフレームは先頭に追加する
                for (int jframe = nOutFrames - 1; jframe >= 0; jframe--) {
                    filterframes.push_front(std::make_pair(*outInfo[jframe], ifilter + 1));
                }
            }
            if (bDrain) {
                return NV_ENC_SUCCESS; //最後までbDrain = trueなら、drain完了
            }

            //エンコードバッファを取得
            EncodeBuffer *pEncodeBuffer = nullptr;
            if (m_dev->encoder()) {
                pEncodeBuffer = m_EncodeBufferQueue.GetAvailable();
                if (!pEncodeBuffer) {
                    pEncodeBuffer = m_EncodeBufferQueue.GetPending();
                    if (ProcessOutput(pEncodeBuffer) != NV_ENC_SUCCESS) {
                        return NV_ENC_ERR_GENERIC;
                    }
                    if (pEncodeBuffer->stInputBfr.pNV12devPtr) {
                        if (pEncodeBuffer->stInputBfr.hInputSurface) {
                            auto nvencret = m_dev->encoder()->NvEncUnmapInputResource(pEncodeBuffer->stInputBfr.hInputSurface);
                            if (nvencret != NV_ENC_SUCCESS) {
                                PrintMes(RGY_LOG_ERROR, _T("Failed to Unmap input buffer %p: %s\n"), pEncodeBuffer->stInputBfr.hInputSurface, char_to_tstring(_nvencGetErrorEnum(nvencret)).c_str());
                                return nvencret;
                            }
                            pEncodeBuffer->stInputBfr.hInputSurface = nullptr;
                        }
                        if (pEncodeBuffer->stInputBfrAlpha.hInputSurface) {
                            auto nvencret = m_dev->encoder()->NvEncUnmapInputResource(pEncodeBuffer->stInputBfrAlpha.hInputSurface);
                            if (nvencret != NV_ENC_SUCCESS) {
                                PrintMes(RGY_LOG_ERROR, _T("Failed to Unmap input alpha buffer %p: %s\n"), pEncodeBuffer->stInputBfrAlpha.hInputSurface, char_to_tstring(_nvencGetErrorEnum(nvencret)).c_str());
                                return nvencret;
                            }
                            pEncodeBuffer->stInputBfrAlpha.hInputSurface = nullptr;
                        }
                    }
                    pEncodeBuffer = m_EncodeBufferQueue.GetAvailable();
                    if (!pEncodeBuffer) {
                        PrintMes(RGY_LOG_ERROR, _T("Error get enc buffer from queue.\n"));
                        return NV_ENC_ERR_GENERIC;
                    }
                }
            }
            //エンコードバッファにコピー
            //エンコード直前のバッファまで転送を完了し、そのフレームのエンコードを開始できることを示すイベント
            //ここでnFilterFrameをインクリメントする
            {
                NVEncCtxAutoLock(ctxlock(m_dev->vidCtxLock()));
                auto& lastFilter = m_vpFilters[m_vpFilters.size()-1];
                //最後のフィルタはNVEncFilterCspCropでなければならない
                if (typeid(*lastFilter.get()) != typeid(NVEncFilterCspCrop)) {
                    PrintMes(RGY_LOG_ERROR, _T("Last filter setting invalid.\n"));
                    return NV_ENC_ERR_GENERIC;
                }
                int nOutFrames = 0;
                RGYFrameInfo *outInfo[16] = { 0 };
                //エンコードバッファの情報を設定１
                RGYFrameInfo encFrameInfo;
                RGYFrameInfo ssimTarget;
                if (!m_dev->encoder()) {
                    encFrameInfo = m_outputFrameHostRaw->frame;
                } else if (pEncodeBuffer->stInputBfr.pNV12devPtr) {
                    encFrameInfo.ptr[0] = (uint8_t *)pEncodeBuffer->stInputBfr.pNV12devPtr;
                    encFrameInfo.pitch[0] = pEncodeBuffer->stInputBfr.uNV12Stride;
                    encFrameInfo.singleAlloc = true;
                    encFrameInfo.width = pEncodeBuffer->stInputBfr.dwWidth;
                    encFrameInfo.height = pEncodeBuffer->stInputBfr.dwHeight;
                    encFrameInfo.mem_type = RGY_MEM_TYPE_GPU;
                    encFrameInfo.csp = getEncCsp(pEncodeBuffer->stInputBfr.bufferFmt, pEncodeBuffer->stInputBfrAlpha.nvRegisteredResource != nullptr, m_rgbAsYUV444);
                    ssimTarget = encFrameInfo;
                } else {
                    //インタレ保持の場合は、NvEncCreateInputBuffer経由でフレームを渡さないと正常にエンコードできない
                    uint32_t lockedPitch = 0;
                    unsigned char *pInputSurface = nullptr;
                    m_dev->encoder()->NvEncLockInputBuffer(pEncodeBuffer->stInputBfr.hInputSurface, (void**)&pInputSurface, &lockedPitch);
                    encFrameInfo.ptr[0] = (uint8_t *)pInputSurface;
                    encFrameInfo.pitch[0] = lockedPitch;
                    encFrameInfo.singleAlloc = true;
                    encFrameInfo.width = pEncodeBuffer->stInputBfr.dwWidth;
                    encFrameInfo.height = pEncodeBuffer->stInputBfr.dwHeight;
                    encFrameInfo.mem_type = RGY_MEM_TYPE_CPU; //CPU側にフレームデータを戻す
                    encFrameInfo.csp = getEncCsp(pEncodeBuffer->stInputBfr.bufferFmt, pEncodeBuffer->stInputBfrAlpha.nvRegisteredResource != nullptr, m_rgbAsYUV444);
                    ssimTarget = filterframes.front().first;
                }
                //エンコードバッファのポインタを渡す
                outInfo[0] = &encFrameInfo;
                auto sts_filter = lastFilter->filter(&filterframes.front().first, (RGYFrameInfo **)&outInfo, &nOutFrames, cudaStreamDefault);
                filterframes.pop_front();
                if (sts_filter != RGY_ERR_NONE) {
                    PrintMes(RGY_LOG_ERROR, _T("Error while running filter \"%s\".\n"), lastFilter->name().c_str());
                    return NV_ENC_ERR_GENERIC;
                }
                PrintMes(RGY_LOG_TRACE, _T("filter_frame(%d): queued last filter : %s.\n"), nFilterFrame, lastFilter->name().c_str());
                if (m_ssim) {
                    int dummy = 0;
                    m_ssim->filter(&ssimTarget, nullptr, &dummy, cudaStreamDefault);
                }
                auto pCudaEvent = vEncStartEvents[nFilterFrame++ % vEncStartEvents.size()].get();
                auto cudaret = cudaEventRecord(*pCudaEvent);
                if (cudaret != cudaSuccess) {
                    PrintMes(RGY_LOG_ERROR, _T("Error cudaEventRecord: %d (%s).\n"), cudaret, char_to_tstring(_cudaGetErrorEnum(cudaret)).c_str());
                    return NV_ENC_ERR_GENERIC;
                }
                if (m_vpFilters.size() == 1) {
                    //フィルタの数が1のときは、ここが最初(かつ最後)のフィルタであり、転送フィルタである
                    add_frame_transfer_data(pCudaEvent, inframe, deviceFrame);
                }
                if (m_dev->encoder()) {
                    unique_ptr<FrameBufferDataEnc> frameEnc(new FrameBufferDataEnc(RGY_CSP_NV12, encFrameInfo.timestamp, encFrameInfo.duration, encFrameInfo.inputFrameId, pEncodeBuffer, pCudaEvent, encFrameInfo.dataList));
                    dqEncFrames.push_back(std::move(frameEnc));
                } else {
                    cudaEventSynchronize(*pCudaEvent);
                    RGYFrameRef outFrame(encFrameInfo);
                    m_pFileWriter->WriteNextFrame(&outFrame);
                }
            }
        }
        return NV_ENC_SUCCESS;
    };

    auto send_encoder = [&](int& nEncodeFrame, unique_ptr<FrameBufferDataEnc>& encFrame) {
        //エンコーダ用のバッファまで転送が終了するのを待機
        NVEncCtxAutoLock(ctxlock(m_dev->vidCtxLock()));
        if (encFrame->m_pEvent) {
            cudaEventSynchronize(*encFrame->m_pEvent);
        }
        EncodeBuffer *pEncodeBuffer = encFrame->m_pEncodeBuffer;
        if (pEncodeBuffer->stInputBfr.pNV12devPtr) {
            auto nvencret = m_dev->encoder()->NvEncMapInputResource(pEncodeBuffer->stInputBfr.nvRegisteredResource, &pEncodeBuffer->stInputBfr.hInputSurface);
            if (nvencret != NV_ENC_SUCCESS) {
                PrintMes(RGY_LOG_ERROR, _T("Failed to Map input buffer %p\n"), pEncodeBuffer->stInputBfr.hInputSurface);
                return nvencret;
            }
            if (pEncodeBuffer->stInputBfrAlpha.nvRegisteredResource) {
                nvencret = m_dev->encoder()->NvEncMapInputResource(pEncodeBuffer->stInputBfrAlpha.nvRegisteredResource, &pEncodeBuffer->stInputBfrAlpha.hInputSurface);
                if (nvencret != NV_ENC_SUCCESS) {
                    PrintMes(RGY_LOG_ERROR, _T("Failed to Map input buffer %p\n"), pEncodeBuffer->stInputBfrAlpha.hInputSurface);
                    return nvencret;
                }
            }
        } else {
            m_dev->encoder()->NvEncUnlockInputBuffer(pEncodeBuffer->stInputBfr.hInputSurface);
        }
        return NvEncEncodeFrame(pEncodeBuffer, nEncodeFrame++, encFrame->m_timestamp, encFrame->m_duration, encFrame->m_inputFrameId, encFrame->m_frameDataList);
    };

#define NV_ENC_ERR_ABORT ((NVENCSTATUS)-1)
    unique_ptr<FrameBufferDataIn> dummyFrame;
    CProcSpeedControl speedCtrl(m_nProcSpeedLimit);
    deque<unique_ptr<FrameBufferDataIn>> dqInFrames;
    deque<unique_ptr<FrameBufferDataEnc>> dqEncFrames;
    int nEncodeFrames = 0;
    bool bInputEmpty = false;
    bool bFilterEmpty = false;
    for (int nInputFrame = 0, nFilterFrame = 0; nvStatus == NV_ENC_SUCCESS && !bInputEmpty && !bFilterEmpty; ) {
        if ((m_pAbortByUser && *m_pAbortByUser) || stdInAbort()) {
            nvStatus = NV_ENC_ERR_ABORT;
            break;
        }
        speedCtrl.wait();
#if ENABLE_AVSW_READER
        if (0 != extract_audio(nInputFrame)) {
            nvStatus = NV_ENC_ERR_GENERIC;
            break;
        }
#endif //#if ENABLE_AVSW_READER

        //転送の終了状況を確認、可能ならリソースの開放を行う
        auto cuerr = check_inframe_transfer(m_pipelineDepth);
        if (cuerr != cudaSuccess) {
            PrintMes(RGY_LOG_ERROR, _T("Error cudaEventSynchronize: %d (%s).\n"), cuerr, char_to_tstring(_cudaGetErrorEnum(cuerr)).c_str());
            return NV_ENC_ERR_GENERIC;
        }

        //デコード
        FrameBufferDataIn inputFrame;
#if ENABLE_AVSW_READER
        if (m_pDecoder) {
            if (m_pDecoder->GetError()
                || (m_pDecoder->frameQueue()->isEndOfDecode() && m_pDecoder->frameQueue()->isEmpty())) {
                bInputEmpty = true;
            }
            if (!bInputEmpty) {
                CUVIDPARSERDISPINFO dispInfo = { 0 };
                if (!m_pDecoder->frameQueue()->dequeue(&dispInfo)) {
                    //転送の終了状況を確認、可能ならリソースの開放を行う
                    cuerr = check_inframe_transfer(m_pipelineDepth);
                    if (cuerr != cudaSuccess) {
                        PrintMes(RGY_LOG_ERROR, _T("Error cudaEventSynchronize: %d (%s).\n"), cuerr, char_to_tstring(_cudaGetErrorEnum(cuerr)).c_str());
                        return NV_ENC_ERR_GENERIC;
                    }
                    m_pDecoder->frameQueue()->waitForQueueUpdate();
                    continue;
                }
                // OpenGOP等でキーフレームより前のフレームが出てくることがあるのを削除
                if (dispInfo.timestamp < hwDecFirstPts) {
                    m_pDecoder->frameQueue()->releaseFrame(&dispInfo);
                    continue;
                }
                inputFrame.setCuvidInfo(shared_ptr<CUVIDPARSERDISPINFO>(new CUVIDPARSERDISPINFO(dispInfo), [&](CUVIDPARSERDISPINFO *ptr) {
                    m_pDecoder->frameQueue()->releaseFrame(ptr);
                    delete ptr;
                }), m_pDecoder->GetDecFrameInfo());
                inputFrame.setInputFrameId(nInputFrame);
                if (streamIn && queueMetadata.size() > 0) {
                    auto pAVCodecReader = std::dynamic_pointer_cast<RGYInputAvcodec>(m_pFileReader);
                    if (pAVCodecReader != nullptr) {
                        const auto timestamp_status = pAVCodecReader->GetFramePosList()->getStreamPtsStatus();
                        if ((timestamp_status & (~RGY_PTS_NORMAL)) != 0) {
                            PrintMes(RGY_LOG_ERROR, _T("HDR10+ dynamic metadata cannot be copied from input file using avhw reader, as timestamp was not properly got from input file.\n"));
                            PrintMes(RGY_LOG_ERROR, _T("Please consider using avsw reader.\n"));
                            return NV_ENC_ERR_GENERIC;
                        }
                        //cuvidのtimestampはかならず分子が1になっているのでもとに戻す
                        const auto orig_pts = rational_rescale(dispInfo.timestamp, srcTimebase, to_rgy(streamIn->time_base));
                        inputFrame.addFrameData(getMetadata(orig_pts));
                    }
                }
                PrintMes(RGY_LOG_TRACE, _T("input frame (dev) #%d, pic_idx %d, timestamp %lld\n"), nInputFrame, dispInfo.picture_index, dispInfo.timestamp);
            }
        } else
#endif //#if ENABLE_AVSW_READER
        if (m_inputHostBuffer.size()) {
            auto& inputFrameBuf = m_inputHostBuffer[nInputFrame % m_inputHostBuffer.size()];
            if (inputFrameBuf.heTransferFin) {
                //対象バッファの転送が終了しているかを確認
                while (WaitForSingleObject(inputFrameBuf.heTransferFin.get(), 0) == WAIT_TIMEOUT) {
                    cuerr = check_inframe_transfer(m_pipelineDepth);
                    if (cuerr != cudaSuccess) {
                        PrintMes(RGY_LOG_ERROR, _T("Error cudaEventSynchronize: %d (%s).\n"), cuerr, char_to_tstring(_cudaGetErrorEnum(cuerr)).c_str());
                        return NV_ENC_ERR_GENERIC;
                    }
                }
            }
            NVTXRANGE(LoadNextFrame);
            RGYFrameRef frame(inputFrameBuf.cubuf->frame);
            auto rgy_err = m_pFileReader->LoadNextFrame(&frame);
            if (rgy_err != RGY_ERR_NONE) {
                if (rgy_err != RGY_ERR_MORE_DATA) { //RGY_ERR_MORE_DATAは読み込みの正常終了を示す
                    nvStatus = err_to_nv(rgy_err);
                }
                bInputEmpty = true;
            }
            auto heTransferFin = shared_ptr<void>(inputFrameBuf.heTransferFin.get(), [&](void *ptr) {
                SetEvent((HANDLE)ptr);
            });
#if ENABLE_VPP_SMOOTH_QP_FRAME
            for (auto &data : frame.dataList()) {
                if (data->dataType() == RGY_FRAME_DATA_QP) {
                    auto dataqp = dynamic_cast<RGYFrameDataQP *>(data.get());
                    NVEncCtxAutoLock(ctxlock(m_dev->vidCtxLock()));
                    rgy_err = dataqp->transferToGPU(cudaStreamDefault);
                    if (rgy_err != RGY_ERR_NONE) {
                        PrintMes(RGY_LOG_ERROR, _T("Error sending QP table to GPU: %s.\n"), get_err_mes(rgy_err));
                        nvStatus = err_to_nv(rgy_err);
                    }
                }
            }
#endif
            inputFrame.setHostFrameInfo(inputFrameBuf.cubuf->frame, heTransferFin);
            inputFrame.setInputFrameId(nInputFrame);
            PrintMes(RGY_LOG_TRACE, _T("input frame (host) #%d, timestamp %lld, duration %lld\n"), nInputFrame, inputFrame.getTimeStamp(), inputFrame.getDuration());
        } else {
            PrintMes(RGY_LOG_ERROR, _T("Unexpected error at Encode().\n"));
            return NV_ENC_ERR_GENERIC;
        }


        if (!bInputEmpty) {
            //trim反映
            const auto trimSts = frame_inside_range(nInputFrame++, m_trimParam.list);
#if ENABLE_AVSW_READER
            const auto inputFramePts = rational_rescale(inputFrame.getTimeStamp(), srcTimebase, m_outputTimebase);
            if (((m_nAVSyncMode & RGY_AVSYNC_VFR) || vpp_rff || vpp_afs_rff_aware)
                && (trimSts.second > 0) //check_pts内で最初のフレームのptsを0とするようnOutFirstPtsが設定されるので、先頭のtrim blockについてはここでは処理しない
                && (lastTrimFramePts != AV_NOPTS_VALUE)) { //前のフレームがtrimで脱落させたフレームなら
                nOutFirstPts += inputFramePts - lastTrimFramePts; //trimで脱落させたフレームの分の時間を加算
            }
            if (!trimSts.first) {
                lastTrimFramePts = inputFramePts; //脱落させたフレームの時間を記憶
            }
#endif
            if (!trimSts.first) {
                continue; //trimにより脱落させるフレーム
            }
            if (!m_pFileReader->checkTimeSeekTo(inputFrame.getTimeStamp(), srcTimebase)) {
                continue; //seektoにより脱落させるフレーム
            }
            lastTrimFramePts = AV_NOPTS_VALUE;
            auto decFrames = check_pts(&inputFrame);

            for (auto idf = decFrames.begin(); idf != decFrames.end(); idf++) {
                dqInFrames.push_back(std::move(*idf));
            }
        }
        inputFrame.resetCuvidInfo();

        while (((dqInFrames.size() || bInputEmpty) && !bFilterEmpty) && nvStatus == NV_ENC_SUCCESS) {
            const bool bDrain = (dqInFrames.size()) ? false : bInputEmpty;
            auto& inframe = (dqInFrames.size()) ? dqInFrames.front() : dummyFrame;
            bool bDrainFin = bDrain;
            auto filter_ret = filter_frame(nFilterFrame, inframe, dqEncFrames, bDrainFin);
            if (filter_ret != NV_ENC_SUCCESS) {
                nvStatus = filter_ret;
                break;
            }
            bFilterEmpty = bDrainFin;
            if (!bDrain) {
                dqInFrames.pop_front();
            }
            while ((int)dqEncFrames.size() >= m_pipelineDepth) {
                auto& encframe = dqEncFrames.front();
                auto enc_ret = send_encoder(nEncodeFrames, encframe);
                if (enc_ret != NV_ENC_SUCCESS) {
                    nvStatus = enc_ret;
                    break;
                }
                dqEncFrames.pop_front();
            }
        }
    }
    //すべての転送を終了させる
    while (dqFrameTransferData.size()) {
        auto cuerr = check_inframe_transfer(1);
        if (cuerr != cudaSuccess) {
            PrintMes(RGY_LOG_ERROR, _T("Error cudaEventSynchronize: %d (%s).\n"), cuerr, char_to_tstring(_cudaGetErrorEnum(cuerr)).c_str());
            return NV_ENC_ERR_GENERIC;
        }
    }
    //エンコードバッファのフレームをすべて転送
    while (dqEncFrames.size()) {
        auto& encframe = dqEncFrames.front();
        auto nvStatusFlush = send_encoder(nEncodeFrames, encframe);
        if (nvStatusFlush != NV_ENC_SUCCESS) {
            nvStatus = nvStatusFlush;
            break;
        }
        dqEncFrames.pop_front();
    }

#if ENABLE_AVSW_READER
    if (th_input.joinable()) {
        //ここでフレームをすべて吐き出し切らないと、中断時にデコードスレッドが終了しない
        PrintMes(RGY_LOG_DEBUG, _T("Flushing Decoder\n"));
        if (m_pDecoder) {
            //エンコード中断時の処理
            while (!m_pDecoder->GetError()
                && !(m_pDecoder->frameQueue()->isEndOfDecode() && m_pDecoder->frameQueue()->isEmpty())) {
                m_pDecoder->frameQueue()->endDecode(); //デコーダの待機ループから強制的に出る
                CUVIDPARSERDISPINFO pInfo;
                if (m_pDecoder->frameQueue()->dequeue(&pInfo)) {
                    m_pDecoder->frameQueue()->releaseFrame(&pInfo);
                }
            }
        }
        th_input.join();
        PrintMes(RGY_LOG_DEBUG, _T("Flushed Decoder\n"));
    }
    for (const auto& writer : m_pFileWriterListAudio) {
        auto pAVCodecWriter = std::dynamic_pointer_cast<RGYOutputAvcodec>(writer);
        if (pAVCodecWriter != nullptr) {
            //エンコーダなどにキャッシュされたパケットを書き出す
            pAVCodecWriter->WriteNextPacket(nullptr);
        }
    }
#endif //#if ENABLE_AVSW_READER
    //FlushEncoderはかならず行わないと、NvEncDestroyEncoderで異常終了する
    auto encstatus = nvStatus;
    if (nEncodeFrames > 0 || nvStatus == NV_ENC_SUCCESS) {
        encstatus = FlushEncoder();
        if (encstatus != NV_ENC_SUCCESS) {
            PrintMes(RGY_LOG_ERROR, _T("Error FlushEncoder: %d.\n"), encstatus);
            nvStatus = encstatus;
        } else {
            PrintMes(RGY_LOG_DEBUG, _T("Flushed Encoder\n"));
        }
        if (m_ssim) {
            PrintMes(RGY_LOG_DEBUG, _T("Flushing ssim/psnr calc.\n"));
            m_ssim->addBitstream(nullptr);
        }
    }
    m_pFileWriter->Close();
    m_pFileReader->Close();
    if (m_deviceUsage) {
        m_deviceUsage->close();
    }
    m_pStatus->WriteResults();
    if (m_ssim) {
        m_ssim->showResult();
    }
    queueMetadata.close([](RGYFrameDataMetadata **ptr) { if (*ptr) { delete *ptr; *ptr = nullptr; }; });
    vector<std::pair<tstring, double>> filter_result;
    for (auto& filter : m_vpFilters) {
        auto avgtime = filter->GetAvgTimeElapsed();
        if (avgtime > 0.0) {
            filter_result.push_back({ filter->name(), avgtime });
        }
    }
    if (filter_result.size()) {
        PrintMes(RGY_LOG_INFO, _T("\nVpp Filter Performance\n"));
        const auto max_len = std::accumulate(filter_result.begin(), filter_result.end(), 0u, [](uint32_t max_length, std::pair<tstring, double> info) {
            return std::max(max_length, (uint32_t)info.first.length());
        });
        for (const auto& info : filter_result) {
            tstring str = info.first + _T(":");
            for (uint32_t i = (uint32_t)info.first.length(); i < max_len; i++) {
                str += _T(" ");
            }
            PrintMes(RGY_LOG_INFO, _T("%s %7.1f us\n"), str.c_str(), info.second * 1000.0);
        }
    }
    return nvStatus;
}
#elif 0
NVENCSTATUS NVEncCore::Encode() {
    NVENCSTATUS nvStatus = NV_ENC_SUCCESS;
    if (m_pDecoder || m_inputHostBuffer.size() > 0) {
        return Encode2();
    }

    m_pStatus->SetStart();

    int ret = 0;
    const int bufferCount = m_encodeBufferCount;
#if ENABLE_AVSW_READER
    const AVCodecContext *pVideoCtx = nullptr;
    RGYInputAvcodec *pReader = dynamic_cast<RGYInputAvcodec *>(m_pFileReader.get());
    if (pReader != nullptr) {
        pVideoCtx = pReader->GetInputVideoCodecCtx();
    }

    //streamのindexから必要なwriteへのポインタを返すテーブルを作成
    std::map<int, shared_ptr<RGYOutputAvcodec>> pWriterForAudioStreams;
    for (auto pWriter : m_pFileWriterListAudio) {
        auto pAVCodecWriter = std::dynamic_pointer_cast<RGYOutputAvcodec>(pWriter);
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
            auto pAVCodecReader = std::dynamic_pointer_cast<RGYInputAvcodec>(m_pFileReader);
            vector<AVPacket> packetList;
            if (pAVCodecReader != nullptr) {
                packetList = pAVCodecReader->GetStreamDataPackets();
            }
            //音声ファイルリーダーからのトラックを結合する
            for (const auto& reader : m_AudioReaders) {
                auto pReader = std::dynamic_pointer_cast<RGYInputAvcodec>(reader);
                if (pReader != nullptr) {
                    vector_cat(packetList, pReader->GetStreamDataPackets());
                }
            }
            //パケットを各Writerに分配する
            for (uint32_t i = 0; i < packetList.size(); i++) {
                const int trackId = (int16_t)(packetList[i].flags >> 16);
                if (pWriterForAudioStreams.count(trackId)) {
                    auto pWriter = pWriterForAudioStreams[trackId];
                    if (pWriter == nullptr) {
                        PrintMes(RGY_LOG_ERROR, _T("Invalid writer found for audio track %d\n"), trackId);
                        return 1;
                    }
                    if (0 != (sts = pWriter->WriteNextPacket(&packetList[i]))) {
                        return 1;
                    }
                } else {
                    PrintMes(RGY_LOG_ERROR, _T("Failed to find writer for audio track %d\n"), trackId);
                    return 1;
                }
            }
        }
        return 0;
    };

    if (m_pDecoder) {
        auto th_input = std::thread([this, pVideoCtx, &nvStatus](){
            CUresult curesult = CUDA_SUCCESS;
            vector<uint8_t> bitstream;
            int sts = NVENC_THREAD_RUNNING;
            for (int i = 0; sts == NVENC_THREAD_RUNNING && nvStatus == NV_ENC_SUCCESS && !m_pDecoder->GetError(); i++) {
                sts = m_pFileReader->LoadNextFrame(nullptr, 0);
                int64_t pts;
                m_pFileReader->GetNextBitstream(bitstream, &pts);
                PrintMes(RGY_LOG_TRACE, _T("Set packet %d\n"), i);
                if (CUDA_SUCCESS != (curesult = m_pDecoder->DecodePacket(bitstream.data(), bitstream.size(), pts, pVideoCtx->pkt_timebase))) {
                    PrintMes(RGY_LOG_ERROR, _T("Error in DecodePacket: %d (%s).\n"), curesult, char_to_tstring(_cudaGetErrorEnum(curesult)).c_str());
                    return curesult;
                }
            }
            if (CUDA_SUCCESS != (curesult = m_pDecoder->DecodePacket(nullptr, 0, AV_NOPTS_VALUE, pVideoCtx->pkt_timebase))) {
                PrintMes(RGY_LOG_ERROR, _T("Error in DecodePacketFin: %d (%s).\n"), curesult, char_to_tstring(_cudaGetErrorEnum(curesult)).c_str());
            }
            return curesult;
        });
        PrintMes(RGY_LOG_DEBUG, _T("Started Encode thread\n"));

        CProcSpeedControl speedCtrl(m_nProcSpeedLimit);
        int64_t nEstimatedPts = AV_NOPTS_VALUE;
        const int nFrameDuration = (int)av_rescale_q(1, av_make_q(m_inputFps.second, m_inputFps.first), pVideoCtx->pkt_timebase);
        int decodedFrame = 0;
        int encodedFrame = 0;
        while (!m_pDecoder->GetError()
            && !(m_pDecoder->frameQueue()->isEndOfDecode() && m_pDecoder->frameQueue()->isEmpty())) {
            if (m_pAbortByUser && *m_pAbortByUser) {
                nvStatus = NV_ENC_ERR_ABORT;
                break;
            }

            CUVIDPARSERDISPINFO pInfo;
            if (m_pDecoder->frameQueue()->dequeue(&pInfo)) {
                int64_t pts = av_rescale_q(pInfo.timestamp, HW_NATIVE_TIMEBASE, pVideoCtx->pkt_timebase);
                if (m_pTrimParam && !frame_inside_range(decodedFrame++, m_trimParam.list)) {
                    m_pDecoder->frameQueue()->releaseFrame(&pInfo);
                    continue;
                }

                auto encode = [&](CUVIDPROCPARAMS oVPP) {
                    speedCtrl.wait();
                    CUresult curesult = CUDA_SUCCESS;
                    PrintMes(RGY_LOG_TRACE, _T("Get decoded frame %d\n"), decodedFrame);
                    CUdeviceptr dMappedFrame = 0;
                    unsigned int pitch;
                    if (CUDA_SUCCESS != (curesult = cuvidMapVideoFrame(m_pDecoder->GetDecoder(), pInfo.picture_index, &dMappedFrame, &pitch, &oVPP))) {
                        PrintMes(RGY_LOG_ERROR, _T("Error cuvidMapVideoFrame: %d (%s).\n"), curesult, char_to_tstring(_cudaGetErrorEnum(curesult)).c_str());
                        return NV_ENC_ERR_GENERIC;
                    }

                    EncodeFrameConfig stEncodeConfig = { 0 };
                    stEncodeConfig.dptr = dMappedFrame;
                    stEncodeConfig.pitch = pitch;
                    stEncodeConfig.width = m_uEncWidth;
                    stEncodeConfig.height = m_uEncHeight;
                    PrintMes(RGY_LOG_TRACE, _T("Set frame to encode %d\n"), decodedFrame);
                    pInfo.timestamp = encodedFrame;
                    auto encstatus = EncodeFrame(&stEncodeConfig, pInfo.timestamp);
                    encodedFrame++;
                    if (NV_ENC_SUCCESS != encstatus) {
                        PrintMes(RGY_LOG_ERROR, _T("Error EncodeFrame: %d.\n"), encstatus);
                        return encstatus;
                    }

                    if (CUDA_SUCCESS != (curesult = cuvidUnmapVideoFrame(m_pDecoder->GetDecoder(), dMappedFrame))) {
                        PrintMes(RGY_LOG_ERROR, _T("Error cuvidMapVideoFrame: %d (%s).\n"), curesult, char_to_tstring(_cudaGetErrorEnum(curesult)).c_str());
                        return NV_ENC_ERR_GENERIC;
                    }
                    return NV_ENC_SUCCESS;
                };
                CUVIDPROCPARAMS oVPP = { 0 };
                oVPP.top_field_first = m_stPicStruct != NV_ENC_PIC_STRUCT_FIELD_BOTTOM_TOP;

                auto encode_frame = [&]() {
                    NVENCSTATUS status = NV_ENC_SUCCESS;
                    auto deint = m_pDecoder->getDeinterlaceMode();
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
                        PrintMes(RGY_LOG_ERROR, _T("Unknown Deinterlace mode\n"));
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
                m_pDecoder->frameQueue()->releaseFrame(&pInfo);
            } else {
                m_pDecoder->frameQueue()->waitForQueueUpdate();
            }
        }
        if (th_input.joinable()) {
            //ここでフレームをすべて吐き出し切らないと、中断時にデコードスレッドが終了しない
            while (!m_pDecoder->GetError()
                && !(m_pDecoder->frameQueue()->isEndOfDecode() && m_pDecoder->frameQueue()->isEmpty())) {
                CUVIDPARSERDISPINFO pInfo;
                if (m_pDecoder->frameQueue()->dequeue(&pInfo)) {
                    m_pDecoder->frameQueue()->releaseFrame(&pInfo);
                }
            }
            th_input.join();
        }
        PrintMes(RGY_LOG_DEBUG, _T("Joined Encode thread\n"));
        if (m_pDecoder->GetError()) {
            nvStatus = NV_ENC_ERR_GENERIC;
        }
    } else
#endif //#if ENABLE_AVSW_READER
    {
        CProcSpeedControl speedCtrl(m_nProcSpeedLimit);
        for (int iFrame = 0; nvStatus == NV_ENC_SUCCESS; iFrame++) {
            if (m_pAbortByUser && *m_pAbortByUser) {
                nvStatus = NV_ENC_ERR_ABORT;
                break;
            }
            speedCtrl.wait();
#if ENABLE_AVSW_READER
            if (0 != extract_audio()) {
                nvStatus = NV_ENC_ERR_GENERIC;
                break;
            }
#endif //#if ENABLE_AVSW_READER
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
#if ENABLE_AVSW_READER
    for (const auto& writer : m_pFileWriterListAudio) {
        auto pAVCodecWriter = std::dynamic_pointer_cast<RGYOutputAvcodec>(writer);
        if (pAVCodecWriter != nullptr) {
            //エンコーダなどにキャッシュされたパケットを書き出す
            pAVCodecWriter->WriteNextPacket(nullptr);
        }
    }
#endif //#if ENABLE_AVSW_READER
    PrintMes(RGY_LOG_INFO, _T("                                                                         \n"));
    //FlushEncoderはかならず行わないと、NvEncDestroyEncoderで異常終了する
    auto encstatus = FlushEncoder();
    if (nvStatus == NV_ENC_SUCCESS && encstatus != NV_ENC_SUCCESS) {
        PrintMes(RGY_LOG_ERROR, _T("Error FlushEncoder: %d.\n"), encstatus);
        nvStatus = encstatus;
    } else {
        PrintMes(RGY_LOG_DEBUG, _T("Flushed Encoder\n"));
    }
    m_pFileReader->Close();
    m_pFileWriter->Close();
    m_pStatus->writeResult();
    return nvStatus;
}
#endif

tstring NVEncCore::GetEncodingParamsInfo(int output_level) {
    tstring str;
    auto add_str =[output_level, &str](int info_level, const TCHAR *fmt, ...) {
        if (info_level >= output_level) {
            va_list args;
            va_start(args, fmt);
            const size_t append_len = _vsctprintf(fmt, args) + 1;
            std::vector<TCHAR> buf(append_len, 0);
            _vstprintf_s(buf.data(), append_len, fmt, args);
            str += buf.data();
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

    tstring gpu_info = m_dev->infostr();
    int cudaDriverVersion = 0;
    {
        NVEncCtxAutoLock(ctxlock(m_dev->vidCtxLock()));
        cuDriverGetVersion(&cudaDriverVersion);
    }
    add_str(RGY_LOG_ERROR, _T("%s\n"), get_encoder_version());
#if defined(_WIN32) || defined(_WIN64)
    add_str(RGY_LOG_INFO,  _T("OS Version     %s [%s]\n"), getOSVersion().c_str(), getACPCodepageStr().c_str());
#else
    add_str(RGY_LOG_INFO,  _T("OS Version     %s\n"), getOSVersion().c_str());
#endif
    add_str(RGY_LOG_INFO,  _T("CPU            %s\n"), cpu_info);
    {
        if (m_parallelEnc && m_devNames.size() > 1) {
            add_str(RGY_LOG_INFO, _T("GPU            %s\n"), m_devNames[0].c_str());
            for (size_t i = 1; i < m_devNames.size(); i++) {
                add_str(RGY_LOG_INFO, _T("               %s\n"), m_devNames[i].c_str());
            }
        } else {
            add_str(RGY_LOG_INFO, _T("GPU            %s\n"), gpu_info.c_str());
        }
    }
    if (m_dev->encoder()) {
        const auto encAPIver = m_dev->encoder()->getAPIver();
        add_str(RGY_LOG_INFO, _T("NVENC / CUDA   NVENC API %d.%d, CUDA %d.%d, schedule mode: %s\n"),
            nvenc_api_ver_major(encAPIver), nvenc_api_ver_minor(encAPIver),
            cudaDriverVersion / 1000, (cudaDriverVersion % 1000) / 10, get_chr_from_value(list_cuda_schedule, m_cudaSchedule));
    } else {
        add_str(RGY_LOG_INFO, _T("CUDA           CUDA %d.%d, schedule mode: %s\n"),
            cudaDriverVersion / 1000, (cudaDriverVersion % 1000) / 10, get_chr_from_value(list_cuda_schedule, m_cudaSchedule));
    }
    add_str(RGY_LOG_ERROR, _T("Input Buffers  %s, %d frames\n"), _T("CUDA"), m_encodeBufferCount);
    tstring inputMes = m_pFileReader->GetInputMessage();
    for (const auto& reader : m_AudioReaders) {
        inputMes += _T("\n") + tstring(reader->GetInputMessage());
    }
    auto inputMesSplitted = split(inputMes, _T("\n"));
    for (uint32_t i = 0; i < (uint32_t)inputMesSplitted.size(); i++) {
        add_str(RGY_LOG_ERROR, _T("%s%s\n"), (i == 0) ? _T("Input Info     ") : _T("               "), inputMesSplitted[i].c_str());
    }
#if ENABLE_AVSW_READER
    if (m_pDecoder && m_pDecoder->getDeinterlaceMode() != cudaVideoDeinterlaceMode_Weave) {
        add_str(RGY_LOG_ERROR, _T("Deinterlace    %s\n"), get_chr_from_value(list_deinterlace, m_pDecoder->getDeinterlaceMode()));
    }
#endif //#if ENABLE_AVSW_READER
    if (m_trimParam.list.size()
        && !(m_trimParam.list[0].start == 0 && m_trimParam.list[0].fin == TRIM_MAX)) {
        add_str(RGY_LOG_ERROR, _T("%s"), _T("Trim           "));
        for (auto trim : m_trimParam.list) {
            if (trim.fin == TRIM_MAX) {
                add_str(RGY_LOG_ERROR, _T("%d-fin "), trim.start + m_trimParam.offset);
            } else {
                add_str(RGY_LOG_ERROR, _T("%d-%d "), trim.start + m_trimParam.offset, trim.fin + m_trimParam.offset);
            }
        }
        add_str(RGY_LOG_ERROR, _T("[offset: %d]\n"), m_trimParam.offset);
    }
    if (m_nAVSyncMode & RGY_AVSYNC_FORCE_CFR) {
        add_str(RGY_LOG_ERROR, _T("AVSync         %s\n"), get_chr_from_value(list_avsync, m_nAVSyncMode));
    }
    tstring vppFilterMes;
    for (const auto& vppblock : m_vpFilters) {
        for (const auto& filter : vppblock.vppnv) {
            vppFilterMes += strsprintf(_T("%s%s\n"), (vppFilterMes.length()) ? _T("               ") : _T("Vpp Filters    "), filter->GetInputMessage().c_str());
        }
    }
    if (m_videoQualityMetric) {
        vppFilterMes += _T("               ") + m_videoQualityMetric->GetInputMessage() + _T("\n");
    }
    add_str(RGY_LOG_ERROR, vppFilterMes.c_str());
    if (!m_dev->encoder()) {
        add_str(RGY_LOG_INFO, _T("\n"));
        return str;
    }
    str += GetEncoderParamsInfo(output_level, true);
    return str;
}

tstring NVEncCore::GetEncoderParamsInfo(int output_level, bool add_output_info) {
    tstring str;
    auto add_str =[output_level, &str](int info_level, const TCHAR *fmt, ...) {
        if (info_level >= output_level) {
            va_list args;
            va_start(args, fmt);
            const size_t append_len = _vsctprintf(fmt, args) + 1;
            std::vector<TCHAR> buf(append_len, 0);
            _vstprintf_s(buf.data(), append_len, fmt, args);
            str += buf.data();
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
    if (m_parallelEnc) {
        tstring parallelEncMes = strsprintf(_T("%d"), m_parallelEnc->parallelCount());
        if (m_parallelEnc->chunks() != m_parallelEnc->parallelCount()) {
            parallelEncMes += strsprintf(_T(" [chunks: %d]"), m_parallelEnc->chunks());
        }
        add_str(RGY_LOG_ERROR, _T("Parallel Enc   %s\n"), parallelEncMes.c_str());
    }
    const auto codecFeature = m_dev->encoder()->getCodecFeature(m_stCodecGUID);
    const RGY_CODEC rgy_codec = codec_guid_enc_to_rgy(m_stCodecGUID);
    const int bitDepth = get_bitDepth(m_stCreateEncodeParams.encodeConfig->encodeCodecConfig, rgy_codec, m_dev->encoder()->getAPIver());
    if (rgy_codec == RGY_CODEC_H264) {
        add_str(RGY_LOG_ERROR, _T("Output Info    %s %s @ Level %s\n"), get_name_from_guid(m_stCodecGUID, list_nvenc_codecs),
            get_codec_profile_name_from_guid(rgy_codec, m_stEncConfig.profileGUID).c_str(),
            get_codec_level_name(rgy_codec, m_stEncConfig.encodeCodecConfig.h264Config.level).c_str());
    } else if (rgy_codec == RGY_CODEC_HEVC) {
        add_str(RGY_LOG_ERROR, _T("Output Info    %s %s%s @ Level %s%s\n"), get_name_from_guid(m_stCodecGUID, list_nvenc_codecs),
            memcmp(&NV_ENC_HEVC_PROFILE_FREXT_GUID, &m_stEncConfig.profileGUID, sizeof(GUID)) == 0 ? _T("main444/422") : get_codec_profile_name_from_guid(rgy_codec, m_stEncConfig.profileGUID).c_str(),
            (rgy_codec == RGY_CODEC_HEVC && 0 == memcmp(&NV_ENC_HEVC_PROFILE_FREXT_GUID, &m_stEncConfig.profileGUID, sizeof(GUID)) && bitDepth > 8) ? _T(" 10bit") : _T(""),
            get_codec_level_name(rgy_codec, m_stEncConfig.encodeCodecConfig.hevcConfig.level).c_str(),
            m_stEncConfig.encodeCodecConfig.hevcConfig.enableAlphaLayerEncoding ? _T(" + alpha") : _T(""));
    } else if (rgy_codec == RGY_CODEC_AV1) {
        add_str(RGY_LOG_ERROR, _T("Output Info    %s %s%s @ Level %s\n"), get_name_from_guid(m_stCodecGUID, list_nvenc_codecs),
            get_codec_profile_name_from_guid(rgy_codec, m_stEncConfig.profileGUID).c_str(),
            (rgy_codec == RGY_CODEC_AV1 && bitDepth > 8) ? _T(" 10bit") : _T(""),
            get_codec_level_name(rgy_codec, m_stEncConfig.encodeCodecConfig.av1Config.level).c_str());
    } else {
        return _T("Invalid codec");
    }
    add_str(RGY_LOG_ERROR, _T("               %dx%d%s %d:%d %.3ffps (%d/%dfps)\n"), m_uEncWidth, m_uEncHeight, (m_stEncConfig.frameFieldMode != NV_ENC_PARAMS_FRAME_FIELD_MODE_FRAME) ? _T("i") : _T("p"), m_sar.n(), m_sar.d(), m_stCreateEncodeParams.frameRateNum / (double)m_stCreateEncodeParams.frameRateDen, m_stCreateEncodeParams.frameRateNum, m_stCreateEncodeParams.frameRateDen);
    if (add_output_info) {
        if (m_pFileWriter) {
            auto inputMesSplitted = split(m_pFileWriter->GetOutputMessage(), _T("\n"));
            for (auto mes : inputMesSplitted) {
                if (mes.length()) {
                    add_str(RGY_LOG_ERROR,_T("%s%s\n"), _T("               "), mes.c_str());
                }
            }
        }
        for (auto pWriter : m_pFileWriterListAudio) {
            if (pWriter && pWriter != m_pFileWriter) {
                auto inputMesSplitted = split(pWriter->GetOutputMessage(), _T("\n"));
                for (auto mes : inputMesSplitted) {
                    if (mes.length()) {
                        add_str(RGY_LOG_ERROR,_T("%s%s\n"), _T("               "), mes.c_str());
                    }
                }
            }
        }
    }
    if (m_dev->encoder()->checkAPIver(10,0)) {
        add_str(RGY_LOG_INFO, _T("Encoder Preset %s\n"), get_name_from_guid(m_stCreateEncodeParams.presetGUID, list_nvenc_preset_names_ver10));
    } else {
        add_str(RGY_LOG_INFO, _T("Encoder Preset %s\n"), get_name_from_guid(m_stCreateEncodeParams.presetGUID, list_nvenc_preset_names_ver9_2));
    }
    add_str(RGY_LOG_ERROR, _T("Rate Control   %s"), get_chr_from_value(list_nvenc_rc_method_en, m_stEncConfig.rcParams.rateControlMode));

    #pragma warning (push)
    #pragma warning (disable: 4996)
    RGY_DISABLE_WARNING_PUSH
    RGY_DISABLE_WARNING_STR("-Wdeprecated-declarations")
    const bool lossless = (rgy_codec == RGY_CODEC_H264 && m_stCreateEncodeParams.encodeConfig->encodeCodecConfig.h264Config.qpPrimeYZeroTransformBypassFlag)
        || memcmp(&m_stCreateEncodeParams.presetGUID, &NV_ENC_PRESET_LOSSLESS_HP_GUID, sizeof(m_stCreateEncodeParams.presetGUID)) == 0
        || memcmp(&m_stCreateEncodeParams.presetGUID, &NV_ENC_PRESET_LOSSLESS_DEFAULT_GUID, sizeof(m_stCreateEncodeParams.presetGUID)) == 0
        || m_stCreateEncodeParams.tuningInfo == NV_ENC_TUNING_INFO_LOSSLESS;
    RGY_DISABLE_WARNING_POP
    #pragma warning (pop)
    if (NV_ENC_PARAMS_RC_CONSTQP == m_stEncConfig.rcParams.rateControlMode) {
        add_str(RGY_LOG_ERROR, _T("  I:%d  P:%d  B:%d%s\n"), m_stEncConfig.rcParams.constQP.qpIntra, m_stEncConfig.rcParams.constQP.qpInterP, m_stEncConfig.rcParams.constQP.qpInterB,
            lossless ? _T(" (lossless)") : _T(""));
        if (m_dev->encoder()->checkAPIver(11, 1)) {
            add_str(RGY_LOG_INFO, _T("ChromaQPOffset cb:%d  cr:%d\n"), m_stEncConfig.rcParams.cbQPIndexOffset, m_stEncConfig.rcParams.crQPIndexOffset);
        }
    } else {
        add_str(RGY_LOG_ERROR, _T("\n"));
        if (m_dev->encoder()->checkAPIver(10, 0)) {
            add_str(RGY_LOG_INFO, _T("Multipass      %s\n"), get_chr_from_value(list_nvenc_multipass_mode, m_stEncConfig.rcParams.multiPass));
        }
        add_str(RGY_LOG_ERROR, _T("Bitrate        %d kbps (Max: %d kbps)\n"), m_stEncConfig.rcParams.averageBitRate / 1000, m_stEncConfig.rcParams.maxBitRate / 1000);
        if (m_stEncConfig.rcParams.targetQuality) {
            double targetQuality = m_stEncConfig.rcParams.targetQuality + m_stEncConfig.rcParams.targetQualityLSB / 256.0;
            add_str(RGY_LOG_ERROR, _T("Target Quality %.2f\n"), targetQuality);
        } else {
            add_str(RGY_LOG_ERROR, _T("Target Quality auto\n"));
        }
        if (rgy_codec == RGY_CODEC_HEVC && m_stEncConfig.encodeCodecConfig.hevcConfig.enableAlphaLayerEncoding) {
            const int alpharatio = m_stCreateEncodeParams.encodeConfig->rcParams.alphaLayerBitrateRatio;
            if (alpharatio > 0) {
                add_str(RGY_LOG_INFO, _T("Alpha Bitrate  %d (%.2f)\n"), alpharatio, 1.0f / (float)(1 + alpharatio));
            }
        }
        if (m_stEncConfig.rcParams.enableInitialRCQP) {
            add_str(RGY_LOG_INFO,  _T("Initial QP     I:%d  P:%d  B:%d\n"), m_stEncConfig.rcParams.initialRCQP.qpIntra, m_stEncConfig.rcParams.initialRCQP.qpInterP, m_stEncConfig.rcParams.initialRCQP.qpInterB);
        }
        if (m_stEncConfig.rcParams.enableMaxQP || m_stEncConfig.rcParams.enableMinQP) {
            const int qpMaxCodec = (rgy_codec == RGY_CODEC_AV1) ? 255 : ((bitDepth > 8) ? 63 : 51);
            int minQPI = (m_stEncConfig.rcParams.enableMinQP) ? m_stEncConfig.rcParams.minQP.qpIntra  :  0;
            int maxQPI = (m_stEncConfig.rcParams.enableMaxQP) ? m_stEncConfig.rcParams.maxQP.qpIntra  : qpMaxCodec;
            int minQPP = (m_stEncConfig.rcParams.enableMinQP) ? m_stEncConfig.rcParams.minQP.qpInterP :  0;
            int maxQPP = (m_stEncConfig.rcParams.enableMaxQP) ? m_stEncConfig.rcParams.maxQP.qpInterP : qpMaxCodec;
            int minQPB = (m_stEncConfig.rcParams.enableMinQP) ? m_stEncConfig.rcParams.minQP.qpInterB :  0;
            int maxQPB = (m_stEncConfig.rcParams.enableMaxQP) ? m_stEncConfig.rcParams.maxQP.qpInterB : qpMaxCodec;
            add_str(RGY_LOG_INFO,  _T("QP range       I:%d-%d  P:%d-%d  B:%d-%d\n"), minQPI, maxQPI, minQPP, maxQPP, minQPB, maxQPB);
        }
        if (m_dev->encoder()->checkAPIver(11, 1)) {
            add_str(RGY_LOG_INFO, _T("QP Offset      cb:%d  cr:%d\n"), m_stEncConfig.rcParams.cbQPIndexOffset, m_stEncConfig.rcParams.crQPIndexOffset);
        }
        add_str(RGY_LOG_INFO,  _T("VBV buf size   %s\n"), value_or_auto(m_stEncConfig.rcParams.vbvBufferSize / 1000,   0, _T("kbit")).c_str());
        add_str(RGY_LOG_DEBUG, _T("VBV init delay %s\n"), value_or_auto(m_stEncConfig.rcParams.vbvInitialDelay / 1000, 0, _T("kbit")).c_str());
    }
    if (m_dynamicRC.size() > 0) {
        tstring strDynamicRC = tstring(_T("DynamicRC      ")) + m_dynamicRC[0].print();
        for (int i = 1; i < (int)m_dynamicRC.size(); i++) {
            strDynamicRC += _T("\n               ") + m_dynamicRC[i].print();
        }
        add_str(RGY_LOG_INFO, _T("%s\n"), strDynamicRC.c_str());
    }

    if (m_dev->encoder()->checkAPIver(12, 1)) {
        add_str(RGY_LOG_INFO, _T("Split Enc Mode %s\n"), get_chr_from_value(list_split_enc_mode, m_stCreateEncodeParams.splitEncodeMode));
    }
    if (ENABLE_NVENC_SDK_TUNE) {
        add_str(RGY_LOG_INFO, _T("Tuning Info    %s\n"), get_chr_from_value(list_tuning_info, m_stCreateEncodeParams.tuningInfo));
    }
    tstring strLookahead = _T("Lookahead      ");
    if (m_stEncConfig.rcParams.enableLookahead) {
        strLookahead += strsprintf(_T("on, %d frames"), m_stEncConfig.rcParams.lookaheadDepth);
        if (rgy_codec == RGY_CODEC_HEVC && m_stEncConfig.rcParams.lookaheadLevel != NV_ENC_LOOKAHEAD_LEVEL_AUTOSELECT) {
            strLookahead += tstring(_T(", Level ")) + get_chr_from_value(list_lookahead_level, m_stEncConfig.rcParams.lookaheadLevel);
        }
        if (!m_stEncConfig.rcParams.disableBadapt || !m_stEncConfig.rcParams.disableIadapt) {
            strLookahead += _T(", Adaptive ");
            if (!m_stEncConfig.rcParams.disableIadapt) strLookahead += _T("I");
            if (!m_stEncConfig.rcParams.disableBadapt && !m_stEncConfig.rcParams.disableIadapt) strLookahead += _T(", ");
            if (!m_stEncConfig.rcParams.disableBadapt) strLookahead += _T("B");
            strLookahead += _T(" Insert");
        }
    } else {
        strLookahead += _T("off");
    }
    add_str(RGY_LOG_INFO,  _T("%s\n"), strLookahead.c_str());
    add_str(RGY_LOG_INFO,  _T("GOP length     %d frames\n"), m_stEncConfig.gopLength);
    const auto bref_mode = get_useBFramesAsRef(m_stEncConfig.encodeCodecConfig, rgy_codec);
    add_str(RGY_LOG_ERROR,  _T("B frames       %d frames [ref mode: %s]\n"), m_stEncConfig.frameIntervalP - 1, get_chr_from_value(list_bref_mode, bref_mode));
    if (rgy_codec == RGY_CODEC_H264) {
        add_str(RGY_LOG_DEBUG, _T("Output         "));
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
        add_str(RGY_LOG_DEBUG, _T("%s\n"), bitstream_info);
    }

    tstring strRef = strsprintf(_T("%d frames"), numRefFrames(m_stEncConfig.encodeCodecConfig, rgy_codec));
    if (m_dev->encoder()->checkAPIver(9, 1)) {
        const auto numRefL0 = get_numRefL0(m_stEncConfig.encodeCodecConfig, rgy_codec);
        const auto numRefL1 = get_numRefL1(m_stEncConfig.encodeCodecConfig, rgy_codec);
        if (codecFeature->getCapLimit(NV_ENC_CAPS_SUPPORT_MULTIPLE_REF_FRAMES)) {
            strRef += strsprintf(_T(", MultiRef L0:%s L1:%s"), get_chr_from_value(list_num_refs, numRefL0), get_chr_from_value(list_num_refs, numRefL1));
        }
    }
    const bool bEnableLTR = get_enableLTR(m_stEncConfig.encodeCodecConfig, rgy_codec);
    if (bEnableLTR) {
        strRef += _T(", LTR:on");
    }
    add_str(RGY_LOG_ERROR,  _T("Ref frames     %s\n"), strRef.c_str());

    tstring strAQ;
    if (m_stEncConfig.rcParams.enableAQ || m_stEncConfig.rcParams.enableTemporalAQ) {
        strAQ = _T("on");
        strAQ += _T(" (");
        if (m_stEncConfig.rcParams.enableAQ)         strAQ += _T("spatial");
        if (m_stEncConfig.rcParams.enableAQ && m_stEncConfig.rcParams.enableTemporalAQ) strAQ += _T(", ");
        if (m_stEncConfig.rcParams.enableTemporalAQ) strAQ += _T("temporal");
        strAQ += _T(", strength ");
        strAQ += (m_stEncConfig.rcParams.aqStrength == 0) ? _T("auto") : strsprintf(_T("%d"), m_stEncConfig.rcParams.aqStrength);
        strAQ += _T(")");
    } else {
        strAQ = _T("off");
    }
    add_str(RGY_LOG_INFO,  _T("AQ             %s\n"), strAQ.c_str());
    if (rgy_codec == RGY_CODEC_H264 || rgy_codec == RGY_CODEC_HEVC) {
        if (get_sliceMode(m_stEncConfig.encodeCodecConfig, rgy_codec) == 3) {
            add_str((get_sliceModeData(m_stEncConfig.encodeCodecConfig, rgy_codec) > 1) ? RGY_LOG_INFO : RGY_LOG_DEBUG, _T("Slices            %d\n"), get_sliceModeData(m_stEncConfig.encodeCodecConfig, rgy_codec));
        } else {
            add_str((get_sliceModeData(m_stEncConfig.encodeCodecConfig, rgy_codec) > 1) ? RGY_LOG_INFO : RGY_LOG_DEBUG, _T("Slice          Mode:%d, ModeData:%d\n"), get_sliceMode(m_stEncConfig.encodeCodecConfig, rgy_codec), get_sliceModeData(m_stEncConfig.encodeCodecConfig, rgy_codec));
        }
    }
    if (rgy_codec == RGY_CODEC_HEVC) {
        add_str(RGY_LOG_INFO, _T("CU max / min   %s / %s\n"),
            get_chr_from_value(list_hevc_cu_size, m_stEncConfig.encodeCodecConfig.hevcConfig.maxCUSize),
            get_chr_from_value(list_hevc_cu_size, m_stEncConfig.encodeCodecConfig.hevcConfig.minCUSize));
    }
    if (rgy_codec == RGY_CODEC_AV1 && m_dev->encoder()->checkAPIver(12, 0)) {
        add_str(RGY_LOG_INFO, _T("Part size      max %s / min %s\n"),
            get_chr_from_value(list_part_size_av1, m_stEncConfig.encodeCodecConfig.av1Config.maxPartSize),
            get_chr_from_value(list_part_size_av1, m_stEncConfig.encodeCodecConfig.av1Config.minPartSize));
        add_str(RGY_LOG_INFO, _T("Tile num       columns %s / rows %s\n"),
            get_chr_from_value(list_av1_tiles, m_stEncConfig.encodeCodecConfig.av1Config.numTileColumns),
            get_chr_from_value(list_av1_tiles, m_stEncConfig.encodeCodecConfig.av1Config.numTileRows));
        add_str(RGY_LOG_INFO, _T("TemporalLayers max %d\n"), m_stEncConfig.encodeCodecConfig.av1Config.maxTemporalLayersMinus1+1);
        add_str(RGY_LOG_INFO, _T("Refs           forward %s, backward %s\n"),
            get_chr_from_value(list_av1_refs_forward, m_stEncConfig.encodeCodecConfig.av1Config.numFwdRefs),
            get_chr_from_value(list_av1_refs_backward, m_stEncConfig.encodeCodecConfig.av1Config.numBwdRefs));
    }
    { const auto &vui_str = m_encVUI.print_all();
        if (vui_str.length() > 0) {
            add_str(RGY_LOG_INFO,  _T("VUI            %s\n"), vui_str.c_str());
        }
    }
    if (m_hdrseiOut) {
        const auto masterdisplay = m_hdrseiOut->print_masterdisplay();
        const auto maxcll = m_hdrseiOut->print_maxcll();
        if (masterdisplay.length() > 0) {
            const tstring tstr = char_to_tstring(masterdisplay);
            const auto splitpos = tstr.find(_T("WP("));
            if (splitpos == std::string::npos) {
                add_str(RGY_LOG_INFO, _T("MasteringDisp  %s\n"), tstr.c_str());
            } else {
                add_str(RGY_LOG_INFO, _T("MasteringDisp  %s\n")
                    _T("               %s\n"),
                    tstr.substr(0, splitpos - 1).c_str(), tstr.substr(splitpos).c_str());
            }
        }
        if (maxcll.length() > 0) {
            add_str(RGY_LOG_INFO, _T("MaxCLL/MaxFALL %s\n"), char_to_tstring(maxcll).c_str());
        }
    }
    if (m_hdr10plus) {
        add_str(RGY_LOG_INFO, _T("Dynamic HDR10  %s\n"), m_hdr10plus->inputJson().c_str());
    } else if (m_hdr10plusMetadataCopy) {
        add_str(RGY_LOG_INFO, _T("Dynamic HDR10  copy\n"));
    }
    if (m_doviProfile != RGY_DOVI_PROFILE_UNSET) {
        tstring profile_copy;
        if (m_doviProfile == RGY_DOVI_PROFILE_COPY) {
            profile_copy = tstring(_T(" (")) + get_cx_desc(list_dovi_profile, m_pFileReader->getInputDOVIProfile()) + tstring(_T(")"));
        }
        add_str(RGY_LOG_INFO, _T("dovi profile   %s%s\n"), get_cx_desc(list_dovi_profile, m_doviProfile), profile_copy.c_str());
    }
    if (m_dovirpu) {
        add_str(RGY_LOG_INFO, _T("dovi rpu       %s\n"), m_dovirpu->get_filepath().c_str());
    } else if (m_dovirpuMetadataCopy) {
        add_str(RGY_LOG_INFO, _T("dovi rpu       copy\n"));
    }
    if (rgy_codec == RGY_CODEC_H264) {
        if (m_stEncConfig.encodeCodecConfig.h264Config.hierarchicalPFrames || m_stEncConfig.encodeCodecConfig.h264Config.hierarchicalBFrames) {
            add_str(RGY_LOG_INFO, _T("Hierarchical   %s%s%s Frames [temporal layers: %d]\n"),
                m_stEncConfig.encodeCodecConfig.h264Config.hierarchicalPFrames ? _T("P") : _T(""),
                m_stEncConfig.encodeCodecConfig.h264Config.hierarchicalPFrames && m_stEncConfig.encodeCodecConfig.h264Config.hierarchicalBFrames ? _T(" + ") : _T(""),
                m_stEncConfig.encodeCodecConfig.h264Config.hierarchicalBFrames ? _T("B") : _T(""),
                m_stEncConfig.encodeCodecConfig.h264Config.numTemporalLayers);
        }
    }
    add_str(RGY_LOG_INFO, _T("Others         "));
    add_str(RGY_LOG_INFO, _T("mv:%s "), get_chr_from_value(list_mv_presicion, m_stEncConfig.mvPrecision));
    if (m_stCreateEncodeParams.enableWeightedPrediction) {
        add_str(RGY_LOG_INFO, _T("weightp "));
    }
    if (m_stCreateEncodeParams.encodeConfig->rcParams.enableNonRefP) {
        add_str(RGY_LOG_INFO, _T("nonrefp "));
    }
    if (m_stCreateEncodeParams.enableUniDirectionalB) {
        add_str(RGY_LOG_INFO, _T("unidirectb "));
    }
    if (rgy_codec == RGY_CODEC_H264) {
        add_str(RGY_LOG_INFO, _T("%s "), get_chr_from_value(list_entropy_coding, m_stEncConfig.encodeCodecConfig.h264Config.entropyCodingMode));
        add_str(RGY_LOG_INFO, (m_stEncConfig.encodeCodecConfig.h264Config.disableDeblockingFilterIDC == 0) ? _T("deblock ") : _T("no_deblock "));
        add_str(RGY_LOG_DEBUG, m_stEncConfig.encodeCodecConfig.h264Config.enableVFR ? _T("VFR ") : _T(""));
        add_str(RGY_LOG_INFO,  _T("adapt-transform:%s "), get_chr_from_value(list_adapt_transform, m_stEncConfig.encodeCodecConfig.h264Config.adaptiveTransformMode));
        add_str(RGY_LOG_DEBUG, _T("fmo:%s "), get_chr_from_value(list_fmo, m_stEncConfig.encodeCodecConfig.h264Config.fmoMode));
        if (m_stCreateEncodeParams.encodeConfig->frameIntervalP - 1 > 0) {
            add_str(RGY_LOG_INFO, _T("bdirect:%s "), get_chr_from_value(list_bdirect, m_stEncConfig.encodeCodecConfig.h264Config.bdirectMode));
        }
    }
    if (rgy_codec == RGY_CODEC_H264 || rgy_codec == RGY_CODEC_HEVC) {
        if (get_outputAUD(m_stEncConfig.encodeCodecConfig, rgy_codec)) {
            add_str(RGY_LOG_INFO, _T("aud "));
        }
        if (get_outputPictureTimingSEI(m_stEncConfig.encodeCodecConfig, rgy_codec)) {
            add_str(RGY_LOG_INFO, _T("pic-struct "));
        }
        if (get_outputBufferingPeriodSEI(m_stEncConfig.encodeCodecConfig, rgy_codec)) {
            add_str(RGY_LOG_INFO, _T("buf-period "));
        }
        if (get_repeatSPSPPS(m_stEncConfig.encodeCodecConfig, rgy_codec)) {
            add_str(RGY_LOG_INFO, _T("repeat-headers "));
        }
    }
    if (get_tfLevel(m_stEncConfig.encodeCodecConfig, rgy_codec) != NV_ENC_TEMPORAL_FILTER_LEVEL_0 && m_stCreateEncodeParams.encodeConfig->frameIntervalP >= 5) {
        add_str(RGY_LOG_INFO, _T("tf%s "), get_chr_from_value(list_temporal_filter_level, get_tfLevel(m_stEncConfig.encodeCodecConfig, rgy_codec)));
    }
    if (rgy_codec == RGY_CODEC_AV1) {
        if (m_stEncConfig.encodeCodecConfig.av1Config.outputAnnexBFormat) {
            add_str(RGY_LOG_INFO, _T("annexb "));
        }
        if (m_stEncConfig.encodeCodecConfig.av1Config.enableBitstreamPadding) {
            add_str(RGY_LOG_INFO, _T("bitstream-pad "));
        }
    }
    if (rgy_codec == RGY_CODEC_HEVC || rgy_codec == RGY_CODEC_AV1) {
        if (get_enableTemporalSVC(m_stEncConfig.encodeCodecConfig, rgy_codec)) {
            add_str(RGY_LOG_INFO, _T("temporal-svc "));
        }
    }
    add_str(RGY_LOG_INFO, _T("\n"));
    return str;
}

void NVEncCore::PrintEncodingParamsInfo(int output_level) {
    PrintMes(RGY_LOG_INFO, _T("%s"), GetEncodingParamsInfo(output_level).c_str());
}
