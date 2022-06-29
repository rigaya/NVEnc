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
#include <iostream>
#include <fstream>
#include <set>
#include "rgy_input.h"
#include "rgy_filesystem.h"
#include "cpu_info.h"

RGYConvertCSPPrm::RGYConvertCSPPrm() :
    abort(false),
    dst(nullptr),
    src(nullptr),
    interlaced(false),
    width(0),
    src_y_pitch_byte(0),
    src_uv_pitch_byte(0),
    dst_y_pitch_byte(0),
    height(0),
    dst_height(0),
    crop(nullptr) {

}


RGYConvertCSP::RGYConvertCSP() : RGYConvertCSP(0, RGYParamThread()) {
}

RGYConvertCSP::RGYConvertCSP(int threads, RGYParamThread threadParam) :
    m_csp(nullptr),
    m_csp_from(RGY_CSP_NA),
    m_csp_to(RGY_CSP_NA),
    m_uv_only(false),
    m_threads(threads),
    m_th(), m_heStart(), m_heFin(), m_heFinCopy(),
    m_threadParam(threadParam), m_prm() {
};

RGYConvertCSP::~RGYConvertCSP() {
    m_prm.abort = true;
    for (size_t i = 0; i < m_heStart.size(); i++) {
        SetEvent(m_heStart[i].get());
    }
    for (size_t i = 0; i < m_th.size(); i++) {
        m_th[i].join();
    }
    m_heFinCopy.clear();
    m_heStart.clear();
    m_heFin.clear();
    m_th.clear();
};
const ConvertCSP *RGYConvertCSP::getFunc(RGY_CSP csp_from, RGY_CSP csp_to, bool uv_only, RGY_SIMD simd) {
    if (m_csp == nullptr
        || (m_csp_from != csp_from || m_csp_to != csp_to || m_uv_only != uv_only)) {
        m_csp_from = csp_from;
        m_csp_to = csp_to;
        m_uv_only = uv_only;
        m_csp = get_convert_csp_func(csp_from, csp_to, uv_only, simd);
    }
    return m_csp;
}

int RGYConvertCSP::run(int interlaced, void **dst, const void **src, int width, int src_y_pitch_byte, int src_uv_pitch_byte, int dst_y_pitch_byte, int height, int dst_height, int *crop) {
    if (m_threads == 0) {
        const int div = (m_csp->simd == RGY_SIMD::NONE) ? 2 : 4;
        const int max = (m_csp->simd == RGY_SIMD::NONE) ? 8 : 4;
        m_threads = (dst_y_pitch_byte % 128 != 0) ? 1 : std::min(max, ((int)get_cpu_info().physical_cores + div) / div);
    }
    if (m_threads > 1 && m_th.size() == 0) {
        m_heFinCopy.clear();
        m_heStart.clear();
        m_heFin.clear();
        for (int ith = 0; ith < m_threads; ith++) {
            auto heStart = std::unique_ptr<void, handle_deleter>(CreateEvent(nullptr, false, false, nullptr), handle_deleter());
            auto heFin = std::unique_ptr<void, handle_deleter>(CreateEvent(nullptr, false, false, nullptr), handle_deleter());
            m_th.push_back(std::thread([heStart = heStart.get(), heFin = heFin.get(), ithId = ith, threadN = m_threads, threadParam = m_threadParam, prm = &m_prm, cspfunc = &m_csp]() {
                threadParam.apply(GetCurrentThread());
                WaitForSingleObject((HANDLE)heStart, INFINITE);
                while (!prm->abort) {
                    (*cspfunc)->func[prm->interlaced](prm->dst, prm->src,
                        prm->width, prm->src_y_pitch_byte, prm->src_uv_pitch_byte, prm->dst_y_pitch_byte,
                        prm->height, prm->dst_height, ithId, threadN, prm->crop);
                    SetEvent((HANDLE)heFin);
                    WaitForSingleObject((HANDLE)heStart, INFINITE);
                }
            }));
            m_heFinCopy.push_back(heFin.get());
            m_heStart.push_back(std::move(heStart));
            m_heFin.push_back(std::move(heFin));
        }
    }
    m_prm.abort = false;
    m_prm.interlaced = interlaced;
    m_prm.dst = dst;
    m_prm.src = src;
    m_prm.width = width;
    m_prm.src_y_pitch_byte = src_y_pitch_byte;
    m_prm.src_uv_pitch_byte = src_uv_pitch_byte;
    m_prm.dst_y_pitch_byte = dst_y_pitch_byte;
    m_prm.height = height;
    m_prm.dst_height = dst_height;
    m_prm.crop = crop;
    for (size_t i = 0; i < m_heStart.size(); i++) {
        SetEvent(m_heStart[i].get());
    }
    if (m_threads == 1) {
        m_csp->func[interlaced](dst, src,
            width, src_y_pitch_byte, src_uv_pitch_byte, dst_y_pitch_byte,
            height, dst_height, 0, 1, crop);
    }
    if (m_th.size() > 0) {
        WaitForMultipleObjects((uint32_t)m_heFinCopy.size(), m_heFinCopy.data(), TRUE, INFINITE);
    }
    return 0;
}

#if !FOR_AUO

std::vector<int> read_keyfile(tstring keyfile) {
    std::set<int> s; //重複回避のため
    std::ifstream ifs(keyfile);
    if (ifs.is_open()) {
        std::string buff;
        while (std::getline(ifs, buff)) {
            if (buff.length() > 0) {
                try {
                    s.insert(std::stoi(buff));
                } catch (...) {
                    return vector<int>();
                }
            }
        }
    }
    return vector<int>(s.begin(), s.end());
}

RGYInput::RGYInput() :
    m_encSatusInfo(),
    m_inputVideoInfo(),
    m_inputCsp(RGY_CSP_NA),
    m_convert(nullptr),
    m_printMes(),
    m_inputInfo(),
    m_readerName(_T("unknown")),
    m_trimParam(),
    m_poolPkt(nullptr),
    m_poolFrame(nullptr) {
    m_trimParam.list.clear();
    m_trimParam.offset = 0;
}

RGYInput::~RGYInput() {
    Close();
}

void RGYInput::Close() {
    AddMessage(RGY_LOG_DEBUG, _T("Closing...\n"));

    m_encSatusInfo.reset();
    m_convert = nullptr;

    m_inputInfo.clear();

    m_trimParam.list.clear();
    m_trimParam.offset = 0;
    m_poolPkt = nullptr;
    m_poolFrame = nullptr;
    AddMessage(RGY_LOG_DEBUG, _T("Close...\n"));
    m_printMes.reset();
}

void RGYInput::CreateInputInfo(const TCHAR *inputTypeName, const TCHAR *inputCSpName, const TCHAR *outputCSpName, const TCHAR *convSIMD, const VideoInfo *inputPrm) {
    std::basic_stringstream<TCHAR> ss;

    ss << inputTypeName;
    ss << _T("(") << inputCSpName << _T(")");
    ss << _T("->") << outputCSpName;
    if (convSIMD && _tcslen(convSIMD)) {
        ss << _T(" [") << convSIMD << _T("]");
    }
    ss << _T(", ");
    ss << inputPrm->srcWidth << _T("x") << inputPrm->srcHeight << _T(", ");
    ss << inputPrm->fpsN << _T("/") << inputPrm->fpsD << _T(" fps");

    if (cropEnabled(inputPrm->crop)) {
        ss << " crop(" << inputPrm->crop.e.left << "," << inputPrm->crop.e.up << "," << inputPrm->crop.e.right << "," << inputPrm->crop.e.bottom << ")";
    }

    m_inputInfo = ss.str();
}

#include "rgy_avutil.h"
#include "rgy_input_raw.h"
#include "rgy_input_avi.h"
#include "rgy_input_avs.h"
#include "rgy_input_vpy.h"
#include "rgy_input_sm.h"
#include "rgy_input_avcodec.h"

#if ENABLE_AVSW_READER
template<bool subtitle, typename T>
static RGY_ERR initOtherReaders(
    vector<shared_ptr<RGYInput>> &otherReaders,
    int& sourceAudioTrackIdStart,
    int& sourceSubtitleTrackIdStart,
    int& sourceDataTrackIdStart,
    const vector<T>& source,
    const VideoInfo *input,
    const RGYParamCommon *common,
    const RGYParamControl *ctrl,
    RGYPoolAVPacket *poolPkt,
    RGYPoolAVFrame *poolFrame,
    shared_ptr<RGYLog> log
) {
    RGYInputPrm inputPrm;
    inputPrm.threadCsp = ctrl->threadCsp;
    inputPrm.simdCsp = ctrl->simdCsp;
    inputPrm.threadParamCsp = ctrl->threadParams.get(RGYThreadType::CSP);

    for (int ifile = 0; ifile < (int)source.size(); ifile++) {
        auto& src = source[ifile];
        VideoInfo inputInfo = *input;

        std::vector<T *> select;
        for (auto &channel : src.select) {
            auto ptr = (T *)&channel.second;
            select.push_back(ptr);
        }

        RGYInputAvcodecPrm inputInfoAVAudioReader(inputPrm);
        inputInfoAVAudioReader.poolPkt = poolPkt;
        inputInfoAVAudioReader.poolFrame = poolFrame;
        inputInfoAVAudioReader.readVideo = false;
        inputInfoAVAudioReader.readChapter = false;
        inputInfoAVAudioReader.readData = false;
        inputInfoAVAudioReader.fileIndex = ifile;
        inputInfoAVAudioReader.videoAvgFramerate = rgy_rational<int>(inputInfo.fpsN, inputInfo.fpsD);
        inputInfoAVAudioReader.inputRetry = common->inputRetry;
        inputInfoAVAudioReader.analyzeSec = common->demuxAnalyzeSec;
        inputInfoAVAudioReader.probesize = common->demuxProbesize;
        inputInfoAVAudioReader.nTrimCount = common->nTrimCount;
        inputInfoAVAudioReader.pTrimList = common->pTrimList;
        inputInfoAVAudioReader.trackStartAudio = sourceAudioTrackIdStart;
        inputInfoAVAudioReader.trackStartSubtitle = sourceSubtitleTrackIdStart;
        inputInfoAVAudioReader.trackStartData = sourceDataTrackIdStart;
        if (subtitle) {
            inputInfoAVAudioReader.readAudio = false;
            inputInfoAVAudioReader.readSubtitle = true;
            inputInfoAVAudioReader.nSubtitleSelectCount = (int)select.size();
            inputInfoAVAudioReader.ppSubtitleSelect = (SubtitleSelect **)select.data();
        } else {
            inputInfoAVAudioReader.readAudio = true;
            inputInfoAVAudioReader.readSubtitle = false;
            inputInfoAVAudioReader.nAudioSelectCount = (int)select.size();
            inputInfoAVAudioReader.ppAudioSelect = (AudioSelect **)select.data();
        }
        inputInfoAVAudioReader.procSpeedLimit = ctrl->procSpeedLimit;
        inputInfoAVAudioReader.AVSyncMode = RGY_AVSYNC_ASSUME_CFR;
        inputInfoAVAudioReader.seekSec = common->seekSec;
        inputInfoAVAudioReader.logFramePosList = (ctrl->logFramePosList) ? src.filename + _T(".framelist.csv") : _T("");
        inputInfoAVAudioReader.threadInput = 0;
        inputInfoAVAudioReader.threadParamInput = ctrl->threadParams.get(RGYThreadType::INPUT);

        shared_ptr<RGYInput> audioReader(new RGYInputAvcodec());
        auto ret = audioReader->Init(src.filename.c_str(), &inputInfo, &inputInfoAVAudioReader, log, nullptr);
        if (ret != 0) {
            log->write(RGY_LOG_ERROR, RGY_LOGT_IN, audioReader->GetInputMessage());
            return ret;
        }
        sourceAudioTrackIdStart += audioReader->GetAudioTrackCount();
        sourceSubtitleTrackIdStart += audioReader->GetSubtitleTrackCount();
        sourceDataTrackIdStart += audioReader->GetDataTrackCount();
        otherReaders.push_back(std::move(audioReader));
    }
    return RGY_ERR_NONE;
}
#endif

static bool check_if_avhw_or_avsw(RGY_INPUT_FMT input_type) {
    return input_type == RGY_INPUT_FMT_AVHW
        || input_type == RGY_INPUT_FMT_AVSW
        || input_type == RGY_INPUT_FMT_AVANY;
};

template<typename T>
bool check_avhw_avsw_only(const T& target, const T& autoval, const char *name, RGYLog *log) {
    if (target == autoval) {
        log->write(RGY_LOG_ERROR, RGY_LOGT_IN, _T("\"%s\" is only supported with avsw/avhw reader.\n"), char_to_tstring(name).c_str());
        return true;
    }
    return false;
}

RGY_ERR initReaders(
    shared_ptr<RGYInput>& pFileReader,
    vector<shared_ptr<RGYInput>>& otherReaders,
    VideoInfo *input,
    const RGY_CSP inputCspOfRawReader,
    const shared_ptr<EncodeStatus> pStatus,
    const RGYParamCommon *common,
    const RGYParamControl *ctrl,
    DeviceCodecCsp& HWDecCodecCsp,
    const int subburnTrackId,
    const bool vpp_afs,
    const bool vpp_rff,
    RGYPoolAVPacket *poolPkt,
    RGYPoolAVFrame *poolFrame,
    RGYListRef<RGYFrameDataQP> *qpTableListRef,
    CPerfMonitor *perfMonitor,
    shared_ptr<RGYLog> log
) {
    int sourceAudioTrackIdStart = 1;    //トラック番号は1スタート
    int sourceSubtitleTrackIdStart = 1; //トラック番号は1スタート
    int sourceDataTrackIdStart = 1;     //トラック番号は1スタート

#if ENABLE_RAW_READER
    if (input->type == RGY_INPUT_FMT_AUTO) {
        if (check_ext(common->inputFilename, { ".y4m" })) {
            input->type = RGY_INPUT_FMT_Y4M;
        } else if (check_ext(common->inputFilename, { ".yuv" })) {
            input->type = RGY_INPUT_FMT_RAW;
#if ENABLE_AVI_READER
        } else if (check_ext(common->inputFilename, { ".avi" })) {
            input->type = RGY_INPUT_FMT_AVI;
#endif
#if ENABLE_AVISYNTH_READER
        } else if (check_ext(common->inputFilename, { ".avs" })) {
            input->type = RGY_INPUT_FMT_AVS;
#endif
#if ENABLE_VAPOURSYNTH_READER
        } else if (check_ext(common->inputFilename, { ".vpy" })) {
            input->type = RGY_INPUT_FMT_VPY_MT;
#endif
        } else {
#if ENABLE_AVSW_READER
            input->type = RGY_INPUT_FMT_AVANY;
#else
            input->type = RGY_INPUT_FMT_RAW;
#endif
        }
    }

    //Check if selected format is enabled
    if (input->type == RGY_INPUT_FMT_AVS && !ENABLE_AVISYNTH_READER) {
        log->write(RGY_LOG_ERROR, RGY_LOGT_IN, _T("avs reader not compiled in this binary.\n"));
        return RGY_ERR_UNSUPPORTED;
    }
    if (input->type == RGY_INPUT_FMT_VPY_MT && !ENABLE_VAPOURSYNTH_READER) {
        log->write(RGY_LOG_ERROR, RGY_LOGT_IN, _T("vpy reader not compiled in this binary.\n"));
        return RGY_ERR_UNSUPPORTED;
    }
    if (input->type == RGY_INPUT_FMT_AVI && !ENABLE_AVI_READER) {
        log->write(RGY_LOG_ERROR, RGY_LOGT_IN, _T("avi reader not compiled in this binary.\n"));
        return RGY_ERR_UNSUPPORTED;
    }
    if (input->type == RGY_INPUT_FMT_AVHW && !ENABLE_AVSW_READER) {
        log->write(RGY_LOG_ERROR, RGY_LOGT_IN, _T("avcodec + cuvid reader not compiled in this binary.\n"));
        return RGY_ERR_UNSUPPORTED;
    }
    if (input->type == RGY_INPUT_FMT_AVSW && !ENABLE_AVSW_READER) {
        log->write(RGY_LOG_ERROR, RGY_LOGT_IN, _T("avsw reader not compiled in this binary.\n"));
        return RGY_ERR_UNSUPPORTED;
    }
    if (!check_if_avhw_or_avsw(input->type)) {
        if (input->type != RGY_INPUT_FMT_Y4M) {
            if (check_avhw_avsw_only(input->picstruct, RGY_PICSTRUCT_AUTO, "--interlace auto", log.get())) return RGY_ERR_UNSUPPORTED;
        }
        if (check_avhw_avsw_only(input->picstruct,           RGY_PICSTRUCT_AUTO,  "--interlace auto",   log.get())) return RGY_ERR_UNSUPPORTED;
        if (check_avhw_avsw_only(common->out_vui.chromaloc,  RGY_CHROMALOC_AUTO,  "--chromaloc auto",   log.get())) return RGY_ERR_UNSUPPORTED;
        if (check_avhw_avsw_only(common->out_vui.format,     COLOR_VALUE_AUTO,    "--videoformat auto", log.get())) return RGY_ERR_UNSUPPORTED;
        if (check_avhw_avsw_only(common->out_vui.matrix,     RGY_MATRIX_AUTO,     "--colormatrix auto", log.get())) return RGY_ERR_UNSUPPORTED;
        if (check_avhw_avsw_only(common->out_vui.colorprim,  RGY_PRIM_AUTO,       "--colorprim auto",   log.get())) return RGY_ERR_UNSUPPORTED;
        if (check_avhw_avsw_only(common->out_vui.transfer,   RGY_TRANSFER_AUTO,   "--transfer auto",    log.get())) return RGY_ERR_UNSUPPORTED;
        if (check_avhw_avsw_only(common->out_vui.colorrange, RGY_COLORRANGE_AUTO, "--colorrange auto",  log.get())) return RGY_ERR_UNSUPPORTED;
        if (check_avhw_avsw_only<std::string>(common->maxCll, maxCLLSource,       "--maxcll copy",      log.get())) return RGY_ERR_UNSUPPORTED;
        if (check_avhw_avsw_only<std::string>(common->masterDisplay, masterDisplaySource, "--master-dsiplay copy", log.get())) return RGY_ERR_UNSUPPORTED;
    }

    RGYInputPrm inputPrm;
    inputPrm.threadCsp = ctrl->threadCsp;
    inputPrm.simdCsp = ctrl->simdCsp;
    inputPrm.threadParamCsp = ctrl->threadParams.get(RGYThreadType::CSP);
    inputPrm.poolPkt = poolPkt;
    inputPrm.poolFrame = poolFrame;
    log->write(RGY_LOG_DEBUG, RGY_LOGT_IN, _T("Set csp thread param: %s.\n"), inputPrm.threadParamCsp.desc().c_str());
    RGYInputPrm *pInputPrm = &inputPrm;

    std::unique_ptr<SubtitleSelect> subBurnTrack;
    auto subTitleSelectList = make_vector<SubtitleSelect *>(common->ppSubtitleSelectList, common->nSubtitleSelectCount);
    if (subburnTrackId > 0) {
        if (std::find_if(subTitleSelectList.begin(), subTitleSelectList.end(), [](SubtitleSelect *sub) {
            return (sub->trackID == 0);
            }) == subTitleSelectList.end()) {
            subBurnTrack = std::make_unique<SubtitleSelect>();
            subBurnTrack->trackID = subburnTrackId;
            subTitleSelectList.push_back(subBurnTrack.get());
        }
    }

    RGYInputPrmRaw inputPrmRaw(inputPrm);
    inputPrmRaw.inputCsp = inputCspOfRawReader;
#if ENABLE_AVISYNTH_READER
    RGYInputAvsPrm inputPrmAvs(inputPrm);
#endif
#if ENABLE_AVSW_READER
    RGYInputAvcodecPrm inputInfoAVCuvid(inputPrm);
#endif
#if ENABLE_SM_READER
    RGYInputSMPrm inputPrmSM(inputPrm);
    inputPrmSM.parentProcessID = ctrl->parentProcessID;
#endif

    switch (input->type) {
#if ENABLE_AVI_READER
    case RGY_INPUT_FMT_AVI:
        log->write(RGY_LOG_DEBUG, RGY_LOGT_IN, _T("avi reader selected.\n"));
        pFileReader.reset(new RGYInputAvi());
        break;
#endif //ENABLE_AVI_READER
#if ENABLE_AVISYNTH_READER
    case RGY_INPUT_FMT_AVS:
        inputPrmAvs.readAudio = common->nAudioSelectCount > 0;
        inputPrmAvs.avsdll = ctrl->avsdll;
        pInputPrm = &inputPrmAvs;
        log->write(RGY_LOG_DEBUG, RGY_LOGT_IN, _T("avs reader selected.\n"));
        pFileReader.reset(new RGYInputAvs());
        break;
#endif //ENABLE_AVISYNTH_READER
#if ENABLE_VAPOURSYNTH_READER
    case RGY_INPUT_FMT_VPY:
    case RGY_INPUT_FMT_VPY_MT:
        log->write(RGY_LOG_DEBUG, RGY_LOGT_IN, _T("vpy reader selected.\n"));
        pFileReader.reset(new RGYInputVpy());
        break;
#endif //ENABLE_VAPOURSYNTH_READER
#if ENABLE_AVSW_READER
    case RGY_INPUT_FMT_AVHW:
    case RGY_INPUT_FMT_AVSW:
    case RGY_INPUT_FMT_AVANY: {
        inputInfoAVCuvid.threadCsp = ctrl->threadCsp;
        inputInfoAVCuvid.simdCsp = ctrl->simdCsp;
        inputInfoAVCuvid.pInputFormat = common->AVInputFormat;
        inputInfoAVCuvid.readVideo = true;
        inputInfoAVCuvid.videoTrack = common->videoTrack;
        inputInfoAVCuvid.videoStreamId = common->videoStreamId;
        inputInfoAVCuvid.readAudio = common->nAudioSelectCount > 0;
        inputInfoAVCuvid.readSubtitle = (common->nSubtitleSelectCount > 0) || (subburnTrackId > 0);
        inputInfoAVCuvid.readData = common->nDataSelectCount > 0;
        inputInfoAVCuvid.readAttachment = common->nAttachmentSelectCount > 0;
        inputInfoAVCuvid.readChapter = true;
        inputInfoAVCuvid.videoAvgFramerate = rgy_rational<int>(input->fpsN, input->fpsD);
        inputInfoAVCuvid.analyzeSec = common->demuxAnalyzeSec;
        inputInfoAVCuvid.probesize = common->demuxProbesize;
        inputInfoAVCuvid.inputRetry = common->inputRetry;
        inputInfoAVCuvid.nTrimCount = common->nTrimCount;
        inputInfoAVCuvid.pTrimList = common->pTrimList;
        inputInfoAVCuvid.fileIndex = -1; //動画ファイルは-1
        inputInfoAVCuvid.trackStartAudio = sourceAudioTrackIdStart;
        inputInfoAVCuvid.trackStartSubtitle = sourceSubtitleTrackIdStart;
        inputInfoAVCuvid.trackStartData = sourceDataTrackIdStart;
        inputInfoAVCuvid.nAudioSelectCount = common->nAudioSelectCount;
        inputInfoAVCuvid.ppAudioSelect = common->ppAudioSelectList;
        inputInfoAVCuvid.ppSubtitleSelect = subTitleSelectList.data();
        inputInfoAVCuvid.nSubtitleSelectCount = (int)subTitleSelectList.size();
        inputInfoAVCuvid.ppDataSelect = common->ppDataSelectList;
        inputInfoAVCuvid.nDataSelectCount = common->nDataSelectCount;
        inputInfoAVCuvid.ppAttachmentSelect = common->ppAttachmentSelectList;
        inputInfoAVCuvid.nAttachmentSelectCount = common->nAttachmentSelectCount;
        inputInfoAVCuvid.procSpeedLimit = ctrl->procSpeedLimit;
        inputInfoAVCuvid.AVSyncMode = RGY_AVSYNC_ASSUME_CFR;
        inputInfoAVCuvid.seekSec = common->seekSec;
        inputInfoAVCuvid.logFramePosList = (ctrl->logFramePosList) ? common->outputFilename + _T(".framelist.csv") : _T("");
        inputInfoAVCuvid.logPackets = (ctrl->logPacketsList) ? common->outputFilename + _T(".packets.csv") : _T("");
        inputInfoAVCuvid.threadInput = ctrl->threadInput;
        inputInfoAVCuvid.threadParamInput = ctrl->threadParams.get(RGYThreadType::INPUT);
        inputInfoAVCuvid.queueInfo = (perfMonitor) ? perfMonitor->GetQueueInfoPtr() : nullptr;
        inputInfoAVCuvid.HWDecCodecCsp = &HWDecCodecCsp;
        inputInfoAVCuvid.videoDetectPulldown = !vpp_rff && !vpp_afs && common->AVSyncMode == RGY_AVSYNC_ASSUME_CFR;
        inputInfoAVCuvid.caption2ass = common->caption2ass;
        inputInfoAVCuvid.hdr10plusMetadataCopy = common->hdr10plusMetadataCopy;
        inputInfoAVCuvid.parseHDRmetadata = common->maxCll == maxCLLSource || common->masterDisplay == masterDisplaySource;
        inputInfoAVCuvid.interlaceAutoFrame = input->picstruct == RGY_PICSTRUCT_AUTO;
        inputInfoAVCuvid.qpTableListRef = qpTableListRef;
        inputInfoAVCuvid.inputOpt = common->inputOpt;
        inputInfoAVCuvid.lowLatency = ctrl->lowLatency;
        pInputPrm = &inputInfoAVCuvid;
        log->write(RGY_LOG_DEBUG, RGY_LOGT_IN, _T("avhw reader selected.\n"));
        pFileReader.reset(new RGYInputAvcodec());
        } break;
#endif //#if ENABLE_AVSW_READER
#if ENABLE_SM_READER
    case RGY_INPUT_FMT_SM: {
        log->write(RGY_LOG_DEBUG, RGY_LOGT_IN, _T("shared mem reader selected.\n"));
        pInputPrm = &inputPrmSM;
        pFileReader.reset(new RGYInputSM());
        } break;
#endif //#if ENABLE_SM_READER
    case RGY_INPUT_FMT_RAW:
    case RGY_INPUT_FMT_Y4M:
    default: {
        if (input->type == RGY_INPUT_FMT_RAW &&
            (input->fpsN <= 0 || input->fpsD <= 0)) {
            log->write(RGY_LOG_ERROR, RGY_LOGT_IN, _T("Please set fps when using raw input.\n"));
            return RGY_ERR_UNSUPPORTED;
        }
        pInputPrm = &inputPrmRaw;
        log->write(RGY_LOG_DEBUG, RGY_LOGT_IN, _T("raw/y4m reader selected.\n"));
        pFileReader.reset(new RGYInputRaw());
        break; }
    }
    log->write(RGY_LOG_DEBUG, RGY_LOGT_IN, _T("InitInput: input selected : %d.\n"), input->type);

    VideoInfo inputParamCopy = *input;
    auto ret = pFileReader->Init(common->inputFilename.c_str(), input, pInputPrm, log, pStatus);
    if (ret != 0) {
        log->write(RGY_LOG_ERROR, RGY_LOGT_IN, pFileReader->GetInputMessage());
        return ret;
    }
    sourceAudioTrackIdStart    += pFileReader->GetAudioTrackCount();
    sourceSubtitleTrackIdStart += pFileReader->GetSubtitleTrackCount();
    sourceDataTrackIdStart     += pFileReader->GetDataTrackCount();

    //ユーザー指定のオプションを必要に応じて復元する
    input->picstruct = inputParamCopy.picstruct;
    if (inputParamCopy.fpsN * inputParamCopy.fpsD > 0) {
        input->fpsN = inputParamCopy.fpsN;
        input->fpsD = inputParamCopy.fpsD;
    }
    if (inputParamCopy.sar[0] * inputParamCopy.sar[1] > 0) {
        input->sar[0] = inputParamCopy.sar[0];
        input->sar[1] = inputParamCopy.sar[1];
    }

#if ENABLE_AVSW_READER
#endif //#if ENABLE_AVSW_READER
    if ((ret = initOtherReaders<false>(otherReaders,
        sourceAudioTrackIdStart, sourceSubtitleTrackIdStart, sourceDataTrackIdStart,
        common->audioSource, input, common, ctrl, poolPkt, poolFrame, log)) != RGY_ERR_NONE) {
        return ret;
    }
    if ((ret = initOtherReaders<true>(otherReaders,
        sourceAudioTrackIdStart, sourceSubtitleTrackIdStart, sourceDataTrackIdStart,
        common->subSource, input, common, ctrl, poolPkt, poolFrame, log)) != RGY_ERR_NONE) {
        return ret;
    }
    return RGY_ERR_NONE;
#else
    return RGY_ERR_INVALID_CALL;
#endif //ENABLE_RAW_READER
}

#endif //#if !FOR_AUO
