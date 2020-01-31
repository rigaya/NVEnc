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


RGYConvertCSP::RGYConvertCSP() : RGYConvertCSP(0) {
}

RGYConvertCSP::RGYConvertCSP(int threads) :
    m_csp(nullptr),
    m_csp_from(RGY_CSP_NA),
    m_csp_to(RGY_CSP_NA),
    m_uv_only(false),
    m_threads(threads),
    m_th(), m_heStart(), m_heFin(), m_heFinCopy(),
    m_prm() {
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
const ConvertCSP *RGYConvertCSP::getFunc(RGY_CSP csp_from, RGY_CSP csp_to, bool uv_only, uint32_t simd) {
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
        const int div = (m_csp->simd == 0) ? 2 : 4;
        const int max = (m_csp->simd == 0) ? 8 : 4;
        m_threads = (dst_y_pitch_byte % 128 != 0) ? 1 : std::min(max, ((int)get_cpu_info().physical_cores + div) / div);
    }
    if (m_threads > 1 && m_th.size() == 0) {
        m_heFinCopy.clear();
        m_heStart.clear();
        m_heFin.clear();
        for (int ith = 1; ith < m_threads; ith++) {
            auto heStart = std::unique_ptr<void, handle_deleter>(CreateEvent(nullptr, false, false, nullptr), handle_deleter());
            auto heFin = std::unique_ptr<void, handle_deleter>(CreateEvent(nullptr, false, false, nullptr), handle_deleter());
            m_th.push_back(std::thread([heStart = heStart.get(), heFin = heFin.get(), ithId = ith, threadN = m_threads, prm = &m_prm, cspfunc = &m_csp]() {
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
    m_csp->func[interlaced](dst, src,
        width, src_y_pitch_byte, src_uv_pitch_byte, dst_y_pitch_byte,
        height, dst_height, 0, (int)m_th.size()+1, crop);
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
    m_trimParam() {
    m_trimParam.list.clear();
    m_trimParam.offset = 0;
    memset(&m_inputVideoInfo, 0, sizeof(m_inputVideoInfo));
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
    shared_ptr<RGYLog> log
) {
    RGYInputPrm inputPrm;
    inputPrm.threadCsp = ctrl->threadCsp;
    inputPrm.simdCsp = ctrl->simdCsp;

    for (auto &src : source) {

        VideoInfo inputInfo = *input;

        std::vector<T *> select;
        for (auto &channel : src.select) {
            auto ptr = (T *)&channel.second;
            select.push_back(ptr);
        }

        RGYInputAvcodecPrm inputInfoAVAudioReader(inputPrm);
        inputInfoAVAudioReader.readVideo = false;
        inputInfoAVAudioReader.readChapter = false;
        inputInfoAVAudioReader.readData = false;
        inputInfoAVAudioReader.videoAvgFramerate = std::make_pair(inputInfo.fpsN, inputInfo.fpsD);
        inputInfoAVAudioReader.analyzeSec = common->demuxAnalyzeSec;
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
        inputInfoAVAudioReader.logFramePosList = ctrl->logFramePosList.c_str();
        inputInfoAVAudioReader.threadInput = 0;

        shared_ptr<RGYInput> audioReader(new RGYInputAvcodec());
        auto ret = audioReader->Init(src.filename.c_str(), &inputInfo, &inputInfoAVAudioReader, log, nullptr);
        if (ret != 0) {
            log->write(RGY_LOG_ERROR, audioReader->GetInputMessage());
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

RGY_ERR initReaders(
    shared_ptr<RGYInput>& pFileReader,
    vector<shared_ptr<RGYInput>>& otherReaders,
    VideoInfo *input,
    const shared_ptr<EncodeStatus> pStatus,
    const RGYParamCommon *common,
    const RGYParamControl *ctrl,
    DeviceCodecCsp& HWDecCodecCsp,
    const int subburnTrackId,
    const bool vpp_afs,
    const bool vpp_rff,
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
        log->write(RGY_LOG_ERROR, _T("avs reader not compiled in this binary.\n"));
        return RGY_ERR_UNSUPPORTED;
    }
    if (input->type == RGY_INPUT_FMT_VPY_MT && !ENABLE_VAPOURSYNTH_READER) {
        log->write(RGY_LOG_ERROR, _T("vpy reader not compiled in this binary.\n"));
        return RGY_ERR_UNSUPPORTED;
    }
    if (input->type == RGY_INPUT_FMT_AVI && !ENABLE_AVI_READER) {
        log->write(RGY_LOG_ERROR, _T("avi reader not compiled in this binary.\n"));
        return RGY_ERR_UNSUPPORTED;
    }
    if (input->type == RGY_INPUT_FMT_AVHW && !ENABLE_AVSW_READER) {
        log->write(RGY_LOG_ERROR, _T("avcodec + cuvid reader not compiled in this binary.\n"));
        return RGY_ERR_UNSUPPORTED;
    }
    if (input->type == RGY_INPUT_FMT_AVSW && !ENABLE_AVSW_READER) {
        log->write(RGY_LOG_ERROR, _T("avsw reader not compiled in this binary.\n"));
        return RGY_ERR_UNSUPPORTED;
    }
    auto check_if_avsw_or_avhw = [](RGY_INPUT_FMT input_type) {
        return input_type == RGY_INPUT_FMT_AVHW
            || input_type == RGY_INPUT_FMT_AVSW
            || input_type == RGY_INPUT_FMT_AVANY;
    };
    if (input->picstruct == RGY_PICSTRUCT_AUTO && !check_if_avsw_or_avhw(input->type)) {
        log->write(RGY_LOG_ERROR, _T("--interlace auto is only supported with avsw/avhw reader.\n"));
        return RGY_ERR_UNSUPPORTED;
    }

    RGYInputPrm inputPrm;
    inputPrm.threadCsp = ctrl->threadCsp;
    inputPrm.simdCsp = ctrl->simdCsp;
    RGYInputPrm *pInputPrm = &inputPrm;

    auto subBurnTrack = std::make_unique<SubtitleSelect>();
    SubtitleSelect *subBurnTrackPtr = subBurnTrack.get();

    RGYInputAvsPrm inputPrmAvs(inputPrm);
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
        log->write(RGY_LOG_DEBUG, _T("avi reader selected.\n"));
        pFileReader.reset(new RGYInputAvi());
        break;
#endif //ENABLE_AVI_READER
#if ENABLE_AVISYNTH_READER
    case RGY_INPUT_FMT_AVS:
        inputPrmAvs.readAudio = common->nAudioSelectCount > 0;
        pInputPrm = &inputPrmAvs;
        log->write(RGY_LOG_DEBUG, _T("avs reader selected.\n"));
        pFileReader.reset(new RGYInputAvs());
        break;
#endif //ENABLE_AVISYNTH_READER
#if ENABLE_VAPOURSYNTH_READER
    case RGY_INPUT_FMT_VPY:
    case RGY_INPUT_FMT_VPY_MT:
        log->write(RGY_LOG_DEBUG, _T("vpy reader selected.\n"));
        pFileReader.reset(new RGYInputVpy());
        break;
#endif //ENABLE_VAPOURSYNTH_READER
#if ENABLE_AVSW_READER
    case RGY_INPUT_FMT_AVHW:
    case RGY_INPUT_FMT_AVSW:
    case RGY_INPUT_FMT_AVANY: {
        subBurnTrack->trackID = subburnTrackId;
        inputInfoAVCuvid.threadCsp = ctrl->threadCsp;
        inputInfoAVCuvid.simdCsp = ctrl->simdCsp;
        inputInfoAVCuvid.pInputFormat = common->AVInputFormat;
        inputInfoAVCuvid.readVideo = true;
        inputInfoAVCuvid.videoTrack = common->videoTrack;
        inputInfoAVCuvid.videoStreamId = common->videoStreamId;
        inputInfoAVCuvid.readAudio = common->nAudioSelectCount > 0;
        inputInfoAVCuvid.readSubtitle = (common->nSubtitleSelectCount > 0) || (subburnTrackId > 0);
        inputInfoAVCuvid.readData = common->nDataSelectCount > 0;
        inputInfoAVCuvid.readChapter = true;
        inputInfoAVCuvid.videoAvgFramerate = std::make_pair(input->fpsN, input->fpsD);
        inputInfoAVCuvid.analyzeSec = common->demuxAnalyzeSec;
        inputInfoAVCuvid.nTrimCount = common->nTrimCount;
        inputInfoAVCuvid.pTrimList = common->pTrimList;
        inputInfoAVCuvid.trackStartAudio = sourceAudioTrackIdStart;
        inputInfoAVCuvid.trackStartSubtitle = sourceSubtitleTrackIdStart;
        inputInfoAVCuvid.trackStartData = sourceDataTrackIdStart;
        inputInfoAVCuvid.nAudioSelectCount = common->nAudioSelectCount;
        inputInfoAVCuvid.ppAudioSelect = common->ppAudioSelectList;
        inputInfoAVCuvid.ppSubtitleSelect = (subburnTrackId) ? &subBurnTrackPtr : common->ppSubtitleSelectList;
        inputInfoAVCuvid.nSubtitleSelectCount = (subburnTrackId) ? 1 : common->nSubtitleSelectCount;
        inputInfoAVCuvid.ppDataSelect = common->ppDataSelectList;
        inputInfoAVCuvid.nDataSelectCount = common->nDataSelectCount;
        inputInfoAVCuvid.procSpeedLimit = ctrl->procSpeedLimit;
        inputInfoAVCuvid.AVSyncMode = RGY_AVSYNC_ASSUME_CFR;
        inputInfoAVCuvid.seekSec = common->seekSec;
        inputInfoAVCuvid.logFramePosList = ctrl->logFramePosList.c_str();
        inputInfoAVCuvid.threadInput = ctrl->threadInput;
        inputInfoAVCuvid.queueInfo = (perfMonitor) ? perfMonitor->GetQueueInfoPtr() : nullptr;
        inputInfoAVCuvid.HWDecCodecCsp = &HWDecCodecCsp;
        inputInfoAVCuvid.videoDetectPulldown = !vpp_rff && !vpp_afs && common->AVSyncMode == RGY_AVSYNC_ASSUME_CFR;
        inputInfoAVCuvid.caption2ass = common->caption2ass;
        inputInfoAVCuvid.pasrseHDRmetadata = common->maxCll == maxCLLSource || common->masterDisplay == masterDisplaySource;
        inputInfoAVCuvid.interlaceAutoFrame = input->picstruct == RGY_PICSTRUCT_AUTO;
        pInputPrm = &inputInfoAVCuvid;
        log->write(RGY_LOG_DEBUG, _T("avhw reader selected.\n"));
        pFileReader.reset(new RGYInputAvcodec());
        } break;
#endif //#if ENABLE_AVSW_READER
#if ENABLE_SM_READER
    case RGY_INPUT_FMT_SM: {
        log->write(RGY_LOG_DEBUG, _T("shared mem reader selected.\n"));
        pInputPrm = &inputPrmSM;
        pFileReader.reset(new RGYInputSM());
        } break;
#endif //#if ENABLE_SM_READER
    case RGY_INPUT_FMT_RAW:
    case RGY_INPUT_FMT_Y4M:
    default: {
        if (input->type == RGY_INPUT_FMT_RAW &&
            (input->fpsN <= 0 || input->fpsD <= 0)) {
            log->write(RGY_LOG_ERROR, _T("Please set fps when using raw input.\n"));
            return RGY_ERR_UNSUPPORTED;
        }
        log->write(RGY_LOG_DEBUG, _T("raw/y4m reader selected.\n"));
        pFileReader.reset(new RGYInputRaw());
        break; }
    }
    log->write(RGY_LOG_DEBUG, _T("InitInput: input selected : %d.\n"), input->type);

    VideoInfo inputParamCopy = *input;
    auto ret = pFileReader->Init(common->inputFilename.c_str(), input, pInputPrm, log, pStatus);
    if (ret != 0) {
        log->write(RGY_LOG_ERROR, pFileReader->GetInputMessage());
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
        common->audioSource, input, common, ctrl, log)) != RGY_ERR_NONE) {
        return ret;
    }
    if ((ret = initOtherReaders<true>(otherReaders,
        sourceAudioTrackIdStart, sourceSubtitleTrackIdStart, sourceDataTrackIdStart,
        common->subSource, input, common, ctrl, log)) != RGY_ERR_NONE) {
        return ret;
    }
    return RGY_ERR_NONE;
#else
    return RGY_ERR_INVALID_CALL;
#endif //ENABLE_RAW_READER
}

#endif //#if !FOR_AUO
