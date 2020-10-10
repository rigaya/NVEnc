// -----------------------------------------------------------------------------------------
// NVEnc by rigaya
// -----------------------------------------------------------------------------------------
//
// The MIT License
//
// Copyright (c) 2019 rigaya
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

#include <map>
#include <libvmaf.h>
#include "rgy_avutil.h"
#include "CuvidDecode.h"
#include "NVEncFilterSsim.h"
#include "NVEncParam.h"

#if ENABLE_SSIM

static double ssim_db(double ssim, double weight) {
    return 10.0 * log10(weight / (weight - ssim));
}

static double get_psnr(double mse, uint64_t nb_frames, int max) {
    return 10.0 * log10((max * max) / (mse / nb_frames));
}

tstring NVEncFilterParamSsim::print() const {
    tstring str;
    if (ssim) str += _T("ssim ");
    if (psnr) str += _T("psnr ");
    if (vmaf.enable) str += _T("vmaf ");
    return str;
}

NVEncFilterSsim::NVEncFilterSsim() :
    m_decodeStarted(false),
    m_deviceId(0),
    m_thread(),
    m_mtx(),
    m_abort(false),
    m_vidctxlock(),
    m_input(),
    m_unused(),
    m_decoder(),
    m_crop(),
    m_cropDToH(),
    m_frameHostSendIndex(0),
    m_frameHostOrg(),
    m_frameHostEnc(),
    m_vmaf(),
    m_decFrameCopy(),
    m_tmpSsim(),
    m_tmpPsnr(),
    m_cropEvent(),
    m_streamCrop(),
    m_streamCalcSsim(),
    m_streamCalcPsnr(),
    m_planeCoef(),
    m_ssimTotalPlane(),
    m_ssimTotal(0.0),
    m_psnrTotalPlane(),
    m_psnrTotal(0.0),
    m_frames(0) {
    m_sFilterName = _T("ssim/psnr/vmaf");
}

NVEncFilterSsim::~NVEncFilterSsim() {
    close();
}

RGY_ERR NVEncFilterSsim::init(shared_ptr<NVEncFilterParam> pParam, shared_ptr<RGYLog> pPrintMes) {
    RGY_ERR sts = RGY_ERR_NONE;
    m_pPrintMes = pPrintMes;

    auto prm = std::dynamic_pointer_cast<NVEncFilterParamSsim>(pParam);
    if (!prm) {
        AddMessage(RGY_LOG_ERROR, _T("Invalid parameter type.\n"));
        return RGY_ERR_INVALID_PARAM;
    }

    m_vidctxlock = prm->vidctxlock;
    m_deviceId = prm->deviceId;
    m_crop.reset();
    if (pParam->frameOut.csp == RGY_CSP_NV12) {
        pParam->frameOut.csp = RGY_CSP_YV12;
    }
    if (pParam->frameIn.csp != pParam->frameOut.csp) {
        unique_ptr<NVEncFilterCspCrop> filterCrop(new NVEncFilterCspCrop());
        shared_ptr<NVEncFilterParamCrop> paramCrop(new NVEncFilterParamCrop());
        paramCrop->frameIn = pParam->frameIn;
        paramCrop->frameOut = pParam->frameOut;
        paramCrop->baseFps = pParam->baseFps;
        paramCrop->frameIn.deivce_mem = true;
        paramCrop->frameOut.deivce_mem = true;
        paramCrop->bOutOverwrite = false;
        NVEncCtxAutoLock(cxtlock(m_vidctxlock));
        sts = filterCrop->init(paramCrop, m_pPrintMes);
        if (sts != RGY_ERR_NONE) {
            return sts;
        }
        m_crop = std::move(filterCrop);
        AddMessage(RGY_LOG_DEBUG, _T("created %s.\n"), m_crop->GetInputMessage().c_str());
        pParam->frameOut = paramCrop->frameOut;
    }
    AddMessage(RGY_LOG_DEBUG, _T("ssim original format %s -> %s.\n"), RGY_CSP_NAMES[pParam->frameIn.csp], RGY_CSP_NAMES[pParam->frameOut.csp]);

    m_cropDToH.reset();
    if (prm->vmaf.enable) {
        unique_ptr<NVEncFilterCspCrop> filterCrop(new NVEncFilterCspCrop());
        shared_ptr<NVEncFilterParamCrop> paramCrop(new NVEncFilterParamCrop());
        paramCrop->frameIn = pParam->frameOut;
        paramCrop->frameOut = pParam->frameOut;
        paramCrop->baseFps = pParam->baseFps;
        paramCrop->frameIn.deivce_mem = true;
        paramCrop->frameOut.deivce_mem = false;
        paramCrop->bOutOverwrite = false;
        NVEncCtxAutoLock(cxtlock(m_vidctxlock));
        sts = filterCrop->init(paramCrop, m_pPrintMes);
        if (sts != RGY_ERR_NONE) {
            return sts;
        }
        m_cropDToH = std::move(filterCrop);
        AddMessage(RGY_LOG_DEBUG, _T("created %s.\n"), m_cropDToH->GetInputMessage().c_str());
    }

    {
        int elemSum = 0;
        for (size_t i = 0; i < m_ssimTotalPlane.size(); i++) {
            const auto plane = getPlane(&pParam->frameOut, (RGY_PLANE)i);
            elemSum += plane.width * plane.height;
        }
        for (size_t i = 0; i < m_ssimTotalPlane.size(); i++) {
            const auto plane = getPlane(&pParam->frameOut, (RGY_PLANE)i);
            m_planeCoef[i] = (double)(plane.width * plane.height) / elemSum;
            AddMessage(RGY_LOG_DEBUG, _T("Plane coef : %f\n"), m_planeCoef[i]);
        }
    }
    //SSIM
    for (size_t i = 0; i < m_ssimTotalPlane.size(); i++) {
        m_ssimTotalPlane[i] = 0.0;
    }
    m_ssimTotal = 0.0;
    //PSNR
    for (size_t i = 0; i < m_psnrTotalPlane.size(); i++) {
        m_psnrTotalPlane[i] = 0.0;
    }
    m_psnrTotal = 0.0;

    setFilterInfo(pParam->print() + _T("(") + RGY_CSP_NAMES[pParam->frameOut.csp] + _T(")"));
    m_pParam = pParam;
    return sts;
}

RGY_ERR NVEncFilterSsim::initDecode(const RGYBitstream *bitstream) {
    AddMessage(RGY_LOG_DEBUG, _T("initDecode() with bitstream size: %d.\n"), (int)bitstream->size());

    auto prm = std::dynamic_pointer_cast<NVEncFilterParamSsim>(m_pParam);
    if (!prm) {
        AddMessage(RGY_LOG_ERROR, _T("Invalid parameter type.\n"));
        return RGY_ERR_INVALID_PARAM;
    }
    int ret = 0;
    const auto avcodecID = getAVCodecId(prm->input.codec);
    const auto codec = avcodec_find_decoder(avcodecID);
    if (codec == nullptr) {
        AddMessage(RGY_LOG_ERROR, _T("failed to find decoder for codec %s.\n"), CodecToStr(prm->input.codec).c_str());
        return RGY_ERR_NULL_PTR;
    }
    auto codecCtx = std::unique_ptr<AVCodecContext, decltype(&avcodec_close)>(avcodec_alloc_context3(codec), &avcodec_close);
    if (0 > (ret = avcodec_open2(codecCtx.get(), codec, nullptr))) {
        AddMessage(RGY_LOG_ERROR, _T("failed to open codec %s: %s.\n"), char_to_tstring(avcodec_get_name(avcodecID)).c_str(), qsv_av_err2str(ret).c_str());
        return RGY_ERR_NULL_PTR;
    }
    AddMessage(RGY_LOG_DEBUG, _T("Opened decoder for codec %s\n"), char_to_tstring(avcodec_get_name(avcodecID)).c_str());

    const char *bsf_name = "extract_extradata";
    const auto bsf = av_bsf_get_by_name(bsf_name);
    if (bsf == nullptr) {
        AddMessage(RGY_LOG_ERROR, _T("failed to bsf %s.\n"), char_to_tstring(bsf_name).c_str());
        return RGY_ERR_NULL_PTR;
    }
    AVBSFContext *bsfctmp = nullptr;
    if (0 > (ret = av_bsf_alloc(bsf, &bsfctmp))) {
        AddMessage(RGY_LOG_ERROR, _T("failed to allocate memory for %s: %s.\n"), char_to_tstring(bsf_name).c_str(), qsv_av_err2str(ret).c_str());
        return RGY_ERR_NULL_PTR;
    }
    unique_ptr<AVBSFContext, RGYAVDeleter<AVBSFContext>> bsfc(bsfctmp, RGYAVDeleter<AVBSFContext>(av_bsf_free));
    bsfctmp = nullptr;

    unique_ptr<AVCodecParameters, RGYAVDeleter<AVCodecParameters>> codecpar(avcodec_parameters_alloc(), RGYAVDeleter<AVCodecParameters>(avcodec_parameters_free));
    if (0 > (ret = avcodec_parameters_from_context(codecpar.get(), codecCtx.get()))) {
        AddMessage(RGY_LOG_ERROR, _T("failed to get codec parameter for %s: %s.\n"), char_to_tstring(bsf_name).c_str(), qsv_av_err2str(ret).c_str());
        return RGY_ERR_UNKNOWN;
    }
    if (0 > (ret = avcodec_parameters_copy(bsfc->par_in, codecpar.get()))) {
        AddMessage(RGY_LOG_ERROR, _T("failed to copy parameter for %s: %s.\n"), char_to_tstring(bsf_name).c_str(), qsv_av_err2str(ret).c_str());
        return RGY_ERR_UNKNOWN;
    }
    if (0 > (ret = av_bsf_init(bsfc.get()))) {
        AddMessage(RGY_LOG_ERROR, _T("failed to init %s: %s.\n"), char_to_tstring(bsf_name).c_str(), qsv_av_err2str(ret).c_str());
        return RGY_ERR_UNKNOWN;
    }
    AddMessage(RGY_LOG_DEBUG, _T("Initialized bsf %s\n"), char_to_tstring(bsf_name).c_str());

    AVPacket pkt;
    av_new_packet(&pkt, (int)bitstream->size());
    memcpy(pkt.data, bitstream->data(), (int)bitstream->size());
    if (0 > (ret = av_bsf_send_packet(bsfc.get(), &pkt))) {
        AddMessage(RGY_LOG_ERROR, _T("failed to send packet to %s bitstream filter: %s.\n"),
            char_to_tstring(bsfc->filter->name).c_str(), qsv_av_err2str(ret).c_str());
        return RGY_ERR_UNKNOWN;
    }
    ret = av_bsf_receive_packet(bsfc.get(), &pkt);
    if (ret == AVERROR(EAGAIN)) {
        return RGY_ERR_NONE;
    } else if ((ret < 0 && ret != AVERROR_EOF) || pkt.size < 0) {
        AddMessage(RGY_LOG_ERROR, _T("failed to run %s bitstream filter: %s.\n"),
            char_to_tstring(bsfc->filter->name).c_str(), qsv_av_err2str(ret).c_str());
        return RGY_ERR_UNKNOWN;
    }
    int side_data_size = 0;
    auto side_data = av_packet_get_side_data(&pkt, AV_PKT_DATA_NEW_EXTRADATA, &side_data_size);
    if (side_data) {
        prm->input.codecExtra = malloc(side_data_size);
        prm->input.codecExtraSize = side_data_size;
        memcpy(prm->input.codecExtra, side_data, side_data_size);
        AddMessage(RGY_LOG_DEBUG, _T("Found extradata of codec %s: size %d\n"), char_to_tstring(avcodec_get_name(avcodecID)).c_str(), side_data_size);
    }
    av_packet_unref(&pkt);

    //比較用のスレッドの開始
    m_thread = std::thread(&NVEncFilterSsim::thread_func_ssim_psnr, this);
    AddMessage(RGY_LOG_DEBUG, _T("Started ssim/psnr calculation thread.\n"));

    //デコードの開始を待つ必要がある
    while (m_thread.joinable() && !m_decodeStarted) {
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }

    AddMessage(RGY_LOG_DEBUG, _T("initDecode(): fin.\n"));
    return (m_decodeStarted) ? RGY_ERR_NONE : RGY_ERR_UNKNOWN;
}

RGY_ERR NVEncFilterSsim::init_cuda_resources() {
    auto prm = std::dynamic_pointer_cast<NVEncFilterParamSsim>(m_pParam);
    if (!prm) {
        AddMessage(RGY_LOG_ERROR, _T("Invalid parameter type.\n"));
        return RGY_ERR_INVALID_PARAM;
    }

    //HWデコーダの出力フォーマットに合わせる
    VideoInfo vidInfo = prm->input;
    if (RGY_CSP_BIT_DEPTH[vidInfo.csp] > 8) {
        if (RGY_CSP_CHROMA_FORMAT[vidInfo.csp] == RGY_CHROMAFMT_YUV420) {
            vidInfo.csp = RGY_CSP_P010;
        } else if (RGY_CSP_CHROMA_FORMAT[vidInfo.csp] == RGY_CHROMAFMT_YUV444) {
            vidInfo.csp = RGY_CSP_YUV444_16;
        } else {
            AddMessage(RGY_LOG_ERROR, _T("Unexpected output format.\n"));
            return RGY_ERR_INVALID_COLOR_FORMAT;
        }
    }
    AddMessage(RGY_LOG_DEBUG, _T("cuvid output format %s.\n"), RGY_CSP_NAMES[vidInfo.csp]);

    m_decoder = std::make_unique<CuvidDecode>();
    auto result = m_decoder->InitDecode(m_vidctxlock, &vidInfo, nullptr, av_make_q(prm->streamtimebase), m_pPrintMes, NV_ENC_AVCUVID_NATIVE, false);
    if (result != CUDA_SUCCESS) {
        AddMessage(RGY_LOG_ERROR, _T("failed to init decoder.\n"));
        return RGY_ERR_INVALID_PARAM;
    }
    //SSIM用のスレッドで使用するリソースもすべてSSIM用のスレッド内で作成する
    {
        CCtxAutoLock ctxLock(m_vidctxlock);
        if (prm->ssim) {
            for (size_t i = 0; i < m_streamCalcSsim.size(); i++) {
                m_streamCalcSsim[i] = std::unique_ptr<cudaStream_t, cudastream_deleter>(new cudaStream_t(), cudastream_deleter());
                auto cudaerr = cudaStreamCreateWithFlags(m_streamCalcSsim[i].get(), cudaStreamDefault);
                if (cudaerr != cudaSuccess) {
                    AddMessage(RGY_LOG_ERROR, _T("failed to cudaStreamCreateWithFlags: %s.\n"), char_to_tstring(cudaGetErrorName(cudaerr)).c_str());
                    return RGY_ERR_CUDA;
                }
                AddMessage(RGY_LOG_DEBUG, _T("cudaStreamCreateWithFlags for m_streamCalcSsim[%d]: Success.\n"), i);
            }
        }
        if (prm->psnr) {
            for (size_t i = 0; i < m_streamCalcPsnr.size(); i++) {
                m_streamCalcPsnr[i] = std::unique_ptr<cudaStream_t, cudastream_deleter>(new cudaStream_t(), cudastream_deleter());
                auto cudaerr = cudaStreamCreateWithFlags(m_streamCalcPsnr[i].get(), cudaStreamDefault);
                if (cudaerr != cudaSuccess) {
                    AddMessage(RGY_LOG_ERROR, _T("failed to cudaStreamCreateWithFlags: %s.\n"), char_to_tstring(cudaGetErrorName(cudaerr)).c_str());
                    return RGY_ERR_CUDA;
                }
                AddMessage(RGY_LOG_DEBUG, _T("cudaStreamCreateWithFlags for m_streamCalcPsnr[%d]: Success.\n"), i);
            }
        }
        m_streamCrop = std::unique_ptr<cudaStream_t, cudastream_deleter>(new cudaStream_t(), cudastream_deleter());
        auto cudaerr = cudaStreamCreateWithFlags(m_streamCrop.get(), cudaStreamDefault);
        if (cudaerr != cudaSuccess) {
            AddMessage(RGY_LOG_ERROR, _T("failed to cudaStreamCreateWithFlags: %s.\n"), char_to_tstring(cudaGetErrorName(cudaerr)).c_str());
            return RGY_ERR_CUDA;
        }
        AddMessage(RGY_LOG_DEBUG, _T("cudaStreamCreateWithFlags for m_streamCrop: Success.\n"));

        m_cropEvent = std::unique_ptr<cudaEvent_t, cudaevent_deleter>(new cudaEvent_t(), cudaevent_deleter());
        cudaerr = cudaEventCreate(m_cropEvent.get());
        if (cudaerr != cudaSuccess) {
            AddMessage(RGY_LOG_ERROR, _T("failed to cudaEventCreate: %s.\n"), char_to_tstring(cudaGetErrorName(cudaerr)).c_str());
            return RGY_ERR_CUDA;
        }
        AddMessage(RGY_LOG_DEBUG, _T("cudaEventCreate for m_cropEvent: Success.\n"));

        if (prm->vmaf.enable) {
            //VMAF用のスレッドで使用するリソースもすべてスレッド内で作成する
            const auto frameInfo = m_cropDToH->GetFilterParam()->frameOut;
            for (auto &frame : m_frameHostOrg) {
                frame = std::make_unique<CUFrameBuf>(frameInfo.width, frameInfo.height, frameInfo.csp);
                cudaerr = frame->allocHost();
                if (cudaerr != cudaSuccess) {
                    AddMessage(RGY_LOG_ERROR, _T("failed to allocate host frame buffer: %s.\n"), char_to_tstring(cudaGetErrorName(cudaerr)).c_str());
                    return RGY_ERR_CUDA;
                }
            }
            for (auto &frame : m_frameHostEnc) {
                frame = std::make_unique<CUFrameBuf>(frameInfo.width, frameInfo.height, frameInfo.csp);
                cudaerr = frame->allocHost();
                if (cudaerr != cudaSuccess) {
                    AddMessage(RGY_LOG_ERROR, _T("failed to allocate host frame buffer: %s.\n"), char_to_tstring(cudaGetErrorName(cudaerr)).c_str());
                    return RGY_ERR_CUDA;
                }
            }

            m_vmaf.thread = std::thread(&NVEncFilterSsim::thread_func_vmaf, this);
            AddMessage(RGY_LOG_DEBUG, _T("Started vmaf calculation thread.\n"));
        }
    }
    return RGY_ERR_NONE;
}

void NVEncFilterSsim::close_cuda_resources() {
    if (m_vidctxlock) {
        CCtxAutoLock ctxLock(m_vidctxlock);
        m_streamCrop.reset();
        m_cropEvent.reset();
        for (auto& st : m_streamCalcSsim) {
            st.reset();
        }
        for (auto &st : m_streamCalcPsnr) {
            st.reset();
        }
        for (auto &buf : m_tmpSsim) {
            buf.clear();
        }
        for (auto &buf : m_tmpPsnr) {
            buf.clear();
        }
        m_decFrameCopy.reset();
        for (auto& frame : m_frameHostOrg) {
            frame.reset();
        }
        for (auto &frame : m_frameHostEnc) {
            frame.reset();
        }
        m_input.clear();
        m_unused.clear();
        AddMessage(RGY_LOG_DEBUG, _T("Freed CUDA resources.\n"));
    }
    if (m_vidctxlock) {
        m_decoder.reset();
        AddMessage(RGY_LOG_DEBUG, _T("Closed Decoder.\n"));
    }
    m_vidctxlock = nullptr;
}

RGY_ERR NVEncFilterSsim::addBitstream(const RGYBitstream *bitstream) {
    if (m_decoder->GetError()) {
        return RGY_ERR_UNKNOWN;
    }

    auto prm = std::dynamic_pointer_cast<NVEncFilterParamSsim>(m_pParam);
    if (!prm) {
        AddMessage(RGY_LOG_ERROR, _T("Invalid parameter type.\n"));
        return RGY_ERR_INVALID_PARAM;
    }

    CUresult curesult = CUDA_SUCCESS;
    if (bitstream != nullptr) {
        if (CUDA_SUCCESS != (curesult = m_decoder->DecodePacket(bitstream->bufptr() + bitstream->offset(), bitstream->size(), bitstream->pts(), av_make_q(prm->streamtimebase)))) {
            AddMessage(RGY_LOG_ERROR, _T("Error in DecodePacket: %d (%s).\n"), curesult, char_to_tstring(_cudaGetErrorEnum(curesult)).c_str());
            return RGY_ERR_UNKNOWN;
        }
    } else {
        if (CUDA_SUCCESS != (curesult = m_decoder->DecodePacket(nullptr, 0, AV_NOPTS_VALUE, av_make_q(prm->streamtimebase)))) {
            AddMessage(RGY_LOG_ERROR, _T("Error in DecodePacketFin: %d (%s).\n"), curesult, char_to_tstring(_cudaGetErrorEnum(curesult)).c_str());
            return RGY_ERR_UNKNOWN;
        }
    }
    return RGY_ERR_NONE;
}

RGY_ERR NVEncFilterSsim::run_filter(const FrameInfo *pInputFrame, FrameInfo **ppOutputFrames, int *pOutputFrameNum, cudaStream_t stream) {
    UNREFERENCED_PARAMETER(ppOutputFrames);
    UNREFERENCED_PARAMETER(pOutputFrameNum);
    RGY_ERR sts = RGY_ERR_NONE;

    std::lock_guard<std::mutex> lock(m_mtx); //ロックを忘れないこと
    if (m_unused.empty()) {
        //待機中のフレームバッファがなければ新たに作成する
        auto frameBuf = std::make_unique<CUFrameBuf>();
        copyFrameProp(&frameBuf->frame, (m_crop) ? &m_crop->GetFilterParam()->frameOut : pInputFrame);
        frameBuf->frame.deivce_mem = true;
        auto curesult = frameBuf->alloc();
        if (curesult != cudaSuccess) {
            AddMessage(RGY_LOG_ERROR, _T("failed to allocate memory.\n"));
            return RGY_ERR_MEMORY_ALLOC;
        }
        m_unused.push_back(std::move(frameBuf));
    }
    auto& copyFrame = m_unused.front();
    if (m_crop) {
        int cropFilterOutputNum = 0;
        FrameInfo *outInfo[1] = { &copyFrame->frame };
        FrameInfo cropInput = *pInputFrame;
        auto sts_filter = m_crop->filter(&cropInput, (FrameInfo **)&outInfo, &cropFilterOutputNum, stream);
        if (outInfo[0] == nullptr || cropFilterOutputNum != 1) {
            AddMessage(RGY_LOG_ERROR, _T("Unknown behavior \"%s\".\n"), m_crop->name().c_str());
            return sts_filter;
        }
        if (sts_filter != RGY_ERR_NONE || cropFilterOutputNum != 1) {
            AddMessage(RGY_LOG_ERROR, _T("Error while running filter \"%s\".\n"), m_crop->name().c_str());
            return sts_filter;
        }
    } else {
        auto cuerr = copyFrameAsync(&copyFrame->frame, pInputFrame, stream);
        if (cuerr != cudaSuccess) {
            AddMessage(RGY_LOG_ERROR, _T("Failed to copy frame.\n"));
            return RGY_ERR_CUDA;
        }
    }
    cudaEventRecord(copyFrame->event, stream);

    //フレームをm_unusedからm_inputに移す
    m_input.push_back(std::move(copyFrame));
    m_unused.pop_front();
    return sts;
}

void NVEncFilterSsim::showResult() {
    auto prm = std::dynamic_pointer_cast<NVEncFilterParamSsim>(m_pParam);
    if (!prm) {
        AddMessage(RGY_LOG_ERROR, _T("Invalid parameter type.\n"));
        return;
    }
    if (m_thread.joinable()) {
        AddMessage(RGY_LOG_DEBUG, _T("Waiting for ssim/psnr/vmaf calculation thread to finish.\n"));
        m_thread.join();
    }
    if (prm->ssim) {
        auto str = strsprintf(_T("\nSSIM YUV:"));
        for (int i = 0; i < RGY_CSP_PLANES[m_pParam->frameOut.csp]; i++) {
            str += strsprintf(_T(" %f (%f),"), m_ssimTotalPlane[i] / m_frames, ssim_db(m_ssimTotalPlane[i], (double)m_frames));
        }
        str += strsprintf(_T(" All: %f (%f), (Frames: %d)\n"), m_ssimTotal / m_frames, ssim_db(m_ssimTotal, (double)m_frames), m_frames);
        AddMessage(RGY_LOG_INFO, _T("%s\n"), str.c_str());
    }
    if (prm->psnr) {
        auto str = strsprintf(_T("\nPSNR YUV:"));
        for (int i = 0; i < RGY_CSP_PLANES[m_pParam->frameOut.csp]; i++) {
            str += strsprintf(_T(" %f,"), get_psnr(m_psnrTotalPlane[i], m_frames, (1 << RGY_CSP_BIT_DEPTH[prm->frameOut.csp]) - 1));
        }
        str += strsprintf(_T(" Avg: %f, (Frames: %d)\n"), get_psnr(m_psnrTotal, m_frames, (1 << RGY_CSP_BIT_DEPTH[prm->frameOut.csp]) - 1), m_frames);
        AddMessage(RGY_LOG_INFO, _T("%s\n"), str.c_str());
    }
    if (prm->vmaf.enable) {
        if (m_vmaf.error == 0) {
            AddMessage(RGY_LOG_INFO, _T("VMAF Score %.6f\n"), m_vmaf.score);
        }
    }
}

NVEncFilterVMAFData::NVEncFilterVMAFData() : heProcFin(), abort(false), procIndex(0), error(0), score(0.0), thread() {
    for (auto &event : heProcFin) {
        event = CreateEvent(nullptr, false, false, nullptr);
    }
};
NVEncFilterVMAFData::~NVEncFilterVMAFData() {
    for (auto &event : heProcFin) {
        if (event) {
            CloseEvent(event);
            event = nullptr;
        }
    }
    thread_fin();
}
void NVEncFilterVMAFData::thread_fin() {
    abort = true;
    if (thread.joinable()) {
        thread.join();
    }
}

template<typename pixType>
void read_frame_vmaf(float *dst, int stride_byte, const FrameInfo *srcFrame) {
    const float factor = 1.f / (1 << (RGY_CSP_BIT_DEPTH[srcFrame->csp] - 8));
    for (int y = 0; y < srcFrame->height; y++) {
        float *ptrDstLine = (float *)((char *)dst + stride_byte * y);
        const pixType *ptrSrcLine = (const pixType *)((char *)srcFrame->ptr + srcFrame->pitch * y);
        for (int x = 0; x < srcFrame->width; x++) {
            ptrDstLine[x] = ptrSrcLine[x] * factor;
        }
    }
}

int read_frames_vmaf(float *ref_data, float *main_data, float *temp_data, int stride_byte, void *user_data) {
    UNREFERENCED_PARAMETER(temp_data);
    NVEncFilterSsim *filter = (NVEncFilterSsim *)user_data;
    if (filter->vmaf().abort
        && filter->vmaf().procIndex >= filter->frameHostSendIndex()) {
        return 2;
    }
    auto &framesEnc = filter->frameHostEnc()[filter->vmaf().procIndex % filter->frameHostEnc().size()];
    auto &framesOrg = filter->frameHostOrg()[filter->vmaf().procIndex % filter->frameHostEnc().size()];
    if (   cudaEventSynchronize(framesOrg->event) != cudaSuccess
        || cudaEventSynchronize(framesEnc->event) != cudaSuccess) {
        return 2;
    }
    if (RGY_CSP_BIT_DEPTH[framesEnc->frame.csp] > 8) {
        read_frame_vmaf<uint16_t>(main_data, stride_byte, &framesEnc->frame);
        read_frame_vmaf<uint16_t>(ref_data,  stride_byte, &framesOrg->frame);
    } else {
        read_frame_vmaf<uint8_t>(main_data, stride_byte, &framesEnc->frame);
        read_frame_vmaf<uint8_t>(ref_data,  stride_byte, &framesOrg->frame);
    }

    SetEvent(filter->vmaf().heProcFin[(filter->vmaf().procIndex++) % filter->vmaf().heProcFin.size()]);
    return 0;
};

RGY_ERR NVEncFilterSsim::thread_func_vmaf() {
    const auto frameInfo = m_frameHostEnc[0]->frame;
    std::unique_ptr<char, decltype(&free)> pix_fmt_name(_strdup(av_pix_fmt_desc_get(csp_rgy_to_avpixfmt(frameInfo.csp))->name), free);

    auto prm = std::dynamic_pointer_cast<NVEncFilterParamSsim>(m_pParam);
    if (!prm) {
        AddMessage(RGY_LOG_ERROR, _T("Invalid parameter type.\n"));
        return RGY_ERR_INVALID_PARAM;
    }
    std::unique_ptr<char, decltype(&free)> model_path(_strdup(prm->vmaf.model_path.c_str()), free);

    for (auto &handle : m_vmaf.heProcFin) {
        SetEvent(handle);
    }

    m_vmaf.error = compute_vmaf(&m_vmaf.score, pix_fmt_name.get(), frameInfo.width, frameInfo.height,
        read_frames_vmaf, this,
        model_path.get(), nullptr /*log_path*/, nullptr /*log_fmt*/, 0, 0,
        0 /*enable_transform*/,
        0 /*phone_model*/,
        false /*psnr*/, false /*ssim*/, false /*ms_ssim*/,
        nullptr /*pool*/,
        prm->vmaf.threads,
        1 /*subsample*/, 0 /*enable_conf_interval*/);
    AddMessage(RGY_LOG_DEBUG, _T("Finishing vmaf calculation thread: %d.\n"), m_vmaf.error);
    return (m_vmaf.error == 0) ? RGY_ERR_NONE : RGY_ERR_UNKNOWN;
}

RGY_ERR NVEncFilterSsim::thread_func_ssim_psnr() {
    auto sts = init_cuda_resources();
    if (sts != RGY_ERR_NONE) {
        return sts;
    }
    m_decodeStarted = true;
    auto ret = compare_frames(true);
    AddMessage(RGY_LOG_DEBUG, _T("Finishing ssim/psnr calculation thread: %s.\n"), get_err_mes(ret));
    m_vmaf.thread_fin();
    close_cuda_resources();
    return ret;
}

RGY_ERR NVEncFilterSsim::compare_frames(bool flush) {
    auto prm = std::dynamic_pointer_cast<NVEncFilterParamSsim>(m_pParam);
    if (!prm) {
        AddMessage(RGY_LOG_ERROR, _T("Invalid parameter type.\n"));
        return RGY_ERR_INVALID_PARAM;
    }
    while (!m_abort) {
        if (m_decoder->GetError()) {
            AddMessage(RGY_LOG_ERROR, _T("Error in decoder!\n"));
            return RGY_ERR_UNKNOWN;
        }
        if (m_decoder->frameQueue()->isEndOfDecode() && m_decoder->frameQueue()->isEmpty()) {
            AddMessage(RGY_LOG_DEBUG, _T("Finished decoding.\n"));
            return RGY_ERR_NONE;
        }
        CUVIDPARSERDISPINFO dispInfo = { 0 };
        if (!m_decoder->frameQueue()->dequeue(&dispInfo)) {
            if (!flush) {
                return RGY_ERR_NONE;
            }
            m_decoder->frameQueue()->waitForQueueUpdate();
            continue;
        }

        auto disp = std::shared_ptr<CUVIDPARSERDISPINFO>(new CUVIDPARSERDISPINFO(dispInfo), [&](CUVIDPARSERDISPINFO *ptr) {
            m_decoder->frameQueue()->releaseFrame(ptr);
            delete ptr;
            });

        CUresult curesult = CUDA_SUCCESS;
        CUVIDPROCPARAMS vppinfo = { 0 };
        vppinfo.top_field_first = dispInfo.top_field_first;
        vppinfo.progressive_frame = dispInfo.progressive_frame;
        vppinfo.unpaired_field = 0;
        CUdeviceptr dMappedFrame = 0;
        uint32_t pitch = 0;
        NVEncCtxAutoLock(ctxlock(m_vidctxlock));
        if (CUDA_SUCCESS != (curesult = cuvidMapVideoFrame(m_decoder->GetDecoder(), dispInfo.picture_index, &dMappedFrame, &pitch, &vppinfo))) {
            AddMessage(RGY_LOG_ERROR, _T("Error cuvidMapVideoFrame: %d (%s).\n"), curesult, char_to_tstring(_cudaGetErrorEnum(curesult)).c_str());
            return RGY_ERR_UNKNOWN;
        }
        auto frameInfo = m_decoder->GetDecFrameInfo();
        frameInfo.pitch = pitch;
        frameInfo.ptr = (uint8_t *)dMappedFrame;
        auto deviceFrame = shared_ptr<void>(frameInfo.ptr, [&](void *ptr) {
            cuvidUnmapVideoFrame(m_decoder->GetDecoder(), (CUdeviceptr)ptr);
            });

        FrameInfo targetFrame = frameInfo;
        if (m_crop) {
            if (!m_decFrameCopy) {
                m_decFrameCopy = std::make_unique<CUFrameBuf>();
                copyFrameProp(&m_decFrameCopy->frame, &m_crop->GetFilterParam()->frameOut);
                m_decFrameCopy->frame.deivce_mem = true;
                auto cuerr = m_decFrameCopy->alloc();
                if (cuerr != cudaSuccess) {
                    AddMessage(RGY_LOG_ERROR, _T("failed to allocate memory.\n"));
                    return RGY_ERR_MEMORY_ALLOC;
                }
            }
            targetFrame = m_decFrameCopy->frame;

            int cropFilterOutputNum = 0;
            FrameInfo *outInfo[1] = { &targetFrame };
            auto sts_filter = m_crop->filter(&frameInfo, (FrameInfo **)&outInfo, &cropFilterOutputNum, *m_streamCrop.get());
            if (outInfo[0] == nullptr || cropFilterOutputNum != 1) {
                AddMessage(RGY_LOG_ERROR, _T("Unknown behavior \"%s\".\n"), m_crop->name().c_str());
                return sts_filter;
            }
            if (sts_filter != RGY_ERR_NONE || cropFilterOutputNum != 1) {
                AddMessage(RGY_LOG_ERROR, _T("Error while running filter \"%s\".\n"), m_crop->name().c_str());
                return sts_filter;
            }
            //cropが終わってから比較が行われるように設定
            cudaEventRecord(*m_cropEvent.get(), *m_streamCrop.get());
            for (int i = 0; i < RGY_CSP_PLANES[targetFrame.csp]; i++) {
                if (prm->ssim) {
                    cudaStreamWaitEvent(*m_streamCalcSsim[i].get(), *m_cropEvent.get(), 0);
                }
                if (prm->psnr) {
                    cudaStreamWaitEvent(*m_streamCalcPsnr[i].get(), *m_cropEvent.get(), 0);
                }
            }
        }
        if (m_cropDToH) {
            WaitForSingleObject(m_vmaf.heProcFin[m_frameHostSendIndex % m_vmaf.heProcFin.size()], INFINITE);
            {
                int cropFilterOutputNum = 0;
                auto &frameHostOrg = m_frameHostOrg[m_frameHostSendIndex % m_frameHostOrg.size()];
                FrameInfo *outInfoOrg[1] = { &frameHostOrg->frame };
                auto sts_filter = m_cropDToH->filter(&m_input.front()->frame, (FrameInfo **)&outInfoOrg, &cropFilterOutputNum, *m_streamCrop.get());
                if (outInfoOrg[0] == nullptr || cropFilterOutputNum != 1) {
                    AddMessage(RGY_LOG_ERROR, _T("Unknown behavior \"%s\".\n"), m_cropDToH->name().c_str());
                    return sts_filter;
                }
                cudaDeviceSynchronize();
                auto cudaerr = cudaGetLastError();
                if (cudaerr != cudaSuccess) {
                    AddMessage(RGY_LOG_ERROR, _T("error at m_cropDToH(Org)->filter: %s.\n"),
                        char_to_tstring(cudaGetErrorString(cudaerr)).c_str());
                    return RGY_ERR_INVALID_CALL;
                }
                cudaEventRecord(frameHostOrg->event, *m_streamCrop.get());
                cudaerr = cudaGetLastError();
                if (cudaerr != cudaSuccess) {
                    AddMessage(RGY_LOG_ERROR, _T("error at cudaEventRecord(Org)->filter: %s.\n"),
                        char_to_tstring(cudaGetErrorString(cudaerr)).c_str());
                    return RGY_ERR_INVALID_CALL;
                }
            }

            {
                int cropFilterOutputNum = 0;
                auto &frameHostEnc = m_frameHostEnc[m_frameHostSendIndex % m_frameHostEnc.size()];
                FrameInfo *outInfoEnc[1] = { &frameHostEnc->frame };
                auto sts_filter = m_cropDToH->filter(&targetFrame, (FrameInfo **)&outInfoEnc, &cropFilterOutputNum, *m_streamCrop.get());
                if (outInfoEnc[0] == nullptr || cropFilterOutputNum != 1) {
                    AddMessage(RGY_LOG_ERROR, _T("Unknown behavior \"%s\".\n"), m_cropDToH->name().c_str());
                    return sts_filter;
                }
                cudaDeviceSynchronize();
                auto cudaerr = cudaGetLastError();
                if (cudaerr != cudaSuccess) {
                    AddMessage(RGY_LOG_ERROR, _T("error at m_cropDToH(Enc)->filter: %s.\n"),
                        char_to_tstring(cudaGetErrorString(cudaerr)).c_str());
                    return RGY_ERR_INVALID_CALL;
                }
                cudaEventRecord(frameHostEnc->event, *m_streamCrop.get());
                cudaerr = cudaGetLastError();
                if (cudaerr != cudaSuccess) {
                    AddMessage(RGY_LOG_ERROR, _T("error at cudaEventRecord(Enc)->filter: %s.\n"),
                        char_to_tstring(cudaGetErrorString(cudaerr)).c_str());
                    return RGY_ERR_INVALID_CALL;
                }
            }
            m_frameHostSendIndex++;
        }

        //比較用のキューの先頭に積まれているものから順次比較していく
        if (m_input.empty()) {
            AddMessage(RGY_LOG_ERROR, _T("Original frame #%d to be compared is missing.\n"), m_frames);
            return RGY_ERR_UNKNOWN;
        }
        FrameInfo original;
        {
            std::lock_guard<std::mutex> lock(m_mtx); //ロックを忘れないこと
            auto &originalFrame = m_input.front();
            //オリジナルのフレームに対するcrop操作が終わっているか確認する (基本的には終わっているはず)
            if (m_crop) {
                cudaEventSynchronize(originalFrame->event);
            }
            original = originalFrame->frame;
        }
        auto sts_filter = calc_ssim_psnr(&original, &targetFrame);
        if (sts_filter != RGY_ERR_NONE) {
            return sts_filter;
        }

        //フレームをm_inputからm_unusedに移す
        std::lock_guard<std::mutex> lock(m_mtx); //ロックを忘れないこと
        m_unused.push_back(std::move(m_input.front()));
        m_input.pop_front();
        m_frames++;
    }
    return RGY_ERR_NONE;
}

void NVEncFilterSsim::close() {
    if (m_thread.joinable()) {
        AddMessage(RGY_LOG_DEBUG, _T("Forcing ssim/psnr calculation thread to finish.\n"));
        m_abort = true;
        m_thread.join();
    }
    close_cuda_resources();
    AddMessage(RGY_LOG_DEBUG, _T("closed ssim/psnr filter.\n"));
}

#endif //#if ENABLE_SSIM
