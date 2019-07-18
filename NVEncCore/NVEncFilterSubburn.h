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

#pragma once

#include "NVEncFilter.h"
#include "NVEncParam.h"
#include "rgy_avutil.h"
#include "rgy_queue.h"
#include "rgy_input_avcodec.h"

#if ENABLE_AVSW_READER

#include "ass/ass.h"

struct subtitle_deleter {
    void operator()(AVSubtitle *subtitle) const {
        avsubtitle_free(subtitle);
        delete subtitle;
    }
};

struct SubImageData {
    unique_ptr<CUFrameBuf> image;
    unique_ptr<CUFrameBuf> imageTemp;
    unique_ptr<void, decltype(&cudaFreeHost)> imageCPU;
    int x, y;

    SubImageData(unique_ptr<CUFrameBuf> img, unique_ptr<CUFrameBuf> imgTemp, unique_ptr<void, decltype(&cudaFreeHost)> imgCPU, int posX, int posY) :
        image(std::move(img)), imageTemp(std::move(imgTemp)), imageCPU(std::move(imgCPU)), x(posX), y(posY){ }
};

class NVEncFilterParamSubburn : public NVEncFilterParam {
public:
    VppSubburn      subburn;
    AVRational      videoTimebase;
    const AVStream *videoInputStream;
    int64_t         videoInputFirstKeyPts;
    VideoInfo       videoInfo;
    AVDemuxStream   streamIn;
    sInputCrop      crop;

    NVEncFilterParamSubburn() : subburn(), videoTimebase(), videoInputStream(nullptr), videoInputFirstKeyPts(0), videoInfo(), streamIn(), crop() {};
    virtual ~NVEncFilterParamSubburn() {};
    virtual tstring print() const override;
};

class NVEncFilterSubburn : public NVEncFilter {
public:
    NVEncFilterSubburn();
    virtual ~NVEncFilterSubburn();
    virtual RGY_ERR init(shared_ptr<NVEncFilterParam> prm, shared_ptr<RGYLog> pPrintMes) override;
    virtual RGY_ERR addStreamPacket(AVPacket *pkt) override;
    virtual int targetTrackIdx() override;
protected:
    virtual RGY_ERR run_filter(const FrameInfo *pInputFrame, FrameInfo **ppOutputFrames, int *pOutputFrameNum) override;
    virtual void close() override;
    virtual RGY_ERR checkParam(const std::shared_ptr<NVEncFilterParamSubburn> prm);
    virtual RGY_ERR initAVCodec(const std::shared_ptr<NVEncFilterParamSubburn> prm);
    virtual RGY_ERR InitLibAss(const std::shared_ptr<NVEncFilterParamSubburn> prm);
    void SetExtraData(AVCodecContext *codecCtx, const uint8_t *data, uint32_t size);
    RGY_ERR readSubFile();
    SubImageData textRectToImage(const ASS_Image *image, cudaStream_t stream);
    SubImageData bitmapRectToImage(const AVSubtitleRect *rect, const sInputCrop &crop, cudaStream_t stream);
    RGY_ERR procFrameText(FrameInfo *pOutputFrame, int64_t frameTimeMs, cudaStream_t stream);
    RGY_ERR procFrameBitmap(FrameInfo *pOutputFrame, const sInputCrop& crop, cudaStream_t stream);
    RGY_ERR procFrame(FrameInfo *pOutputFrame, cudaStream_t stream);

    int m_subType; //字幕の種類
    unique_ptr<AVFormatContext, RGYAVDeleter<AVFormatContext>> m_formatCtx;     //ファイル読み込みの際に使用する(トラックを受け取る場合はnullptr)
    int m_subtitleStreamIndex; //ファイル読み込みの際に使用する(トラックを受け取る場合は-1)
    const AVCodec *m_outCodecDecode; //変換する元のコーデック
    unique_ptr<AVCodecContext, decltype(&avcodec_close)> m_outCodecDecodeCtx;     //変換する元のCodecContext

    unique_ptr<AVSubtitle, subtitle_deleter> m_subData;
    vector<SubImageData> m_subImages;

    unique_ptr<ASS_Library, decltype(&ass_library_done)> m_assLibrary; //libassのコンテキスト
    unique_ptr<ASS_Renderer, decltype(&ass_renderer_done)> m_assRenderer; //libassのレンダラ
    unique_ptr<ASS_Track, decltype(&ass_free_track)> m_assTrack; //libassのトラック

    unique_ptr<NVEncFilterResize> m_resize;

    RGYQueueSPSP<AVPacket> m_queueSubPackets; //入力から得られた字幕パケット
};

#endif //#if ENABLE_AVSW_READER
