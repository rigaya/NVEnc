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

#pragma once
#ifndef __RGY_INPUT_AVS_H__
#define __RGY_INPUT_AVS_H__

#include "rgy_version.h"
#if ENABLE_AVISYNTH_READER
#pragma warning(push)
#pragma warning(disable:4244)
#pragma warning(disable:4456)
#if defined(_WIN32) || defined(_WIN64)
#include "avisynth_c.h" //Avisynth+のヘッダを想定
#define IS_AVXSYNTH 0
#else
#include "avxsynth_c.h"
#define IS_AVXSYNTH 1
#endif
#include "rgy_osdep.h"
#include "rgy_input.h"
#pragma warning(pop)

#define AVS_FUNCTYPE(x) typedef decltype(avs_ ## x)* func_avs_ ## x;

AVS_FUNCTYPE(invoke);
AVS_FUNCTYPE(take_clip);
AVS_FUNCTYPE(release_value);
AVS_FUNCTYPE(create_script_environment);
AVS_FUNCTYPE(get_video_info);
AVS_FUNCTYPE(get_audio);
AVS_FUNCTYPE(get_frame);
AVS_FUNCTYPE(release_video_frame);
AVS_FUNCTYPE(release_clip);
AVS_FUNCTYPE(delete_script_environment);
AVS_FUNCTYPE(get_version);
AVS_FUNCTYPE(get_pitch_p);
AVS_FUNCTYPE(get_read_ptr_p);
AVS_FUNCTYPE(clip_get_error);
#if !IS_AVXSYNTH
AVS_FUNCTYPE(is_420);
AVS_FUNCTYPE(is_422);
AVS_FUNCTYPE(is_444);
#endif

#undef AVS_FUNCTYPE

#define AVS_FUNCDECL(x) func_avs_ ## x f_ ## x;

struct avs_dll_t {
    HMODULE h_avisynth;
    AVS_FUNCDECL(invoke)
    AVS_FUNCDECL(take_clip)
    AVS_FUNCDECL(release_value)
    AVS_FUNCDECL(create_script_environment)
    AVS_FUNCDECL(get_video_info)
    AVS_FUNCDECL(get_audio)
    AVS_FUNCDECL(get_frame)
    AVS_FUNCDECL(release_video_frame)
    AVS_FUNCDECL(release_clip)
    AVS_FUNCDECL(delete_script_environment)
    AVS_FUNCDECL(get_version)
    AVS_FUNCDECL(get_pitch_p)
    AVS_FUNCDECL(get_read_ptr_p)
    AVS_FUNCDECL(clip_get_error);
#if !IS_AVXSYNTH
    AVS_FUNCDECL(is_420)
    AVS_FUNCDECL(is_422)
    AVS_FUNCDECL(is_444)
#endif
};

#undef AVS_FUNCDECL


class RGYInputAvsPrm : public RGYInputPrm {
public:
    bool readAudio;
    tstring avsdll;
    RGYInputAvsPrm(RGYInputPrm base);

    virtual ~RGYInputAvsPrm() {};
};

class RGYInputAvs : public RGYInput {
public:
    RGYInputAvs();
    virtual ~RGYInputAvs();

    virtual RGY_ERR LoadNextFrame(RGYFrame *pSurface) override;
    virtual void Close() override;

#if ENABLE_AVSW_READER
    virtual int GetAudioTrackCount() override { return (int)m_audio.size(); };

    //音声・字幕パケットの配列を取得する
    virtual vector<AVPacket> GetStreamDataPackets(int inputFrame) override;

    //音声・字幕のコーデックコンテキストを取得する
    virtual vector<AVDemuxStream> GetInputStreamInfo() override { return m_audio; };
#endif // #if ENABLE_AVSW_READER

protected:
    virtual RGY_ERR Init(const TCHAR *strFileName, VideoInfo *pInputInfo, const RGYInputPrm *prm) override;
    RGY_ERR load_avisynth(const tstring& avsdll);
    void release_avisynth();

    AVS_ScriptEnvironment *m_sAVSenv;
    AVS_Clip *m_sAVSclip;
    const AVS_VideoInfo *m_sAVSinfo;

    avs_dll_t m_sAvisynth;

#if ENABLE_AVSW_READER
    RGY_ERR InitAudio();

    vector<AVDemuxStream> m_audio;
    unique_ptr<AVFormatContext, decltype(&avformat_free_context)> m_format;
    int64_t m_audioCurrentSample;
#endif //#if ENABLE_AVSW_READER
};

#endif //ENABLE_AVISYNTH_READER

#endif //__RGY_INPUT_AVS_H__
