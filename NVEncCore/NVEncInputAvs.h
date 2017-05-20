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

#include <stdio.h>
#include <tchar.h>
#include <string>
#include "NVEncUtil.h"
#include "rgy_status.h"
#include "rgy_version.h"
#include "NVEncInput.h"

#if ENABLE_AVISYNTH_READER
#pragma warning(push)
#pragma warning(disable:4244)
#pragma warning(disable:4456)
#include "avisynth_c.h" //Avisynth ver 2.5.8 (2.6.0の機能等は不要)
#pragma warning(pop)

typedef AVS_Value (__stdcall *func_avs_invoke)(AVS_ScriptEnvironment *scriptEnv, const char *name, AVS_Value args, const char** arg_names);
typedef AVS_Clip * (__stdcall *func_avs_take_clip)(AVS_Value value, AVS_ScriptEnvironment *scriptEnv);
typedef void  (__stdcall *func_avs_release_value)(AVS_Value value);
typedef AVS_ScriptEnvironment * (__stdcall *func_avs_create_script_environment)(int version);
typedef const AVS_VideoInfo * (__stdcall *func_avs_get_video_info)(AVS_Clip *clip);
typedef AVS_VideoFrame * (__stdcall *func_avs_get_frame)(AVS_Clip *clip, int n);
typedef void (__stdcall *func_avs_release_video_frame)(AVS_VideoFrame * frame);
typedef void (__stdcall *func_avs_release_clip)(AVS_Clip *clip);
typedef void (__stdcall *func_avs_delete_script_environment) (AVS_ScriptEnvironment *scriptEnv);
typedef float (__stdcall *func_avs_get_version)(void);

typedef struct {
    HMODULE h_avisynth;
    func_avs_invoke invoke;
    func_avs_take_clip take_clip;
    func_avs_release_value release_value;
    func_avs_create_script_environment create_script_environment;
    func_avs_get_video_info get_video_info;
    func_avs_get_frame get_frame;
    func_avs_release_video_frame release_video_frame;
    func_avs_release_clip release_clip;
    func_avs_delete_script_environment delete_script_environment;
    func_avs_get_version get_version;
} avs_dll_t;

class NVEncInputAvs : public NVEncBasicInput {
public:
    NVEncInputAvs();
    ~NVEncInputAvs();

    virtual RGY_ERR LoadNextFrame(RGYFrame *pSurface) override;
    virtual void Close() override;

protected:
    virtual RGY_ERR Init(const TCHAR *strFileName, VideoInfo *pInputInfo, const void *prm) override;
    RGY_ERR load_avisynth();
    void release_avisynth();

    AVS_ScriptEnvironment *m_sAVSenv;
    AVS_Clip *m_sAVSclip;
    const AVS_VideoInfo *m_sAVSinfo;
    bool m_bInterlaced;

    avs_dll_t m_sAvisynth;
};

#endif //ENABLE_AVISYNTH_READER
