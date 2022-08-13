// -----------------------------------------------------------------------------------------
// x264guiEx/x265guiEx/svtAV1guiEx/ffmpegOut/QSVEnc/NVEnc/VCEEnc by rigaya
// -----------------------------------------------------------------------------------------
// The MIT License
//
// Copyright (c) 2010-2022 rigaya
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
// --------------------------------------------------------------------------------------------

#ifndef _AUO_WIN7_TASKBAR_H_
#define _AUO_WIN7_TASKBAR_H_

#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <Windows.h>
#include <Shobjidl.h>
#pragma comment(lib, "ole32.lib")
#include "auo_util.h"
#include "auo_frm.h"

class taskbarProgress {
private:
    bool visible = false; //表示・非表示の切り替え
    int currentMode = PROGRESSBAR_DISABLED; //visibleに関係ない、現在のモード
    HWND hWnd = NULL; //ウィンドウハンドル
    ITaskbarList3 *pTaskbarList = nullptr;
    static const int MAX_PROGRESS = 100;

    void init() {
        if (nullptr != pTaskbarList) {
            currentMode = PROGRESSBAR_DISABLED;
            pTaskbarList->SetProgressState(hWnd, TBPF_NOPROGRESS);
            pTaskbarList->SetProgressValue(hWnd, 0, MAX_PROGRESS);
            visible = true;
        }
    }
public:
    taskbarProgress(HWND _hWnd) {
        hWnd = _hWnd;
        if (!check_OS_Win7orLater()
            || S_OK != CoCreateInstance(CLSID_TaskbarList, nullptr, CLSCTX_INPROC_SERVER, IID_PPV_ARGS(&pTaskbarList))) {
            pTaskbarList = nullptr;
        }
        init();
    };
    ~taskbarProgress() {
        init();
        if (nullptr != pTaskbarList) {
            CoUninitialize();
            pTaskbarList = nullptr;
        }
    };
    //表示・非表示の切り替え
    void set_visible(bool value) {
        visible = value;
        set_mode(currentMode);
    }
    //モードの設定
    void set_mode(int mode) {
        currentMode = mode;
        if (nullptr != pTaskbarList) {
            pTaskbarList->SetProgressState(hWnd, (!visible || mode == PROGRESSBAR_DISABLED) ? TBPF_NOPROGRESS : ((mode == PROGRESSBAR_MARQUEE) ? TBPF_INDETERMINATE : TBPF_NORMAL));
        }
    }
    //一時停止、解除にはrestartを使うこと
    void pause() {
        if (nullptr != pTaskbarList) {
            pTaskbarList->SetProgressState(hWnd, TBPF_PAUSED);
        }
    }
    //一時停止解除
    void restart() {
        if (nullptr != pTaskbarList) {
            set_mode(currentMode);
        }
    }
    //進捗を設定(0～1)
    void set_progress(double progress) {
        if (nullptr != pTaskbarList) {
            pTaskbarList->SetProgressValue(hWnd, (int)(MAX_PROGRESS * progress + 0.5), MAX_PROGRESS);
        }
    }
};

#endif //_AUO_WIN7_TASKBAR_H_