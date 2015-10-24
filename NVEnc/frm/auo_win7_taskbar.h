//  -----------------------------------------------------------------------------------------
//    拡張 x264 出力(GUI) Ex  v1.xx/2.xx by rigaya
//  -----------------------------------------------------------------------------------------
//   ソースコードについて
//   ・無保証です。
//   ・本ソースコードを使用したことによるいかなる損害・トラブルについてrigayaは責任を負いません。
//   以上に了解して頂ける場合、本ソースコードの使用、複製、改変、再頒布を行って頂いて構いません。
//  -----------------------------------------------------------------------------------------

#ifndef _AUO_WIN7_TASKBAR_H_
#define _AUO_WIN7_TASKBAR_H_

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
	void set_visible(bool bVisible) {
		visible = bVisible;
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