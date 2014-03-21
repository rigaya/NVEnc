//  -----------------------------------------------------------------------------------------
//    拡張 x264 出力(GUI) Ex  v1.xx by rigaya
//  -----------------------------------------------------------------------------------------
//   ソースコードについて
//   ・無保証です。
//   ・本ソースコードを使用したことによるいかなる損害・トラブルについてrigayaは責任を負いません。
//   以上に了解して頂ける場合、本ソースコードの使用、複製、改変、再頒布を行って頂いて構いません。
//  -----------------------------------------------------------------------------------------

#include <Windows.h>
// Windows SDK 7.1の<Shobjidl.h>が呼ばれるようにすること
#include <Shobjidl.h>
#pragma comment(lib, "ole32.lib")
#include "auo_util.h"
#include "auo_frm.h"

static ITaskbarList3 *g_pTaskbarList = NULL;
static BOOL enabled = false;
static const int MAX_PROGRESS = 1000;

void taskbar_progress_enable(BOOL _enable) {
	enabled = _enable;
}

void taskbar_progress_init() {
	if (!check_OS_Win7orLater() || CoCreateInstance(CLSID_TaskbarList, NULL, CLSCTX_INPROC_SERVER, IID_PPV_ARGS(&g_pTaskbarList)) != S_OK)
		g_pTaskbarList = NULL;
}

void taskbar_progress_start(HWND hWnd, int mode) {
	if (g_pTaskbarList) {
		g_pTaskbarList->SetProgressValue(hWnd, 0, MAX_PROGRESS);
		g_pTaskbarList->SetProgressState(hWnd, (!enabled || mode == PROGRESSBAR_DISABLED) ? TBPF_NOPROGRESS : ((mode == PROGRESSBAR_MARQUEE) ? TBPF_INDETERMINATE : TBPF_NORMAL));
	}
}

void taskbar_progress_paused(HWND hWnd) {
	if (g_pTaskbarList && enabled)
		g_pTaskbarList->SetProgressState(hWnd, TBPF_PAUSED);
}

void taskbar_setprogress(HWND hWnd, double progress) {
	if (g_pTaskbarList) {
		const int MAX_PROGRESS = 1000;
		g_pTaskbarList->SetProgressValue(hWnd, (enabled) ? (int)(MAX_PROGRESS * progress + 0.5) : 0, MAX_PROGRESS);
		g_pTaskbarList->SetProgressState(hWnd, (enabled) ? TBPF_NORMAL : TBPF_NOPROGRESS);
	}
}