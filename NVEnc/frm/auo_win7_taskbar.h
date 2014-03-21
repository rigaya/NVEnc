//  -----------------------------------------------------------------------------------------
//    拡張 x264 出力(GUI) Ex  v1.xx by rigaya
//  -----------------------------------------------------------------------------------------
//   ソースコードについて
//   ・無保証です。
//   ・本ソースコードを使用したことによるいかなる損害・トラブルについてrigayaは責任を負いません。
//   以上に了解して頂ける場合、本ソースコードの使用、複製、改変、再頒布を行って頂いて構いません。
//  -----------------------------------------------------------------------------------------

#ifndef _AUO_WIN7_TASKBAR_H_
#define _AUO_WIN7_TASKBAR_H_

#include <Windows.h>

void taskbar_progress_enable(BOOL _enable);
void taskbar_progress_init();
void taskbar_progress_start(HWND hWnd, int mode);
void taskbar_progress_paused(HWND hWnd);
void taskbar_setprogress(HWND hWnd, double progress);

#endif //_AUO_WIN7_TASKBAR_H_