//  -----------------------------------------------------------------------------------------
//    拡張 x264 出力(GUI) Ex  v1.xx by rigaya
//  -----------------------------------------------------------------------------------------
//   ソースコードについて
//   ・無保証です。
//   ・本ソースコードを使用したことによるいかなる損害・トラブルについてrigayaは責任を負いません。
//   以上に了解して頂ける場合、本ソースコードの使用、複製、改変、再頒布を行って頂いて構いません。
//  -----------------------------------------------------------------------------------------

#ifndef _AUO_AUDIO_PARALLEL_H_
#define _AUO_AUDIO_PARALLEL_H_

#include <process.h>
#include "auo.h"
#include "auo_system.h"

static inline void if_valid_wait_for_single_object(HANDLE he, DWORD dwMilliseconds) {
	if (he) WaitForSingleObject(he, dwMilliseconds);
}
static inline void if_valid_set_event(HANDLE he) {
	if (he) SetEvent(he);
}

void release_audio_parallel_events(PRM_ENC *pe);

#endif //_AUO_AUDIO_PARALLEL_H_
