//  -----------------------------------------------------------------------------------------
//    拡張 x264 出力(GUI) Ex  v1.xx by rigaya
//  -----------------------------------------------------------------------------------------
//   ソースコードについて
//   ・無保証です。
//   ・本ソースコードを使用したことによるいかなる損害・トラブルについてrigayaは責任を負いません。
//   以上に了解して頂ける場合、本ソースコードの使用、複製、改変、再頒布を行って頂いて構いません。
//  -----------------------------------------------------------------------------------------

#ifndef _FAWCHECK_H_
#define _FAWCHECK_H_

enum {
    NON_FAW  = 0,
    FAW_FULL = 1,
    FAW_HALF = 2,
    FAW_MIX  = 3,
    FAWCHECK_ERROR_OTHER     = -1,
    FAWCHECK_ERROR_TOO_SHORT = -2,
};

static const char *const FAW_TYPE_NAME[] = { "non-FAW", "full size", "half size", "half size mix" };

int FAWCheck(short *audio_dat, int audio_n, int audio_rate, int audio_size); //FAWCheckを行い、判定結果を返す

#endif //_FAWCHECK_H_