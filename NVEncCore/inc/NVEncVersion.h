//  -----------------------------------------------------------------------------------------
//    NVEnc by rigaya
//  -----------------------------------------------------------------------------------------
//   ソースコードについて
//   ・無保証です。
//   ・本ソースコードを使用したことによるいかなる損害・トラブルについてrigayaは責任を負いません。
//   以上に了解して頂ける場合、本ソースコードの使用、複製、改変、再頒布を行って頂いて構いません。
//  -----------------------------------------------------------------------------------------

#pragma once

#define VER_FILEVERSION              0,1,12,0
#define VER_STR_FILEVERSION          "1.12"
#define VER_STR_FILEVERSION_TCHAR _T("1.12")

#ifdef _M_IX86
#define BUILD_ARCH_STR _T("x86")
#else
#define BUILD_ARCH_STR _T("x64")
#endif

#ifdef NVENC_AUO
#define FOR_AUO    1
#define RAW_READER 0
#define AVI_READER 0
#define AVS_READER 0
#define VPY_READER 0
#else
#define FOR_AUO    0
#define RAW_READER 1
#define AVI_READER 0
#define AVS_READER 1
#define VPY_READER 1
#endif

