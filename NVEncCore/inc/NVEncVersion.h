#pragma once
#ifndef _NVENC_VERSION_H_
#define _NVENC_VERSION_H_

#define VER_FILEVERSION              0,0,0,0
#define VER_STR_FILEVERSION          "0.00"
#define VER_STR_FILEVERSION_TCHAR _T("0.00")

#ifdef _M_IX86
#define BUILD_ARCH_STR _T("x86")
#else
#define BUILD_ARCH_STR _T("x64")
#endif

#endif //_NVENC_VERSION_H_
