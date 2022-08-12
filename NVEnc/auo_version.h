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

#ifndef _AUO_VERSION_H_
#define _AUO_VERSION_H_

#include "rgy_version.h"

#define AUO_VERSION          VER_FILEVERSION
#define AUO_VERSION_STR      VER_STR_FILEVERSION
#define AUO_NAME_WITHOUT_EXT "NVEnc"
#define AUO_NAME             "NVEnc.auo"
#define AUO_NAME_W          L"NVEnc.auo"
#define AUO_FULL_NAME        "拡張 NVEnc 出力"
#define AUO_VERSION_NAME     "拡張 NVEnc 出力 " AUO_VERSION_STR
#define AUO_VERSION_INFO     "拡張 NVEnc 出力 " AUO_VERSION_STR " by rigaya"
#define AUO_EXT_FILTER       "All Support Formats (*.*)\0*.mp4;*.mkv;*.264;*.mp4\0mp4 file (*.mp4)\0*.mp4\0mkv file (*.mkv)\0*.mkv\0raw file (*.264)\0*.264\0"

#ifdef DEBUG
#define VER_DEBUG   VS_FF_DEBUG
#define VER_PRIVATE VS_FF_PRIVATEBUILD
#else
#define VER_DEBUG   0
#define VER_PRIVATE 0
#endif

#define VER_STR_COMMENTS         AUO_FULL_NAME
#define VER_STR_COMPANYNAME      ""
#define VER_STR_FILEDESCRIPTION  AUO_FULL_NAME
#define VER_STR_INTERNALNAME     AUO_FULL_NAME
#define VER_STR_ORIGINALFILENAME AUO_NAME
#define VER_STR_LEGALCOPYRIGHT   "拡張 NVEnc 出力 by rigaya"
#define VER_STR_PRODUCTNAME      "NVEnc"
#define VER_PRODUCTVERSION       VER_FILEVERSION
#define VER_STR_PRODUCTVERSION   VER_STR_FILEVERSION

#endif //_AUO_VERSION_H_
