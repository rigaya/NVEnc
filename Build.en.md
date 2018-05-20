
# How to build NVEnc
by rigaya  

## 0. Requirements
To build NVEnc, components below are required.

- Visual Studio 2015
- CUDA 8.0
- yasm
- Avisynth SDK
- VapourSynth SDK

Please set yasm to your environment PATH.

## 1. Download source code

```Batchfile
git clone https://github.com/rigaya/NVEnc --recursive
```

## 2. Build ffmpeg dll

NVEncC requires ffmpeg dlls, and it should be placed as the structure below.
```
NVEnc root
 |-NVEnc
 |-NVEncC
 |-NVEncCore
 |-NVEncSDK
 |-<others>...
 `-ffmpeg_lgpl
    |- include
    |   |-libavcodec
    |   |  `- libavcodec header files
    |   |-libavfilter
    |   |  `- libavfilter header files
    |   |-libavformat
    |   |  `- libavfilter header files
    |   |-libavutil
    |   |  `- libavutil header files
    |   `-libswresample
    |      `- libswresample header files
    `- lib
        |-win32 (for win32 build)
        |  `- avocdec, avfilter, avformat, avutil, swresample
        |     x86 lib & dlls
        `- x64 (for x64 build)
           `- avocdec, avfilter, avformat, avutil, swresample
              x64 lib & dlls
```

One of the way to build ffmpeg dlls is to use msys+mingw, and when Visual Studio's environment path is set, ffmpeg will build dlls & libs on shared lib build.

For example, if you need x64 build, you can set Visual Studio's environment path be calling vcvarsall.bat before msys.bat call.

Sample script to build dlls can ne found [here](https://github.com/rigaya/build_scripts/tree/master/ffmpeg_dll).

By starting MSYS2 from the bat file in "laucher" dir, MSYS2 could be run with Visual Studio's environment path set. Then, running build_ffmpeg_dll.sh will build the dlls. 

## 3. Build NVEncC / NVEnc.auo

After preparations are done, open NVEnc.sln, and set headers below in the include path.

 - "avisynth_c.h"、
 - "VapourSynth.h", "VSScript.h"

Finally, start build of NVEnc by Visual Studio.

|  |For Debug build|For Release build|
|:--------------|:--------------|:--------|
|NVEnc.auo (win32 only) | Debug | Release |
|NVEncC(64).exe | DebugStatic | RelStatic |
