
# NVEncのビルド方法
by rigaya  

## 0. 準備
ビルドには、下記のものが必要です。

- Visual Studio 2015
- CUDA 8.0
- yasm
- Avisynth SDK
- VapourSynth SDK

yasmはパスに追加しておきます。

## 1. ソースのダウンロード

```Batchfile
git clone https://github.com/rigaya/NVEnc --recursive
```

## 2. ffmpeg dllのビルド
ffmpegのdllをビルドし、下記のように配置します。
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

ffmpegのdllのビルド方法はいろいろあるかと思いますが、例えばmsys + mingw環境の場合には、
Visual Studioの環境変数がセットされた状態でビルドすると、
自動的にVC用のdllとlibが作成されます。

例えば、x64のdllをビルドするのであれば、

```Batchfile
call "%VS140COMNTOOLS%\..\..\VC\vcvarsall.bat" x64
call msys.bat
```
としてmsysを起動し、
```
--enable-shared --enable-swresample
```
をつけてビルドすると必要なdllとlibがビルドされます。

## 3. NVEnc.auo / NVEncC のビルド

NVEnc.slnを開きます。

Avisynth SDKの"avisynth_c.h"、
VapourSynth SDKの"VapourSynth.h", "VSScript.h"が
includeパスに含まれるよう、Visual Studio設定した後、ビルドしてください。

ビルドしたいものに合わせて、構成を選択してください。

||Debug用構成|Release用構成|
|:--------------|:----------------------------------|
|NVEnc.auo (win32のみ) | Debug | Release |
|NVEncC(64).exe | DebugStatic | RelStatic |
