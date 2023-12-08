
# 如何建立NVEnc
by rigaya  

## 0. 环境需求
要构建NVEnc，需要以下组件。

- Visual Studio 2019
- CUDA 10.1 (x64)
- CUDA 11.0 (x86)
- [Avisynth](https://github.com/AviSynth/AviSynthPlus) SDK
- [VapourSynth](http://www.vapoursynth.com/) SDK

使用sdk安装Avisynth+和VapourSynth。

然后，Avisynth+ SDK 的 "avisynth_c.h" 和VapourSynth SDK 的 "VapourSynth.h" 应被添加到 Visual Studio 的包含路径中。

这些包括路径可以通过环境变量“AVISYNTH_SDK”和“VAPOURSYNTH_SDK”来传递。

使用默认安装，环境变量可以设置如下。
```Batchfile
setx AVISYNTH_SDK "C:\Program Files (x86)\AviSynth+\FilterSDK"
setx VAPOURSYNTH_SDK "C:\Program Files (x86)\VapourSynth\sdk"
```PTION2ASS_SRC Caption2Ass_PCR <path-to-clone>/src
```

## 1. 下载源代码

```Batchfile
git clone https://github.com/rigaya/NVEnc --recursive
cd NVEnc
curl -s -o ffmpeg_lgpl.7z -L https://github.com/rigaya/ffmpeg_dlls_for_hwenc/releases/download/20231123/ffmpeg_dlls_for_hwenc_20231123.7z
7z x -offmpeg_lgpl -y ffmpeg_lgpl.7z
```

## 2. 构建 NVEncC.exe / NVEnc.auo

最后，打开 NVEnc.sln，然后开始使用 Visual Studio 构建 NVEnc。

|   | For Debug build | For Release build |
|:---------------------|:------|:--------|
|NVEncC(64).exe | DebugStatic | RelStatic |
|NVEnc.auo (win32 only) | Debug | Release |
|cufilters.auf (win32 only) | DebugFilters | RelFilters |
