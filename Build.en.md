
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

Install Avisynth+ and VapourSynth, with the SDKs.

Then, "avisynth_c.h" of the Avisynth+ SDK and "VapourSynth.h" of the VapourSynth SDK should be added to the include path of Visual Studio.

These include path can be passed by environment variables "AVISYNTH_SDK" and "VAPOURSYNTH_SDK".

With default installation, environment variables could be set as below.
```Batchfile
setx AVISYNTH_SDK "C:\Program Files (x86)\AviSynth+\FilterSDK"
setx VAPOURSYNTH_SDK "C:\Program Files (x86)\VapourSynth\sdk"
```

## 1. Download source code

```Batchfile
git clone https://github.com/rigaya/NVEnc --recursive
```
```

## 2. Build NVEncC.exe / NVEnc.auo

Finally, open NVEnc.sln, and start build of NVEnc by Visual Studio.

|  |For Debug build|For Release build|
|:--------------|:--------------|:--------|
|NVEnc.auo (win32 only) | Debug | Release |
|NVEncC(64).exe | DebugStatic | RelStatic |
