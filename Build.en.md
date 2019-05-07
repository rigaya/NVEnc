
# How to build NVEnc
by rigaya  

## 0. Requirements
To build NVEnc, components below are required.

- Visual Studio 2015 + CUDA 8.0 (x86)
- Visual Studio 2015/2017/2019 + CUDA 10.1 (x64)
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

You will also need source code of Caption2Ass_PCR.

```Batchfile
git clone https://github.com/maki-rxrz/Caption2Ass_PCR <path-to-clone>
setx CAPTION2ASS_SRC Caption2Ass_PCR <path-to-clone>/src
```

## 1. Download source code

```Batchfile
git clone https://github.com/rigaya/NVEnc --recursive
```

## 2. Build NVEncC.exe / NVEnc.auo

Finally, open NVEnc.sln, and start build of NVEnc by Visual Studio.

|   | For Debug build | For Release build |
|:---------------------|:------|:--------|
|NVEncC(64).exe | DebugStatic | RelStatic |
|NVEnc.auo (win32 only) | Debug | Release |
|cufilters.auf (win32 only) | DebugFilters | RelFilters |
