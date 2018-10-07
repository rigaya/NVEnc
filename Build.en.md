
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

## 2. Build NVEncC.exe / NVEnc.auo

After preparations are done, open NVEnc.sln, and set headers below in the include path.

 - "avisynth_c.h"、
 - "VapourSynth.h", "VSScript.h"

Finally, start build of NVEnc by Visual Studio.

|  |For Debug build|For Release build|
|:--------------|:--------------|:--------|
|NVEnc.auo (win32 only) | Debug | Release |
|NVEncC(64).exe | DebugStatic | RelStatic |
