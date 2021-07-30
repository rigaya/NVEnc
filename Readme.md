
# NVEnc
by rigaya

**[日本語版はこちら＞＞](./Readme.ja.md)**
**[简体中文版本＞＞](./Readme.cn.md)**

[![Build status](https://ci.appveyor.com/api/projects/status/dmlkxw4rbrby0oi9/branch/master?svg=true)](https://ci.appveyor.com/project/rigaya/nvenc/branch/master)  [![Build Linux Packages](https://github.com/rigaya/NVEnc/actions/workflows/build_packages.yml/badge.svg)](https://github.com/rigaya/NVEnc/actions/workflows/build_packages.yml)   

This software is meant to investigate performance and image quality of HW encoder (NVENC) of NVIDIA.
There are 2 types of software developed, one is command line version that runs independently, and the nother is a output plug-in of [Aviutl](http://spring-fragrance.mints.ne.jp/aviutl/).

- NVEncC.exe ... Command line version supporting transcoding.  
- NVEnc.auo ... Output plugin for [Aviutl](http://spring-fragrance.mints.ne.jp/aviutl/).

## Downloads & update history
[rigayaの日記兼メモ帳＞＞](http://rigaya34589.blog135.fc2.com/blog-category-17.html)  
[github releases](https://github.com/rigaya/NVEnc/releases)  
  
## Install
[Install instructions for Windows and Linux](./Install.en.md).

## System Requirements
Windows 10 (x86 / x64)  
Linux (x64)  
Aviutl 1.00 or later (NVEnc.auo)  
Hardware which supports NVENC  
  NVIDIA GPU GeForce Kepler gen or later (GT / GTX 6xx or later)  
  ※ Since GT 63x, 62x etc. are renames of the Fermi generation, they cannot run NVEnc.

| NVEnc               | Supported NVENC API | Required graphics driver version       |
|:----------------- |:------------------ |:----------------------------        |
| NVEnc 0.00 or later | 4.0                  | NVIDIA graphics driver 334.89 or later |
| NVEnc 1.00 or later | 5.0                  | NVIDIA graphics driver 347.09 or later |
| NVEnc 2.00 or later | 6.0                  | NVIDIA graphics driver 358 or later    |
| NVEnc 2.08 or later | 7.0                  | NVIDIA graphics driver 368.69 or later |
| NVEnc 3.02 or later | 7.0                  | NVIDIA graphics driver 369.30 or later |
| NVEnc 3.08 or later | 8.0                  | NVIDIA graphics driver 378.66 or later |
| NVEnc 4.00 or later | 8.1                  | NVIDIA graphics driver 390.77 or later |
| NVEnc 4.31 or later | 9.0                  | NVIDIA graphics driver 418.81 or later |
| NVEnc 4.51 or later | 9.1                  | NVIDIA graphics driver 436.15 or later |
| NVEnc 5.10 or later | 9.0, 9.1, 10.0       | NVIDIA graphics driver 418.81 or later |
| NVEnc 5.18 or later | 9.0, 9.1, 10.0, 11.0 | NVIDIA graphics driver 418.81 or later |
| NVEnc 5.24 or later | 9.0, 9.1, 10.0, 11.0 | NVIDIA graphics driver 418.81 or later (x64) <br> NVIDIA graphics driver 456.81 or later (x86) |
| NVEnc 5.36 or later | 9.0, 9.1, 10.0, 11.0, 11.1 | NVIDIA graphics driver 418.81 or later (x64) <br> NVIDIA graphics driver 456.81 or later (x86) |

| Supported NVENC API | Required graphics driver version |
|:-------------- |:--------------------------------- |
| 9.0  | NVIDIA graphics driver (Win 418.81 / Linux 418.30) or later |
| 9.1  | NVIDIA graphics driver (Win 436.15 / Linux 435.21) or later |
| 10.0 | NVIDIA graphics driver (Win 445.87 / Linux 450.51) or later |
| 11.0 | NVIDIA graphics driver (Win 456.71 / Linux 455.28) or later |
| 11.1 | NVIDIA graphics driver (Win 471.41 / Linux 470.57.02) or later |

| CUDA version | Required graphics driver version |
|:------ |:--------------------------------- |
| 10.1    | NVIDIA graphics driver (Win 418.96 / Linux 418.39)    or later |
| 10.2.89 | NVIDIA graphics driver (Win 440.33 / Linux 441.22)    or later |
| 11.0.2  | NVIDIA graphics driver (Win 451.48 / Linux 450.51.05) or later |
| 11.0.3  | NVIDIA graphics driver (Win 451.82 / Linux 450.51.06) or later |
| 11.1.0  | NVIDIA graphics driver (Win 456.38 / Linux 455.23)    or later |
| 11.1.1  | NVIDIA graphics driver (Win 456.81 / Linux 455.32)    or later |
| 11.2    | NVIDIA graphics driver (Win 460.89 / Linux 460.27.04) or later |

## Usage and options of NVEncC
[Option list and details of NVEncC](./NVEncC_Options.en.md)

## Examples of supported encode features  
Result of --check-features, a feature list returned from the driver. May depend on the driver version.  

| GPU Gen | Windows | Linux |
|:---|:---|:---|
| Kepler | [GTX660Ti](./GPUFeatures/gtx660ti.txt) | [Tesla K80](./GPUFeatures/teslaK80_linux.txt) |
| Maxwell | [GTX970](./GPUFeatures/gtx970.txt) | [Tesla M80](./GPUFeatures/teslaM80_linux.txt) |
| Pascal | [GTX1080](./GPUFeatures/gtx1080.txt), [GTX1070](./GPUFeatures/gtx1070.txt), [GTX1060](./GPUFeatures/gtx1060.txt) | [GTX1080](./GPUFeatures/gtx1080_linux.txt) |
| Volta | [GTX1650](./GPUFeatures/gtx1650.txt) | |
| Turing | [RTX2070](./GPUFeatures/rtx2070.txt), [RTX2060](./GPUFeatures/rtx2060.txt), [GTX1660Ti](./GPUFeatures/gtx1660ti.txt), [GTX1650 Super](./GPUFeatures/gtx1650super.txt)  | [Tesla T4](./GPUFeatures/teslaT4_linux.txt)  |
| Ampere | [RTX3090](./GPUFeatures/rtx3090.txt), [RTX3080](./GPUFeatures/rtx3080.txt)  | |

## Precautions for using NVEnc
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND.

The output from NVEnc may not include the max_dec_frame_buffering field,
which might cause problem in some playback environments.

## Main usable functions
### Common to NVEnc / NVEncC
- Encoding using NVENC
   - H.264 / AVC
      - YUV420 / YUV444
   - H.265 / HEVC (2nd Gen Maxwell or later)
      - YUV420 / YUV444
      - 10 bits
- Each encode mode of NVENC
   - CQP (fixed quantization)
   - CBR (Constant bitrate)
   - CBRHQ (Constant bitrate, high quality)
   - VBR (Variable bitrate)
   - VBRHQ (Variable bitrate, high quality)
- Interlaced encoding (by PAFF)
- Lossless output (YUV 420 / YUV 444)
- supports setting of codec profile & level, SAR, colormatrix, maxbitrate, GOP len, etc...

### NVEncC
- Supports cuvid decoding
  - MPEG1
  - MPEG2
  - H.264 / AVC
  - HEVC (10 bit / 12bitdepth with YUV444 support)
  - VC-1
  - VP9
- Supports various formats such as avs, vpy, y4m, and raw
- Supports demux/muxing using libavformat
- Supports decode using libavcodec
- Calculation of ssim/psnr/vmaf of the encode
- High performance filtering (VPP, Video Pre-Processing)
  - cuvid built-in hw processing
    - resize
    - deinterlace (normal / bob)
  - GPU filtering by CUDA
    - rff (apply rff flag)
    - deinterlacer
      - afs (Automatic field shift)
      - nnedi
      - yadif
    - decimate
    - mpdecimate
    - colorspace conversion (x64 version only)
      - hdr2sdr
    - delogo
    - subburn
    - resize
      - bilinear
      - spline16, spline36, spline64
      - lanczos2, lanczos3, lanczos4
      - various algorithms by npp library are available (x64 version only)
    - transpose / rotate / flip
    - padding
    - select-every
    - deband
    - noise reduction
      - knn (K-nearest neighbor)
      - pmd (modified pmd method)
      - gauss (npp library, x64 version only)
    - edge / detail enhancement
      - unsharp
      - edgelevel (edge ​​level adjustment)
      - warpsharp

### NVEnc.auo (Aviutl plugin)
- Audio encoding
- Mux audio and chapter
- afs (Automatic field shift) support

### cufilters.auf (Aviutl plugin)
- supported filters
  - nnedi
  - resize
  - noise reduction
    - knn (K-nearest neighbor)
    - pmd (modified pmd method)
  - edge / detail enhancement
    - unsharp
    - edgelevel (edge ​​level adjustment)
  - deband  


## Auto GPU selection in multi GPU envinronment

NVEncC will automatically select a GPU depending on the options used,
when there are multiple GPUs available which support NVENC.
--device option can be used to specify on which GPU to run manually. 

1. Select GPU which supports...  
  Items below will be checked whether the GPU supports it or not  
  - Codec, Profile, Level
  - Additionally, below items will be checked if specified
    - 10bit depth encoding
    - Lossless encoding
    - Interlaced encoding
    - HW decode support for ssim/psnr/vmaf calculation
  
2. Prefer GPU which supports...  
  - B frame support
  
3. If there are multiple GPUs which suports all the items checked in 1. and 2., GPU below will be prefered.  
  - GPU which has low Video Engine(VE) utilization
  - GPU which has low GPU core utilization
  - newer Generation GPU
  - GPU with more CUDA cores
  
  The purpose of selecting GPU with lower VE/GPU ulitization is to assign tasks to mulitple GPUs
  and improve the throughput of the tasks. Also, newer Gen GPU and GPU with more cores are assumed to
  have improved performance.  
  
  Please note that VE and GPU ulitization are check at the initialization pahse of the app,
  and there are delays in values taken. Therefore, it is likely that the multiple tasks started at the same time
  to run on the same GPU, and divided into multiple GPUs, even if the options are supported in every GPUs.

## NVEnc source code
- MIT license.
- This program is based on NVIDA CUDA Samples and includes sample code.
  This software contains source code provided by NVIDIA Corporation.
- This software depends on
  [jitify](https://github.com/NVIDIA/jitify),
  [ffmpeg](https://ffmpeg.org/),
  [vmaf](https://github.com/Netflix/vmaf),
  [tinyxml2](http://www.grinninglizard.com/tinyxml2/),
  [dtl](https://github.com/cubicdaiya/dtl),
  [libass](https://github.com/libass/libass),
  [ttmath](http://www.ttmath.org/) &
  [Caption2Ass](https://github.com/maki-rxrz/Caption2Ass_PCR).
  For these licenses, please see the header part of the corresponding source and NVEnc_license.txt.

- [How to build](./Build.en.md)

### About source code
Windows ... VC build

Character code: UTF-8-BOM  
Line feed: CRLF  
Indent: blank x4  