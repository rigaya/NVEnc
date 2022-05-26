
# NVEnc
by rigaya

**[日本語版はこちら＞＞](./Readme.ja.md)**

[![Build Windows Releases](https://github.com/rigaya/NVEnc/actions/workflows/build_releases.yml/badge.svg)](https://github.com/rigaya/NVEnc/actions/workflows/build_releases.yml) [![Build Linux Packages](https://github.com/rigaya/NVEnc/actions/workflows/build_packages.yml/badge.svg)](https://github.com/rigaya/NVEnc/actions/workflows/build_packages.yml)  

本软件旨在研究NVIDIA的HW编码器(NVENC)的性能和图像质量。
所开发的软件有两种类型，一种是独立运行的命令行版本，另一种是输出插件 [Aviutl](http://spring-fragrance.mints.ne.jp/aviutl/).

- NVEncC.exe ... 支持代码转换的命令行版本。  
- NVEnc.auo ... [Aviutl](http://spring-fragrance.mints.ne.jp/aviutl/) 的输出插件。

## 下载 & update history
[rigayaの日記兼メモ帳＞＞](http://rigaya34589.blog135.fc2.com/blog-category-17.html)

## 系统需求
Windows 10/11 (x86 / x64)  
Aviutl 1.00 or later (NVEnc.auo)  
支持NVENC的硬件  
  NVIDIA GPU GeForce Kepler gen或更新(GT / GTX 6xx或更新)  
  ※ 由于GT 63x, 62x等是费米代的重命名，他们不能运行NVEnc。

| NVEnc | 所需图形驱动程序版本 |
|:-------------- |:--------------------------------- |
| NVEnc 0.00 or later | NVIDIA graphics driver 334.89 or later |
| NVEnc 1.00 or later | NVIDIA graphics driver 347.09 or later |
| NVEnc 2.00 or later | NVIDIA graphics driver 358 or later |
| NVEnc 2.08 or later | NVIDIA graphics driver 368.69 or later |
| NVEnc 3.02 or later | NVIDIA graphics driver 369.30 or later |
| NVEnc 3.08 or later | NVIDIA graphics driver 378.66 or later |
| NVEnc 4.00 or later | NVIDIA graphics driver 390.77 or later |
| NVEnc 4.31 or later | NVIDIA graphics driver 418.81 or later |
| NVEnc 4.51 or later | NVIDIA graphics driver 436.15 or later |

## NVEncC的使用和选项
[选项列表和详细的NVEncC](./NVEncC_Options.en.md)

## 支持的编码特性的示例  
check-features的结果，驱动程序返回的功能列表。可能取决于驱动程序版本。

| GPU Gen | Windows | Linux |
|:---|:---|:---|
| Kepler | [GTX660Ti](./GPUFeatures/gtx660ti.txt) | [Tesla K80](./GPUFeatures/teslaK80_linux.txt) |
| Maxwell | [GTX970](./GPUFeatures/gtx970.txt) | [Tesla M80](./GPUFeatures/teslaM80_linux.txt) |
| Pascal | [GTX1080](./GPUFeatures/gtx1080.txt), [GTX1070](./GPUFeatures/gtx1070.txt), [GTX1060](./GPUFeatures/gtx1060.txt), [GTX1050Ti](./GPUFeatures/gtx1050ti.txt) | [GTX1080](./GPUFeatures/gtx1080_linux.txt) |
| Volta | [GTX1650](./GPUFeatures/gtx1650.txt) | |
| Turing | [RTX2070](./GPUFeatures/rtx2070.txt), [RTX2060](./GPUFeatures/rtx2060.txt), [GTX1660Ti](./GPUFeatures/gtx1660ti.txt), [GTX1650 Super](./GPUFeatures/gtx1650super.txt)  | [Tesla T4](./GPUFeatures/teslaT4_linux.txt)  |
| Ampere | [RTX3090](./GPUFeatures/rtx3090.txt), [RTX3080](./GPUFeatures/rtx3080.txt), [RTX3050Ti](./GPUFeatures/rtx3050ti.txt)  | |

## 使用NVEnc的注意事项
本软件是“按原样”提供的，没有任何形式的保证。

NVEnc的输出可能不包括max_dec_frame_buffering字段，
这可能会在某些回放环境中引起问题。

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
  - VP9
- Supports various formats such as avs, vpy, y4m, and raw
- Supports demux/muxing using libavformat
- Supports decode using libavcodec
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
    - colorspace conversion (x64 version only)
      - hdr2sdr
    - delogo
    - subburn
    - resize
      - bilinear
      - spline16, spline36, spline64
      - lanczos2, lanczos3, lanczos4
      - various algorithms by npp library are available (x64 version only)
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

### NVEnc 源码
- MIT license.
- 这个程序是基于NVIDA CUDA样本，包括样本代码。
  本软件包含NVIDIA公司提供的源代码。
- 这个软件依赖于
  [jitify](https://github.com/NVIDIA/jitify),
  [ffmpeg](https://ffmpeg.org/),
  [tinyxml2](http://www.grinninglizard.com/tinyxml2/),
  [dtl](https://github.com/cubicdaiya/dtl),
  [libass](https://github.com/libass/libass),
  [ttmath](http://www.ttmath.org/) &
  [Caption2Ass](https://github.com/maki-rxrz/Caption2Ass_PCR).
  对于这些许可证，请参阅相应源代码的标题部分和NVEnc_license.txt。

- [如何构建](./Build.cn.md)

### 关于源代码
Windows ... VC build

Character code: UTF-8-BOM  
Line feed: CRLF  
Indent: blank x4  
