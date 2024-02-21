
# NVEnc
by rigaya

**[日本語版はこちら＞＞](./Readme.ja.md)**

[![Build Windows Releases](https://github.com/rigaya/NVEnc/actions/workflows/build_releases.yml/badge.svg)](https://github.com/rigaya/NVEnc/actions/workflows/build_releases.yml) [![Build Linux Packages](https://github.com/rigaya/NVEnc/actions/workflows/build_packages.yml/badge.svg)](https://github.com/rigaya/NVEnc/actions/workflows/build_packages.yml)  

本软件旨在研究NVIDIA的HW编码器(NVENC)的性能和图像质量。
所开发的软件有两种类型，一种是独立运行的命令行版本，另一种是输出插件 [Aviutl](http://spring-fragrance.mints.ne.jp/aviutl/).

- NVEncC.exe ... 支持代码转换的命令行版本。  
- NVEnc.auo ... [Aviutl](http://spring-fragrance.mints.ne.jp/aviutl/) 的输出插件。

## 下载 & 更新历史
[rigayaの日記兼メモ帳＞＞](http://rigaya34589.blog135.fc2.com/blog-category-17.html)
[github releases](https://github.com/rigaya/NVEnc/releases)  

## 安装
[如何安装](./Install.cn.md)

## 构建
[如何构建](./Build.cn.md)


## 系统需求
Windows 10/11 (x86 / x64)  
Linux (x64/aarch64)  
Aviutl 1.00 or later (NVEnc.auo)  
支持NVENC的硬件  
  NVIDIA GPU GeForce Kepler gen或更新(GT / GTX 6xx或更新)  
  ※ 由于GT 63x, 62x等是费米代的重命名，他们不能运行NVEnc。

| NVEnc               | 支持的NVENC API  | 所需图形驱动程序版本 |
|:----------------- |:------------------ |:----------------------------        |
| NVEnc 0.00 or later | 4.0              | NVIDIA graphics driver 334.89 or later |
| NVEnc 1.00 or later | 5.0              | NVIDIA graphics driver 347.09 or later |
| NVEnc 2.00 or later | 6.0              | NVIDIA graphics driver 358 or later    |
| NVEnc 2.08 or later | 7.0              | NVIDIA graphics driver 368.69 or later |
| NVEnc 3.02 or later | 7.0              | NVIDIA graphics driver 369.30 or later |
| NVEnc 3.08 or later | 8.0              | NVIDIA graphics driver 378.66 or later |
| NVEnc 4.00 or later | 8.1              | NVIDIA graphics driver 390.77 or later |
| NVEnc 4.31 or later | 9.0              | NVIDIA graphics driver 418.81 or later |
| NVEnc 4.51 or later | 9.1              | NVIDIA graphics driver 436.15 or later |
| NVEnc 5.10 or later | 9.0 - 10.0       | NVIDIA graphics driver 418.81 or later |
| NVEnc 5.18 or later | 9.0 - 11.0       | NVIDIA graphics driver 418.81 or later |
| NVEnc 5.24 or later | 9.0 - 11.0       | NVIDIA graphics driver 418.81 or later (x64) <br> NVIDIA graphics driver 456.81 or later (x86) |
| NVEnc 5.36 or later | 9.0 - 11.1       | NVIDIA graphics driver 418.81 or later (x64) <br> NVIDIA graphics driver 456.81 or later (x86) |
| NVEnc 7.00 or later | 9.0 - 12.0       | NVIDIA graphics driver 418.81 or later (x64) <br> NVIDIA graphics driver 456.81 or later (x86) |
| NVEnc 7.26 or later | 9.0 - 12.1       | NVIDIA graphics driver 418.81 or later (x64) <br> NVIDIA graphics driver 456.81 or later (x86) |

| 支持的NVENC API | 所需图形驱动程序版本 |
|:-------------- |:--------------------------------- |
| 9.0  | NVIDIA graphics driver (Win 418.81 / Linux 418.30) or later |
| 9.1  | NVIDIA graphics driver (Win 436.15 / Linux 435.21) or later |
| 10.0 | NVIDIA graphics driver (Win 445.87 / Linux 450.51) or later |
| 11.0 | NVIDIA graphics driver (Win 456.71 / Linux 455.28) or later |
| 11.1 | NVIDIA graphics driver (Win 471.41 / Linux 470.57.02) or later |
| 12.0 | ??? |
| 12.1 | NVIDIA graphics driver (Win 531.61 / Linux 530.41.03) or later |

| CUDA 版本 | 所需图形驱动程序版本 |
|:------ |:--------------------------------- |
| 10.1    | NVIDIA graphics driver (Win 418.96 / Linux 418.39)    or later |
| 10.2.89 | NVIDIA graphics driver (Win 440.33 / Linux 441.22)    or later |
| 11.0.2  | NVIDIA graphics driver (Win 451.48 / Linux 450.51.05) or later |
| 11.0.3  | NVIDIA graphics driver (Win 451.82 / Linux 450.51.06) or later |
| 11.1.0  | NVIDIA graphics driver (Win 456.38 / Linux 455.23)    or later |
| 11.1.1  | NVIDIA graphics driver (Win 456.81 / Linux 455.32)    or later |
| 11.2    | NVIDIA graphics driver (Win 460.89 / Linux 460.27.04) or later |

## NVEncC的使用方法和选项
[NVEncC的选项列表](./NVEncC_Options.cn.md)

注:中文文档更新可能不及时，不同之处请参考[其他语言](./NVEncC_Options.en.md)

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
| Ada Lovelace | [RTX4090](./GPUFeatures/rtx4090.txt), [RTX4080](./GPUFeatures/rtx4080.txt) | |

## 使用NVEnc的注意事项
本软件是“按原样”提供的，没有任何形式的保证。

## 主要功能
### NVEnc / NVEncC 的公共特性
- 利用 NVENC 编码
   - H.264 / AVC
      - YUV420 / YUV444
   - H.265 / HEVC (2nd Gen Maxwell or later)
      - YUV420 / YUV444
      - 10 bits
   - AV1 (Ada Lovelace or later)
- NVENC 的编码模式
   - CQP (fixed quantization)
   - CBR (Constant bitrate)
   - CBRHQ (Constant bitrate, high quality)
   - VBR (Variable bitrate)
   - VBRHQ (Variable bitrate, high quality)
- 隔行扫描编码 (利用 PAFF)
- 无损输出 (YUV 420 / YUV 444)
- 支持设置编解码器配置和级别、SAR、颜色滤镜、最大比特率、GOP长度等

### NVEncC
- 支持 cuvid 解码 (NVIDIA 硬件解码)
  - MPEG1
  - MPEG2
  - H.264 / AVC
  - HEVC (10 bit / 12bitdepth with YUV444 support)
  - VC-1
  - VP9
  - AV1
- 支持 avs, vpy, y4m, raw 等格式
- 通过 libavformat 支持封装/解封装
- 通过 libavcodec 支持解码
- 统计编码的ssim/psnr/vmaf参数
- 高性能过滤 (VPP, 视频预处理)
  - cuvid 内建硬件处理
    - resize
    - deinterlace (normal / bob)
  - 使用 CUDA 的 GPU filtering
    - rff (apply rff flag)
    - deinterlacer
      - afs (Automatic field shift)
      - nnedi
      - yadif
    - decimate
    - mpdecimate
    - colorspace conversion (x64 version only)
      - hdr2sdr
      - lut3d
    - delogo
    - subburn
    - resize
      - bilinear
      - spline16, spline36, spline64
      - lanczos2, lanczos3, lanczos4
      - 一系列npp库提供的算法 (x64 version only)
      - nvvfx-superres (超分辨率)
    - transpose / rotate / flip
    - padding
    - select-every
    - deband
    - noise reduction
      - knn (K-nearest neighbor)
      - pmd (modified pmd method)
      - gauss (npp library, x64 version only)
      - convolution3d
      - nvvfx-artifact-reduction
      - nvvfx-denoise
    - edge / detail enhancement
      - unsharp
      - edgelevel (edge ​​level adjustment)
      - warpsharp

### NVEnc.auo (Aviutl 插件)
- 音频编码
- 封装音频和章节
- 支持afs (Automatic field shift)

### cufilters.auf (Aviutl 插件)
- 支持的过滤器:
  - nnedi
  - resize
  - noise reduction
    - knn (K-nearest neighbor)
    - pmd (modified pmd method)
  - edge / detail enhancement
    - unsharp
    - edgelevel (edge ​​level adjustment)
  - deband

## 在多GPU环境中自动选择GPU

当有多个支持NVENC的GPU可用时，NVEncC将根据当前的选项自动选择一个GPU，--device 选项用于手动指定在哪个GPU上运行。

1. 选择支持以下项目的GPU 
  将检查GPU是否支持以下项目
  - 当前使用的编解码器，配置，级别
  - 此外，如果选定了以下项目，也将进行检查
    - 10bit 位深度编码
    - 无损编码
    - 交错编码
    - 硬件支持统计ssim/psnr/vmaf
  
2. 支持以下项目的GPU优先  
  - 支持B帧
  
3. 如果有多个GPU支持1.和2.列出的条件，将选择以下GPU  
  - 媒体引擎占用率较低的GPU
  - 核心占用率较低的GPU
  - 架构较新的GPU
  - 有更多CUDA core的GPU
  
  选择媒体引擎占用率较低的GPU的目的是希望将多个任务分配给不同的GPU并提高任务的吞吐量。
  此外，我们假设较新架构的GPU以及有更多内核的GPU有更高的性能
  
  注意软件会在启动阶段获取媒体引擎和GPU占用率，并且取值过程存在延迟。因此，在有多个GPU且它们都支持当前设定的选项的情况下，若有多个任务同时启动，那么它们可能会选择在同一GPU上运行。

### NVEnc 源码
- MIT license.
- 这个程序是基于NVIDA CUDA样本，包括样本代码。
  本软件包含NVIDIA公司提供的源代码。
- --vpp-nvvfx 过滤器由 NVIDIA BROADCAST 提供技术支持
- 这个软件依赖于
  [jitify](https://github.com/NVIDIA/jitify),
  [ffmpeg](https://ffmpeg.org/),
  [vmaf](https://github.com/Netflix/vmaf),
  [tinyxml2](http://www.grinninglizard.com/tinyxml2/),
  [dtl](https://github.com/cubicdaiya/dtl),
  [libass](https://github.com/libass/libass),
  [ttmath](http://www.ttmath.org/) &
  [Caption2Ass](https://github.com/maki-rxrz/Caption2Ass_PCR).
  对于这些许可证，请参阅相应源代码的标题部分和NVEnc_license.txt。

### 关于源代码
Windows ... VC build

Character code: UTF-8-BOM  
Line feed: CRLF  
Indent: blank x4  
