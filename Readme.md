
# NVEnc
by rigaya

**[日本語版はこちら＞＞](./Readme.ja.md)**

This software is meant to investigate performance and image quality of HW encoder (NVENC) of NVIDIA.
There are 2 types of software developed, one is command line version that runs independently, and the nother is a output plug-in of [Aviutl](http://spring-fragrance.mints.ne.jp/aviutl/).

- NVEncC.exe ... Command line version supporting transcoding.  
- NVEnc.auo ... Output plugin for [Aviutl](http://spring-fragrance.mints.ne.jp/aviutl/).

## Downloads & update history
[rigayaの日記兼メモ帳＞＞](http://rigaya34589.blog135.fc2.com/blog-category-17.html)

## System Requirements
Windows 7, 8, 8.1, 10 (x86 / x64)  
Aviutl 0.99g4 or later (NVEnc.auo)  
Hardware which supports NVENC  
  NVIDIA GPU GeForce Kepler gen or later (GT / GTX 6xx or later)  
  ※ Since GT 63x, 62x etc. are renames of the Fermi generation, they cannot run NVEnc.

| NVEnc | required graphics driver version |
|:-------------- |:--------------------------------- |
| NVEnc 0.00 or later | NVIDIA graphics driver 334.89 or later |
| NVEnc 1.00 or later | NVIDIA graphics driver 347.09 or later |
| NVEnc 2.00 or later | NVIDIA graphics driver 358 or later |
| NVEnc 2.08 or later | NVIDIA graphics driver 368.69 or later |
| NVEnc 3.02 or later | NVIDIA graphics driver 369.30 or later |
| NVEnc 3.08 or later | NVIDIA graphics driver 378.66 or later |
| NVEnc 4.00 or later | NVIDIA graphics driver 390.77 or later |

## Usage and options of NVEncC
[Option list and details of NVEncC](./NVEncC_Options.en.md)

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
- Lossless output (YUV 444)
- supports setting of codec profile & level, SAR, colormatrix, maxbitrate, GOP len, etc...

### NVEncC
- Supports cuvid decoding
  - MPEG1
  - MPEG2
  - H.264 / AVC
  - HEVC (10 bitdepth support)
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
   - afs (deinterlacer, Automatic field shift)
   - delogo
   - resize
     In addition to bilinear, spline36, various algorithms by npp library are available for x64 version
   - deband
   - noise reduction
     - knn (K-nearest neighbor)
     - pmd (modified pmd method)
     - gauss (npp library, x64 version only)
  - edge / detail enhancement
    - unsharp
    - edgelevel (edge ​​level adjustment)

### NVEnc.auo (Aviutl plugin)
- Audio encoding
- Mux audio and chapter
- afs (Automatic field shift) support

### cufilters.auf (Aviutl plugin)
- supported filters
  - resize
  - noise reduction
    - knn (K-nearest neighbor)
    - pmd (modified pmd method)
  - edge / detail enhancement
    - unsharp
    - edgelevel (edge ​​level adjustment)
  - deband

### NVEnc source code
- MIT license.
- This program is based on NVIDA CUDA Samples and includes sample code.
  This software contains source code provided by NVIDIA Corporation.
- This software depends on
  [ffmpeg](https://ffmpeg.org/),
  [tinyxml2](http://www.grinninglizard.com/tinyxml2/),
  [dtl](https://github.com/cubicdaiya/dtl) & 
  [ttmath](http://www.ttmath.org/).
  For these licenses, please see the header part of the corresponding source and NVEnc_license.txt.

- [How to build](./Build.en.md)

### About source code
Windows ... VC build

Character code: UTF-8-BOM  
Line feed: CRLF  
Indent: blank x4  