
# NVEnc  
by rigaya  

[![Build status](https://ci.appveyor.com/api/projects/status/dmlkxw4rbrby0oi9/branch/master?svg=true)](https://ci.appveyor.com/project/rigaya/nvenc/branch/master) [![Build Status](https://travis-ci.org/rigaya/NVEnc.svg?branch=master)](https://travis-ci.org/rigaya/NVEnc)  

このソフトウェアは、NVIDIAのGPU/APUに搭載されているHWエンコーダ(NVENC)の画質や速度といった性能の実験を目的としています。[Aviutl](http://spring-fragrance.mints.ne.jp/aviutl/)の出力プラグイン版と単体で動作するコマンドライン版があります。  

- NVEnc.auo … NVIDIAのNVEncを使用してエンコードを行う[Aviutl](http://spring-fragrance.mints.ne.jp/aviutl/)の出力プラグインです。  
- NVEncC.exe … 上記のコマンドライン版です。
- cufilters … [Aviutl](http://spring-fragrance.mints.ne.jp/aviutl/)用CUDAフィルタです。

## 配布場所 & 更新履歴  
[rigayaの日記兼メモ帳＞＞](http://rigaya34589.blog135.fc2.com/blog-category-17.html)  

## 基本動作環境  
Windows 10 (x86/x64)  
Linux (x64)  
[Aviutl](http://spring-fragrance.mints.ne.jp/aviutl/) 1.00 以降 (NVEnc.auo)  
NVEncが載ったハードウェア  
  NVIDIA製 GPU GeForce Kepler世代以降 (GT/GTX 6xx 以降)  
  ※GT 63x, 62x等はFermi世代のリネームであるため非対応なものがあります。  

|NVEnc       |対応するNVENC SDK API|必要なグラフィックドライバのバージョン   |
|:--------------- |:-------------- |:--------------------------------------- |
| NVEnc 0.00 以降 | 4.0            | NVIDIA グラフィックドライバ 334.89 以降 |
| NVEnc 1.00 以降 | 5.0            | NVIDIA グラフィックドライバ 347.09 以降 |
| NVEnc 2.00 以降 | 6.0            | NVIDIA グラフィックドライバ 358 以降    |
| NVEnc 2.08 以降 | 7.0            | NVIDIA グラフィックドライバ 368.69 以降 |
| NVEnc 3.02 以降 | 7.0            | NVIDIA グラフィックドライバ 369.30 以降 |
| NVEnc 3.08 以降 | 8.0            | NVIDIA グラフィックドライバ 378.66 以降 |
| NVEnc 4.00 以降 | 8.1            | NVIDIA グラフィックドライバ 390.77 以降 |
| NVEnc 4.31 以降 | 9.0            | NVIDIA グラフィックドライバ 418.81 以降 |
| NVEnc 4.51 以降 | 9.1            | NVIDIA グラフィックドライバ 436.15 以降 |
| NVEnc 5.10 以降 | 9.0, 9.1, 10.0 | NVIDIA グラフィックドライバ 418.81 以降 |

| 対応するNVENC SDK API | required graphics driver version |
|:-------------- |:--------------------------------- |
| 9.0  | NVIDIA graphics driver (Win 418.81 / Linux 418.30) or later |
| 9.1  | NVIDIA graphics driver (Win 436.15 / Linux 435.21) or later |
| 10.0 | NVIDIA graphics driver (Win 445.87 / Linux 450.51) or later |

## NVEncCの使用方法とオプション  
[NVEncCのオプションの説明](./NVEncC_Options.ja.md)

## 各GPUのエンコード機能情報の調査結果  
NVEncC --check-features の結果をまとめたものです。ドライバに問い合わせた結果となっています。そのため、ドライバのバージョンによって結果が異なる可能性があります。 

| GPU世代 | Windows | Linux |
|:---|:---|:---|
| Kepler | [GTX660Ti](./GPUFeatures/gtx660ti.txt) | [Tesla K80](./GPUFeatures/teslaK80_linux.txt) |
| Maxwell | [GTX970](./GPUFeatures/gtx970.txt) | [Tesla M80](./GPUFeatures/teslaM80_linux.txt) |
| Pascal | [GTX1080](./GPUFeatures/gtx1080.txt), [GTX1070](./GPUFeatures/gtx1070.txt), [GTX1060](./GPUFeatures/gtx1060.txt) | [GTX1080](./GPUFeatures/gtx1080_linux.txt) |
| Volta | [GTX1650](./GPUFeatures/gtx1650.txt) | |
| Turing | [RTX2070](./GPUFeatures/rtx2070.txt), [RTX2060](./GPUFeatures/rtx2060.txt), [GTX1660Ti](./GPUFeatures/gtx1660ti.txt), [GTX1650 Super](./GPUFeatures/gtx1650super.txt)  | [Tesla T4](./GPUFeatures/teslaT4_linux.txt)  |

## NVEnc 使用にあたっての注意事項  
無保証です。自己責任で使用してください。   
NVEncを使用したことによる、いかなる損害・トラブルについても責任を負いません。  

NVEncによる出力は、max_dec_frame_buffering フィールドを含まないことがあり、
一部の再生環境では問題となることがあります。

## 使用出来る主な機能
### NVEnc/NVEncC共通
- NVENCを使用したエンコード
   - H.264/AVC
      - YUV4:4:4対応
   - H.265/HEVC (第2世代Maxwell以降)
      - YUV4:4:4対応
      - 10bit
- NVENCの各エンコードモード
   - CQP       固定量子化量
   - CBR       固定ビットレート
   - CBRHQ     固定ビットレート (高品質)
   - VBR       可変ビットレート
   - VBRHQ     可変ビットレート (高品質)
- インタレ保持エンコード (PAFF方式)
- colormatrix等の指定
- SAR比指定
- H.264 Level / Profileの指定
- 最大ビットレート等の指定
- 最大GOP長の指定
- ロスレス出力 (YUV 420 / YUV 444)

### NVEnc.auo
- 音声エンコード
- 音声及びチャプターとのmux機能
- 自動フィールドシフト対応

### NVEncC
- cuvidデコードに対応
  - MPEG1
  - MPEG2
  - H.264/AVC
  - HEVC (10bit/12bit YUV4:4:4対応)
  - VC-1
  - VP9
- avs, vpy, y4m, rawなど各種形式に対応
- エンコード結果のSSIM/PSNRを計算
- GPUを使用した高速フィルタリング
  - cuvid内蔵のhw処理
    - リサイズ
    - インタレ解除 (normal / bob)
  - CUDAによるGPUフィルタリング
    - rff (rffフラグの適用)
    - インタレ解除
      - afs (自動フィールドシフト)
      - nnedi
      - yadif
    - delogo
    - 字幕焼きこみ
    - 色空間変換 (x64版のみ)
      - hdr2sdr
    - リサイズ  
      - bilinear
      - spline16, spline36, spline64
      - lanczos2, lanczos3, lanczos4
      - nppライブラリによる各種アルゴリズム (x64版のみ)
    - パディング(黒帯)の追加
    - フレーム間引き(select every)
    - バンディング低減
    - ノイズ除去
      - knn (K-nearest neighbor)
      - pmd (正則化pmd法)
      - gauss (nppライブラリ、x64版のみ)
    - 輪郭・ディテール強調
      - unsharp
      - edgelevel (エッジレベル調整)

### cufilters.auf
- 対応フィルタ
  - nnedi
  - リサイズ
  - ノイズ除去
    - knn (K-nearest neighbor)
    - pmd (正則化pmd法)
  - 輪郭・ディテール強調
    - unsharp
    - edgelevel (エッジレベル調整)
  - バンディング低減

### NVEncのソースコードについて
- MITライセンスです。
- 本プログラムは、NVIDA CUDA Samplesをベースに作成されており、サンプルコードを含みます。  
  This software contains source code provided by NVIDIA Corporation.  
- 本ソフトウェアでは、
  [jitify](https://github.com/NVIDIA/jitify),
  [ffmpeg](https://ffmpeg.org/),
  [tinyxml2](http://www.grinninglizard.com/tinyxml2/),
  [dtl](https://github.com/cubicdaiya/dtl),
  [libass](https://github.com/libass/libass),
  [ttmath](http://www.ttmath.org/),
  [Caption2Ass](https://github.com/maki-rxrz/Caption2Ass_PCR)を使用しています。  
  これらのライセンスにつきましては、該当ソースのヘッダ部分や、NVEnc_license.txtをご覧ください。

- ビルド方法については[こちら](./Build.ja.md)

### ソースの構成
Windows ... VCビルド  

文字コード: UTF-8-BOM  
改行: CRLF  
インデント: 空白x4  
