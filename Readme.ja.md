
# NVEnc  
by rigaya  

[![Build status](https://ci.appveyor.com/api/projects/status/dmlkxw4rbrby0oi9/branch/master?svg=true)](https://ci.appveyor.com/project/rigaya/nvenc/branch/master)  

このソフトウェアは、NVIDIAのGPU/APUに搭載されているHWエンコーダ(NVENC)の画質や速度といった性能の実験を目的としています。[Aviutl](http://spring-fragrance.mints.ne.jp/aviutl/)の出力プラグイン版と単体で動作するコマンドライン版があります。  

- NVEnc.auo … NVIDIAのNVEncを使用してエンコードを行う[Aviutl](http://spring-fragrance.mints.ne.jp/aviutl/)の出力プラグインです。  
- NVEncC.exe … 上記のコマンドライン版です。
- cufilters … [Aviutl](http://spring-fragrance.mints.ne.jp/aviutl/)用CUDAフィルタです。

## 配布場所 & 更新履歴  
[rigayaの日記兼メモ帳＞＞](http://rigaya34589.blog135.fc2.com/blog-category-17.html)  

## 基本動作環境  
Windows 7, 8, 8.1, 10 (x86/x64)  
[Aviutl](http://spring-fragrance.mints.ne.jp/aviutl/) 0.99g4 以降 (NVEnc.auo)  
NVEncが載ったハードウェア  
  NVIDIA製 GPU GeForce Kepler世代以降 (GT/GTX 6xx 以降)  
  ※GT 63x, 62x等はFermi世代のリネームであるため非対応なものがあります。  

|NVEnc|必要なグラフィックドライバのバージョン|
|:--------------|:----------------------------------|
|NVEnc 0.00 以降 | NVIDIA グラフィックドライバ 334.89以降 |
|NVEnc 1.00 以降 | NVIDIA グラフィックドライバ 347.09以降 |
|NVEnc 2.00 以降 | NVIDIA グラフィックドライバ 358   以降 |
|NVEnc 2.08 以降 | NVIDIA グラフィックドライバ 368.69以降 |
|NVEnc 3.02 以降 | NVIDIA グラフィックドライバ 369.30以降 |
|NVEnc 3.08 以降 | NVIDIA グラフィックドライバ 378.66以降 |
|NVEnc 4.00 以降 | NVIDIA グラフィックドライバ 390.77以降 |

## NVEncCの使用方法とオプション  
NVEncCのオプションの説明 ([blog](http://rigaya34589.blog135.fc2.com/blog-entry-739.html), [github](./NVEncC_Options.ja.md)) 

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
- ロスレス出力 (YUV444)

### NVEnc.auo
- 音声エンコード
- 音声及びチャプターとのmux機能
- 自動フィールドシフト対応

### NVEncC
- cuvidデコードに対応
  - MPEG1
  - MPEG2
  - H.264/AVC
  - HEVC (10bit対応)
  - VP9
- avs, vpy, y4m, rawなど各種形式に対応
- GPUを使用した高速フィルタリング
  - cuvid内蔵のhw処理
   - リサイズ
   - インタレ解除 (normal / bob)
  - CUDAによるGPUフィルタリング
   - rff (rffフラグの適用)
   - afs (自動フィールドシフト)
   - delogo
   - リサイズ  
     bilinear,spline36に加え、x64版ではnppライブラリによる各種アルゴリズムが利用可
   - パディング(黒帯)の追加
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
  [ffmpeg](https://ffmpeg.org/),
  [tinyxml2](http://www.grinninglizard.com/tinyxml2/),
  [dtl](https://github.com/cubicdaiya/dtl),
  [ttmath](http://www.ttmath.org/)を使用しています。  
  これらのライセンスにつきましては、該当ソースのヘッダ部分や、NVEnc_license.txtをご覧ください。

- ビルド方法については[こちら](./Build.ja.md)

### ソースの構成
Windows ... VCビルド  

文字コード: UTF-8-BOM  
改行: CRLF  
インデント: 空白x4  
