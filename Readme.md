# NVEnc
by rigaya

NVEnc.auo … NVIDIAのNVEncを使用してエンコードを行うAviutlの出力プラグインです。

NVEncC.exe … 上記のコマンドライン版です。

## 配布場所 & 更新履歴
[rigayaの日記兼メモ帳＞＞](http://rigaya34589.blog135.fc2.com/blog-category-17.html)

## 基本動作環境
Windows 7,8,8.1,10 (x86/x64)  
Aviutl 0.99g4 以降  
NVEncが載ったハードウェア  
  NVIDIA製 GPU GeForce Kepler世代以降 (GT/GTX 6xx 以降)  
  ※GT 63x, 62x等はFermi世代のリネームであるため非対応なものがあります。  
NVEnc 0.xx NVIDIA グラフィックドライバ 334.89以降  
NVEnc 1.xx NVIDIA グラフィックドライバ 347.09以降  
NVEnc 2.xx NVIDIA グラフィックドライバ 358   以降  

## NVEnc 使用にあたっての注意事項
無保証です。自己責任で使用してください。  
NVEncを使用したことによる、いかなる損害・トラブルについても責任を負いません。  

### NVEncのソースコードについて
・MITライセンスです。  
・本プログラムは、NVIDA CUDA Samplesをベースに作成されており、サンプルコードを含みます。  
  This software contains source code provided by NVIDIA Corporation.  

## 使用出来る主な機能
### NVEnc/NVEncC共通
・NVENCを使用したエンコード  
   - H.264/AVC  
   - H.265/HEVC  
・NVENCの各エンコードモード  
   - CQP       固定量子化量  
   - CBR       固定ビットレート  
   - VBR       可変ビットレート  
   - VBR2      可変ビットレート2  
・インタレ保持エンコード (PAFF方式)  
・SAR比指定  
・colormatrix等の指定  
・H.264 Level / Profileの指定  
・最大ビットレート等の指定  
・最大GOP長の指定  

### NVEnc
・音声エンコード  
・音声及びチャプターとのmux機能  
・自動フィールドシフト対応  

### NVEncC
・cuvidデコードに対応。  
  - MPEG1  
  - MPEG2  
  - H.264/AVC  
・avs, vpy, y4m, rawなど各種形式に対応  
・GPUを使用した高速フィルタリング  
   - リサイズ  
   - インタレ解除 (normal / bob / it)  

## ソースコードについて
無保証です。  
ソースコードを使用したことによるいかなる損害・トラブルについてrigayaは責任を負いません。  
以上に了解して頂ける場合、ソースコードの使用、複製、改変、再頒布を行って頂いて構いません。

### ソースの構成
Windows ... VCビルド  

文字コード: UTF-8-BOM  
改行: CRLF  
インデント: 空白x4  
コード例:  
```
    for (int i = 0; i < LOOP_MAX; i++) {
        //コメント
        if (i > 0) {
            ptr[i] = ptr[i - 1] + i + 2;
        }
    }

```