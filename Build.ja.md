
# NVEncのビルド方法
by rigaya  

## 0. 準備
ビルドには、下記のものが必要です。

- Visual Studio 2015
- CUDA 8.0
- yasm
- Avisynth+ SDK
- VapourSynth SDK

yasmはパスに追加しておきます。

Avisynth+とVapourSynthは、SDKがインストールされるよう設定してインストールします。

Avisynth+ SDKの"avisynth_c.h"とVapourSynth SDKの"VapourSynth.h", "VSScript.h"がVisual Studioのincludeパスに含まれるよう設定します。

includeパスは環境変数 "AVISYNTH_SDK" / "VAPOURSYNTH_SDK" で渡すことができます。

Avisynth+ / VapourSynthインストーラのデフォルトの場所にインストールした場合、下記のように設定することになります。
```Batchfile
setx AVISYNTH_SDK "C:\Program Files (x86)\AviSynth+\FilterSDK"
setx VAPOURSYNTH_SDK "C:\Program Files (x86)\VapourSynth\sdk"
```

## 1. ソースのダウンロード

```Batchfile
git clone https://github.com/rigaya/NVEnc --recursive
```

## 2. NVEnc.auo / NVEncC のビルド

NVEnc.slnを開き、ビルドします。

ビルドしたいものに合わせて、構成を選択してください。

|              |Debug用構成|Release用構成|
|:---------------------|:------|:--------|
|NVEnc.auo (win32のみ) | Debug | Release |
|NVEncC(64).exe | DebugStatic | RelStatic |
