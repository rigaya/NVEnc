
# NVEncのビルド方法
by rigaya  

## Windows

### 0. 準備
ビルドには、下記のものが必要です。

- Visual Studio 2015 + CUDA 8.0 (x86)
- Visual Studio 2015/2017/2019 + CUDA 10.2 (x64)
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

さらにビルドに必要なCaption2Ass_PCRをcloneし、環境変数 "CAPTION2ASS_SRC" を設定します。

```Batchfile
git clone https://github.com/maki-rxrz/Caption2Ass_PCR <path-to-clone>
setx CAPTION2ASS_SRC Caption2Ass_PCR <path-to-clone>/src
```

### 1. ソースのダウンロード

```Batchfile
git clone https://github.com/rigaya/NVEnc --recursive
```

### 2. NVEnc.auo / NVEncC のビルド

NVEnc.slnを開き、ビルドします。

ビルドしたいものに合わせて、構成を選択してください。

|              |Debug用構成|Release用構成|
|:---------------------|:------|:--------|
|NVEncC(64).exe | DebugStatic | RelStatic |
|NVEnc.auo (win32のみ) | Debug | Release |
|cufilters.auf (win32のみ) | DebugFilters | RelFilters |

## Linux (Ubuntu 19.10)

### 0. ビルドに必要なもののインストール

```Shell
sudo apt install build-essential git yasm nasm
```

### 1. NVIDIA ドライバのインストール

導入可能なドライバの確認を行うため、下記を実行します。
```Shell
sudo add-apt-repository ppa:graphics-drivers/ppa
sudo apt update
ubuntu-drivers devices
```

すると、下記のような出力が出ます。
```Shell
== /sys/devices/pci0000:00/0000:00:03.1/0000:0d:00.0 ==
modalias : pci:v000010DEd00001B80sv000019DAsd00001426bc03sc00i00
vendor   : NVIDIA Corporation
model    : GP104 [GeForce GTX 1080]
driver   : nvidia-driver-435 - distro non-free
driver   : nvidia-driver-440 - third-party free recommended
driver   : nvidia-driver-390 - third-party free
driver   : xserver-xorg-video-nouveau - distro free builtin
```

最新のものをインストールします。
```Shell
sudo apt install nvidia-driver-440
sudo reboot
```

再起動後、正常に導入されたか確認します。下記のように出れば正常です。
```Shell
rigaya@rigaya6-linux:~$ nvidia-smi
Fri Apr 24 22:39:10 2020
+-----------------------------------------------------------------------------+
| NVIDIA-SMI 440.82       Driver Version: 440.82       CUDA Version: 10.2     |
|-------------------------------+----------------------+----------------------+
| GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
|===============================+======================+======================|
|   0  GeForce GTX 1080    Off  | 00000000:0D:00.0  On |                  N/A |
|  0%   31C    P8     9W / 230W |     89MiB /  8111MiB |      0%      Default |
+-------------------------------+----------------------+----------------------+

+-----------------------------------------------------------------------------+
| Processes:                                                       GPU Memory |
|  GPU       PID   Type   Process name                             Usage      |
|=============================================================================|
|    0      1345      G   /usr/lib/xorg/Xorg                            39MiB |
|    0      1786      G   /usr/bin/gnome-shell                          46MiB |
+-----------------------------------------------------------------------------+
```

### 2. CUDAのインストール
```Shell
sudo apt install nvidia-cuda-toolkit
```

### 3. ビルドに必要なライブラリのインストール

ffmpegと関連ライブラリを導入します。
```Shell
sudo apt install ffmpeg \
  libavcodec-extra libavcodec-dev libavutil-dev libavformat-dev libswresample-dev libavfilter-dev \
  libass9 libass-dev
```

### 4. [オプション] VapourSynthのビルド
VapourSynthのインストールは必須ではありませんが、インストールしておくとvpyを読み込めるようになります。

必要のない場合は 5. NVEncCのビルド に進んでください。

#### 4.1 ビルドに必要なツールのインストール
```Shell
sudo apt install python3-pip autoconf automake libtool meson
```

#### 4.2 zimgのインストール
```Shell
git clone https://github.com/sekrit-twc/zimg.git
cd zimg
./autogen.sh
./configure
sudo make install -j16
cd ..
```

#### 4.3 cythonのインストール
```Shell
sudo pip3 install Cython
```

#### 4.4 VapourSynthのビルド
```Shell
git clone https://github.com/vapoursynth/vapoursynth.git
cd vapoursynth
./autogen.sh
./configure
make -j16
sudo make install

# vapoursynthが自動的にロードされるようにする
# "python3.x" は環境に応じて変えてください。これを書いた時点ではpython3.7でした
sudo ln -s /usr/local/lib/python3.x/site-packages/vapoursynth.so /usr/lib/python3.x/lib-dynload/vapoursynth.so
sudo ldconfig
```

#### 4.5 VapourSynthの動作確認
エラーが出ずにバージョンが表示されればOK。
```Shell
vspipe --version
```

#### 4.6 [おまけ] vslsmashsourceのビルド
```Shell
# lsmashのビルド
git clone https://github.com/l-smash/l-smash.git
cd l-smash
./configure --enable-shared
sudo make install -j16
cd ..
 
# vslsmashsourceのビルド
git clone https://github.com/HolyWu/L-SMASH-Works.git
cd L-SMASH-Works/VapourSynth
meson build
cd build
sudo ninja install
cd ../../../
```

### 5. NVEncCのビルド
下記を実行します。
```Shell
git clone https://github.com/rigaya/NVEnc --recursive
cd NVEnc
./configure
make -j16
```

動作確認をします。正常にGPUが認識されていればOKです。
```Shell
./nvencc --check-hw
```

こんな感じでNVENCのサポートしているコーデックが表示されればOKです。
```
#0: GeForce GTX 1080 (2560 cores, 1822 MHz)[PCIe3x16][440.82]
Avaliable Codec(s)
H.264/AVC
H.265/HEVC
```