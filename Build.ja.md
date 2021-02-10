
# NVEncのビルド方法

- [Windows](./Build.ja.md#windows)
- [Linux (Ubuntu 19.10)](./Build.ja.md#linux-ubuntu-1910)
- [Linux (Ubuntu 18.04)](./Build.ja.md#linux-ubuntu-1804)

## Windows

### 0. ビルドに必要なもの
ビルドには、下記のものが必要です。

- Visual Studio 2019
- CUDA 10.1 以降 (x64)
- CUDA 11.0 以降 (x86)
- [Avisynth](https://github.com/AviSynth/AviSynthPlus) SDK
- [VapourSynth](http://www.vapoursynth.com/) SDK

### 1. 環境準備

Avisynth+とVapourSynthは、SDKがインストールされるよう設定してインストールします。

Avisynth+ SDKの"avisynth_c.h"とVapourSynth SDKの"VapourSynth.h", "VSScript.h"がVisual Studioのincludeパスに含まれるよう設定します。

includeパスは環境変数 "AVISYNTH_SDK" / "VAPOURSYNTH_SDK" で渡すことができます。

Avisynth+ / VapourSynthインストーラのデフォルトの場所にインストールした場合、下記のように設定することになります。
```Batchfile
setx AVISYNTH_SDK "C:\Program Files (x86)\AviSynth+\FilterSDK"
setx VAPOURSYNTH_SDK "C:\Program Files (x86)\VapourSynth\sdk"
```

さらにビルドに必要な[Caption2Ass_PCR](https://github.com/maki-rxrz/Caption2Ass_PCR)をcloneし、環境変数 "CAPTION2ASS_SRC" を設定します。

```Batchfile
git clone https://github.com/maki-rxrz/Caption2Ass_PCR <path-to-clone>
setx CAPTION2ASS_SRC "<path-to-clone>/src"
```

### 2. ソースのダウンロード

```Batchfile
git clone https://github.com/rigaya/NVEnc --recursive
```

### 3. NVEnc.auo / NVEncC のビルド

NVEnc.slnを開き、ビルドします。

ビルドしたいものに合わせて、構成を選択してください。

|              |Debug用構成|Release用構成|
|:---------------------|:------|:--------|
|NVEncC(64).exe | DebugStatic | RelStatic |
|NVEnc.auo (win32のみ) | Debug | Release |
|cufilters.auf (win32のみ) | DebugFilters | RelFilters |

## Linux (Ubuntu 19.10)

### 0. ビルドに必要なもの
- GPUドライバ 435.21 以上
- C++17 コンパイラ
- CUDA 10
- git
- ライブラリ群
  - ffmpeg 4.x系のライブラリ群 (libavcodec58, libavformat58, libavfilter7, libavutil56, libswresample3)
  - libass9
  - [オプション] VapourSynth

### 1. コンパイラ等のインストール

```Shell
sudo apt install build-essential git
```

### 2. NVIDIA ドライバのインストール

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
$ nvidia-smi
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

### 3. CUDAのインストール
```Shell
sudo apt install nvidia-cuda-toolkit
```

### 4. ビルドに必要なライブラリのインストール

ffmpegと関連ライブラリを導入します。
```Shell
sudo apt install \
  libavcodec-extra libavcodec-dev libavutil-dev libavformat-dev libswresample-dev libavfilter-dev \
  libass9 libass-dev
```
### 5. [オプション] VapourSynthのビルド
VapourSynthのインストールは必須ではありませんが、インストールしておくとvpyを読み込めるようになります。

必要のない場合は 6. NVEncCのビルド に進んでください。

<details><summary>VapourSynthのビルドの詳細はこちら</summary>

#### 5.1 ビルドに必要なツールのインストール
```Shell
sudo apt install python3-pip autoconf automake libtool meson
```

#### 5.2 zimgのインストール
```Shell
git clone https://github.com/sekrit-twc/zimg.git
cd zimg
./autogen.sh
./configure
sudo make install -j16
cd ..
```

#### 5.3 cythonのインストール
```Shell
sudo pip3 install Cython
```

#### 5.4 VapourSynthのビルド
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

#### 5.5 VapourSynthの動作確認
エラーが出ずにバージョンが表示されればOK。
```Shell
vspipe --version
```

#### 5.6 [おまけ] vslsmashsourceのビルド
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

</details>

### 6. NVEncCのビルド
下記を実行します。
```Shell
git clone https://github.com/rigaya/NVEnc --recursive
cd NVEnc
./configure
make -j16
```

動作確認をします。
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


## Linux (Ubuntu 18.04)

### 0. ビルドに必要なもの
- GPUドライバ 435.21 以上
- C++17 コンパイラ
- CUDA 10
- git
- ライブラリ群
  - ffmpeg 4.x系のライブラリ群 (libavcodec58, libavformat58, libavfilter7, libavutil56, libswresample3)
  - libass9
  - [オプション] VapourSynth

### 1. コンパイラ等のインストール

```Shell
sudo apt install git
```

### 2. NVIDIA ドライバ + CUDA のインストール

```Shell
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/cuda-ubuntu1804.pin
sudo mv cuda-ubuntu1804.pin /etc/apt/preferences.d/cuda-repository-pin-600
wget http://developer.download.nvidia.com/compute/cuda/10.2/Prod/local_installers/cuda-repo-ubuntu1804-10-2-local-10.2.89-440.33.01_1.0-1_amd64.deb
sudo dpkg -i cuda-repo-ubuntu1804-10-2-local-10.2.89-440.33.01_1.0-1_amd64.deb
sudo apt-key add /var/cuda-repo-10-2-local-10.2.89-440.33.01/7fa2af80.pub
sudo apt-get update
sudo apt-get -y install cuda
export CUDA_PATH=/usr/local/cuda
```

再起動後、正常に導入されたか確認します。下記のように出れば正常です。 (下記はAWS g3s.xlargeでテストしたもの)
```Shell
$ nvidia-smi
+-----------------------------------------------------------------------------+
| NVIDIA-SMI 440.33.01    Driver Version: 440.33.01    CUDA Version: 10.2     |
|-------------------------------+----------------------+----------------------+
| GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
|===============================+======================+======================|
|   0  Tesla M60           On   | 00000000:00:1E.0 Off |                    0 |
| N/A   28C    P8    15W / 150W |      0MiB /  7618MiB |      0%      Default |
+-------------------------------+----------------------+----------------------+

+-----------------------------------------------------------------------------+
| Processes:                                                       GPU Memory |
|  GPU       PID   Type   Process name                             Usage      |
|=============================================================================|
|  No running processes found                                                 |
+-----------------------------------------------------------------------------+
```

### 3. ビルドに必要なライブラリのインストール

Ubuntu 18.04の標準ではffmpeg 3.x系が導入されてしまうため、下記のように明示的にffmpeg 4.x系のライブラリを導入します。

```Shell
sudo add-apt-repository ppa:jonathonf/ffmpeg-4
sudo apt update
sudo apt install ffmpeg \
  libavcodec-extra58 libavcodec-dev libavutil56 libavutil-dev libavformat58 libavformat-dev \
  libswresample3 libswresample-dev libavfilter-extra7 libavfilter-dev libass9 libass-dev
```
### 4. [オプション] VapourSynthのビルド
VapourSynthのインストールは必須ではありませんが、インストールしておくとvpyを読み込めるようになります。

必要のない場合は 5. NVEncCのビルド に進んでください。

<details><summary>VapourSynthのビルドの詳細はこちら</summary>

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
sudo make install -j4
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
make -j4
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
sudo make install -j4
cd ..
 
# vslsmashsourceのビルド
git clone https://github.com/HolyWu/L-SMASH-Works.git
cd L-SMASH-Works/VapourSynth
meson build
cd build
sudo ninja install
cd ../../../
```

</details>

### 5. NVEncCのビルド
下記を実行します。
```Shell
git clone https://github.com/rigaya/NVEnc --recursive
cd NVEnc
./configure
make -j4
```

動作確認をします。
```Shell
./nvencc --check-hw
```

こんな感じでNVENCのサポートしているコーデックが表示されればOKです。 (AWS g3s.xlargeの例)
```
#0: Tesla M60 (2048 cores, 1177 MHz)[PCIe3x16][440.33]
Avaliable Codec(s)
H.264/AVC
H.265/HEVC
```