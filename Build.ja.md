
# NVEncのビルド方法

- [Windows](./Build.ja.md#windows)
- [Linux (Ubuntu 20.04 - 24.04)](./Build.ja.md#linux-ubuntu-2004---2404)
- [Linux (Ubuntu 18.04)](./Build.ja.md#linux-ubuntu-1804)
- [Linux (Fedora 33)](./Build.ja.md#linux-fedora-33)

## Windows

### 0. ビルドに必要なもの
ビルドには、下記のものが必要です。

- Visual Studio 2022
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
setx VAPOURSYNTH_SDK "C:\Program Files\VapourSynth\sdk"
```

また、ReleaseNVDLLで、NVEncNVSDKNGX.dllとNVEncNVOFFRUC.dllをビルドするには、RTX Video SDKとOptical Flow SDKをダウンロード後、適当なフォルダに展開し、環境変数 "RTX_VIDEO_SDK" と "OPTICAL_FLOW_SDK" を設定する必要があります。

```Batchfile
setx RTX_VIDEO_SDK "<...>\RTX_Video_SDK_v1.1.0"
setx OPTICAL_FLOW_SDK "<...>\Optical_Flow_SDK_5.0.7"
```

### 2. ソースのダウンロード

```Batchfile
git clone https://github.com/rigaya/NVEnc --recursive
cd NVEnc
curl -s -o ffmpeg_lgpl.7z -L https://github.com/rigaya/ffmpeg_dlls_for_hwenc/releases/download/20250830/ffmpeg_dlls_for_hwenc_20250830.7z
7z x -offmpeg_lgpl -y ffmpeg_lgpl.7z
```

### 3. NVEnc.auo / NVEncC のビルド

NVEnc.slnを開き、ビルドします。

ビルドしたいものに合わせて、構成を選択してください。

|              |Debug用構成|Release用構成|
|:---------------------|:------|:--------|
|NVEncC(64).exe | DebugStatic | RelStatic |
|NVEnc.auo (win32のみ) | Debug | Release |
|cufilters.auf (win32のみ) | DebugFilters | RelFilters |



## Linux (Ubuntu 20.04 - 24.04)

### 0. ビルドに必要なもの
- C++17 コンパイラ
- meson + ninja-build
- CUDA 10-12
- rust + cargo-c (libdoviビルド用)
- git
- ライブラリ群
  - ffmpeg 4.x-7.x ライブラリ群 (libavcodec*, libavformat*, libavfilter*, libavutil*, libswresample*, libavdevice*)
  - libass-dev
  - [オプション] AvisynthPlus
  - [オプション] VapourSynth

### 1. コンパイラ等のインストール

- ビルドツールのインストール

  ```Shell
  sudo apt install build-essential git meson ninja-build
  ```

- rust + cargo-cのインストール

  ```Shell
  sudo apt install libssl-dev curl pkgconf
  curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --profile minimal \
    && . ~/.cargo/env \
    && cargo install cargo-c
  ```

### 2. NVIDIA ドライバのインストール

### 3. CUDAのインストール
CUDAインストール用のdebファイルをダウンロードします。
```Shell
# Ubuntu 20.04用
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-keyring_1.1-1_all.deb
# Ubuntu 22.04用 (Ubuntu 24.04もとりあえず22.04用を使用)
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
```

CUDAをインストールします。
```Shell
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt update
sudo apt -y install --no-install-recommends cuda-toolkit
export CUDA_PATH=/usr/local/cuda
```

> [!NOTE]
> Ubuntu 24.04で上記でインストールに失敗する場合、下記のように必要なパッケージのみを選択してインストールすると回避できます。
> CUDA_VER_MAJORとCUDA_VER_MINORは適宜変更が必要かもしれません。
> ```
> CUDA_VER_MAJOR=12
> CUDA_VER_MINOR=4
> apt-get -y install cuda-drivers cuda-compiler-${CUDA_VER_MAJOR}-${CUDA_VER_MINOR} \
>   cuda-cudart-dev-${CUDA_VER_MAJOR}-${CUDA_VER_MINOR} cuda-driver-dev-${CUDA_VER_MAJOR}-${CUDA_VER_MINOR} \
>   cuda-nvrtc-dev-${CUDA_VER_MAJOR}-${CUDA_VER_MINOR} libcurand-dev-${CUDA_VER_MAJOR}-${CUDA_VER_MINOR} \
>   libnpp-dev-${CUDA_VER_MAJOR}-${CUDA_VER_MINOR} cuda-nvml-dev-${CUDA_VER_MAJOR}-${CUDA_VER_MINOR}
> ```


### 4. ビルドに必要なライブラリのインストール

ffmpegと関連ライブラリを導入します。
```Shell
sudo apt install \
  libavcodec-extra libavcodec-dev libavutil-dev libavformat-dev libswresample-dev libavfilter-dev libavdevice-dev \
  libass-dev libx11-dev libplacebo-dev
```

### 5. [オプション] AvisynthPlusのビルド

AvisynthPlusのインストールは必須ではありませんが、インストールしておくとavsを読み込めるようになります。

必要のない場合は 7. NVEncCのビルド に進んでください。

<details><summary>AvisynthPlusのビルドの詳細はこちら</summary>
#### 5.1 ビルドに必要なツールのインストール
```Shell
sudo apt install cmake
```

#### 5.2 AvisynthPlusのインストール
```Shell
git clone https://github.com/AviSynth/AviSynthPlus.git
cd AviSynthPlus
mkdir avisynth-build && cd avisynth-build 
cmake ../
make && sudo make install
cd ../..
```

</details>


### 6. [オプション] VapourSynthのビルド

VapourSynthのインストールは必須ではありませんが、インストールしておくとvpyを読み込めるようになります。

必要のない場合は 7. NVEncCのビルド に進んでください。

<details><summary>VapourSynthのビルドの詳細はこちら</summary>

#### 6.1 ビルドに必要なツールのインストール
```Shell
sudo apt install python3-pip cython3 autoconf automake libtool meson libzimg-dev
```

#### 6.2 VapourSynthのビルド
```Shell
git clone https://github.com/vapoursynth/vapoursynth.git
cd vapoursynth
./autogen.sh
./configure
make && sudo make install

# vapoursynthが自動的にロードされるようにする
# "python3.x" は環境に応じて変えてください。
sudo ln -s /usr/local/lib/python3.x/site-packages/vapoursynth.so /usr/lib/python3.x/lib-dynload/vapoursynth.so
sudo ldconfig
```

#### 6.3 VapourSynthの動作確認
エラーが出ずにバージョンが表示されればOK。
```Shell
LD_LIBRARY_PATH=/usr/local/lib vspipe --version
```

</details>

### 7. NVEncCのビルド
下記を実行します。
```Shell
git clone https://github.com/rigaya/NVEnc --recursive
cd NVEnc
meson setup ./build --buildtype=release
meson compile -C ./build
```

動作確認をします。
```Shell
./build/nvencc --check-hw
```

こんな感じでNVENCのサポートしているコーデックが表示されればOKです。
```
NVEnc (x64) 7.24 (r2526) by rigaya, Apr 28 2023 14:52:09 (gcc 12.2.0/Linux)
  [NVENC API v12.0, CUDA 12.1]
 reader: raw, y4m, avs, vpy, avsw, avhw [H.264/AVC, H.265/HEVC, MPEG2, VP8, VP9, VC-1, MPEG1, MPEG4, AV1]
#0: NVIDIA GeForce RTX 4080 (9728 cores, 2505 MHz)[2147483.64]
Avaliable Codec(s)
H.264/AVC
H.265/HEVC
AV1
```


## Linux (Ubuntu 18.04)

### 0. ビルドに必要なもの
- GPUドライバ 435.21 以上
- C++17 コンパイラ
- meson + ninja-build
- CUDA 10
- git
- ライブラリ群
  - ffmpeg 4.x系のライブラリ群 (libavcodec58, libavformat58, libavfilter7, libavutil56, libswresample3, libavdevice58)
  - libass9
  - [オプション] VapourSynth

### 1. コンパイラ等のインストール


- ビルドツールのインストール

  ```Shell
  sudo apt install git g++-8 meson ninja-build
  ```

- rust + cargo-cのインストール

  ```Shell
  sudo apt install libssl-dev curl
  curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --profile minimal \
    && . ~/.cargo/env \
    && cargo install cargo-c
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
  libavcodec-extra58 libavcodec-dev libavutil56 libavutil-dev libavformat58 libavformat-dev libavdevice58 libavdevice-dev \
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
git clone https://github.com/sekrit-twc/zimg.git --recursive
cd zimg
./autogen.sh
./configure
make && sudo make install
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
make && sudo make install

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
ninja && sudo ninja install
cd ../../../
```

</details>

### 5. NVEncCのビルド
下記を実行します。
```Shell
git clone https://github.com/rigaya/NVEnc --recursive
cd NVEnc
CXX=g++-8 meson setup ./build --buildtype=release
meson compile -C ./build
```

動作確認をします。
```Shell
./build/nvencc --check-hw
```

こんな感じでNVENCのサポートしているコーデックが表示されればOKです。 (AWS g3s.xlargeの例)
```
#0: Tesla M60 (2048 cores, 1177 MHz)[PCIe3x16][440.33]
Avaliable Codec(s)
H.264/AVC
H.265/HEVC
```

## Linux (Fedora 33)

### 0. ビルドに必要なもの
- C++17 コンパイラ
- meson + ninja-build
- CUDA 11
- git
- ライブラリ群
  - ffmpeg 4.x系のライブラリ群 (libavcodec58, libavformat58, libavfilter7, libavutil56, libswresample3, libavdevice58)
  - libass9
  - [オプション] AvisynthPlus
  - [オプション] VapourSynth

### 1. コンパイラ等のインストール

- コンパイラ等のインストール

  ```Shell
  sudo dnf install @development-tools meson ninja-build
  ```

- rust + cargo-cのインストール

  ```Shell
  sudo apt install libssl-dev curl
  curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --profile minimal \
    && . ~/.cargo/env \
    && cargo install cargo-c
  ```

### 2. CUDA・ドライバインストールの準備

```Shell
sudo dnf update
sudo dnf upgrade
sudo dnf clean all
sudo dnf install kernel-devel
sudo dnf install make pciutils acpid libglvnd-devel
sudo dnf install dkms
```

### 3. CUDAとNVIDIA ドライバのインストール
CUDAと付属ドライバをインストールします。
```Shell
wget https://developer.download.nvidia.com/compute/cuda/11.2.1/local_installers/cuda-repo-fedora33-11-2-local-11.2.1_460.32.03-1.x86_64.rpm
sudo rpm -ivh cuda-repo-fedora33-11-2-local-11.2.1_460.32.03-1.x86_64.rpm
sudo dnf clean all
sudo dnf install cuda
reboot
```
手元の環境では、cudaは/usr/local/cudaにインストールされました。

再起動後、正常に導入されたか確認します。下記のように出れば正常です。
```Shell
$ nvidia-smi
Sun Mar  7 14:27:45 2021
+-----------------------------------------------------------------------------+
| NVIDIA-SMI 460.32.03    Driver Version: 460.32.03    CUDA Version: 11.2     |
|-------------------------------+----------------------+----------------------+
| GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
|                               |                      |               MIG M. |
|===============================+======================+======================|
|   0  GeForce GTX 1080    Off  | 00000000:0D:00.0 Off |                  N/A |
|  0%   27C    P8     7W / 230W |     65MiB /  8111MiB |      0%      Default |
|                               |                      |                  N/A |
+-------------------------------+----------------------+----------------------+

+-----------------------------------------------------------------------------+
| Processes:                                                                  |
|  GPU   GI   CI        PID   Type   Process name                  GPU Memory |
|        ID   ID                                                   Usage      |
|=============================================================================|
|    0   N/A  N/A      1335      G   /usr/libexec/Xorg                  56MiB |
|    0   N/A  N/A      1476      G   /usr/bin/gnome-shell                6MiB |
+-----------------------------------------------------------------------------+
```

### 4. ビルドに必要なライブラリのインストール

ffmpegと関連ライブラリを導入します。
```Shell
sudo dnf install https://download1.rpmfusion.org/free/fedora/rpmfusion-free-release-$(rpm -E %fedora).noarch.rpm
sudo dnf install ffmpeg ffmpeg-devel libass libass-devel
```

### 5. [オプション] AvisynthPlusのビルド

AvisynthPlusのインストールは必須ではありませんが、インストールしておくとavsを読み込めるようになります。

必要のない場合は 7. NVEncCのビルド に進んでください。

<details><summary>AvisynthPlusのビルドの詳細はこちら</summary>
#### 5.1 ビルドに必要なツールのインストール
```Shell
sudo dnf install cmake
```

#### 5.2 AvisynthPlusのインストール
```Shell
git clone git://github.com/AviSynth/AviSynthPlus.git
cd AviSynthPlus
mkdir avisynth-build && cd avisynth-build 
cmake ../
make && sudo make install
cd ../..
```

#### 5.3 [おまけ] lsmashsourceのビルド
```Shell
# lsmashのビルド
git clone https://github.com/l-smash/l-smash.git
cd l-smash
./configure --enable-shared
make && sudo make install
cd ..

# lsmashsourceのビルド
git clone https://github.com/HolyWu/L-SMASH-Works.git
cd L-SMASH-Works
# libavcodec の要求バージョンをクリアするためバージョンを下げる
git checkout -b 20200531 refs/tags/20200531
cd AviSynth
meson build
cd build
ninja && sudo ninja install
cd ../../../
```

</details>


### 6. [オプション] VapourSynthのビルド
VapourSynthのインストールは必須ではありませんが、インストールしておくとvpyを読み込めるようになります。

必要のない場合は 7. NVEncCのビルド に進んでください。

<details><summary>VapourSynthのビルドの詳細はこちら</summary>

#### 6.1 ビルドに必要なツールのインストール
```Shell
sudo dnf install zimg zimg-devel meson autotools automake libtool python3-devel ImageMagick
```

#### 6.2 cythonのインストール
```Shell
sudo pip3 install Cython --install-option="--no-cython-compile"
```

#### 6.3 VapourSynthのビルド
```Shell
git clone https://github.com/vapoursynth/vapoursynth.git
cd vapoursynth
./autogen.sh
./configure
make && sudo make install

# vapoursynthが自動的にロードされるようにする
# "python3.x" は環境に応じて変えてください。これを書いた時点ではpython3.9でした
sudo ln -s /usr/local/lib/python3.x/site-packages/vapoursynth.so /usr/lib/python3.x/lib-dynload/vapoursynth.so
sudo ldconfig
```

#### 6.4 VapourSynthの動作確認
エラーが出ずにバージョンが表示されればOK。
```Shell
vspipe --version
```

#### 6.5 [おまけ] vslsmashsourceのビルド
```Shell
# lsmashのビルド (5.3 でビルドしていたら不要)
git clone https://github.com/l-smash/l-smash.git
cd l-smash
./configure --enable-shared
make && sudo make install
cd ..
 
# vslsmashsourceのビルド
git clone https://github.com/HolyWu/L-SMASH-Works.git
cd L-SMASH-Works
# libavcodec の要求バージョンをクリアするためバージョンを下げる
git checkout -b 20200531 refs/tags/20200531
cd VapourSynth
meson build
cd build
ninja && sudo ninja install
cd ../../../
```

</details>

### 7. NVEncCのビルド
下記を実行します。
```Shell
git clone https://github.com/rigaya/NVEnc --recursive
cd NVEnc
meson setup ./build --buildtype=release
meson compile -C ./build
```

動作確認をします。
```Shell
./build/nvencc --check-hw
```

こんな感じでNVENCのサポートしているコーデックが表示されればOKです。
```
#0: GeForce GTX 1080 (2560 cores, 1822 MHz)[PCIe3x16][460.32]
Avaliable Codec(s)
H.264/AVC
H.265/HEVC
```
