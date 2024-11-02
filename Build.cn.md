
# 如何建立NVEnc
by rigaya  

- [Windows](./Build.cn.md#windows)
- [Linux (Ubuntu 22.04 - 23.04)](./Build.cn.md#linux-ubuntu-2204---2304)
- [Linux (Ubuntu 20.04)](./Build.cn.md#linux-ubuntu-2004)
- [Linux (Ubuntu 18.04)](./Build.cn.md#linux-ubuntu-1804)
- [Linux (Fedora 33)](./Build.cn.md#linux-fedora-33)

## Windows

### 0. 环境需求
要构建NVEnc，需要以下组件。

- Visual Studio 2019
- CUDA 10.1 或更高版本 (x64)
- CUDA 11.0 或更高版本 (x86)
- [Avisynth](https://github.com/AviSynth/AviSynthPlus) SDK
- [VapourSynth](http://www.vapoursynth.com/) SDK

### 1. 安装构建工具

使用sdk安装Avisynth+和VapourSynth。

然后，Avisynth+ SDK 的 "avisynth_c.h" 和VapourSynth SDK 的 "VapourSynth.h" 应被添加到 Visual Studio 的包含路径中。

这些包含路径可以通过环境变量"AVISYNTH_SDK"和"VAPOURSYNTH_SDK"来传递。

使用默认安装，环境变量可以设置如下。
```Batchfile
setx AVISYNTH_SDK "C:\Program Files (x86)\AviSynth+\FilterSDK"
setx VAPOURSYNTH_SDK "C:\Program Files (x86)\VapourSynth\sdk"
```

## 2. 下载源代码

```Batchfile
git clone https://github.com/rigaya/NVEnc --recursive
cd NVEnc
curl -s -o ffmpeg_lgpl.7z -L https://github.com/rigaya/ffmpeg_dlls_for_hwenc/releases/download/20241102/ffmpeg_dlls_for_hwenc_20241102.7z
7z x -offmpeg_lgpl -y ffmpeg_lgpl.7z
```

## 3. 构建 NVEncC.exe / NVEnc.auo

最后，打开 NVEnc.sln，然后开始使用 Visual Studio 构建 NVEnc。

|   | For Debug build | For Release build |
|:---------------------|:------|:--------|
|NVEncC(64).exe | DebugStatic | RelStatic |
|NVEnc.auo (win32 only) | Debug | Release |
|cufilters.auf (win32 only) | DebugFilters | RelFilters |

## Linux (Ubuntu 22.04 - 23.04)

### 0. 环境需求

- GPU 驱动版本 435.21 或更高
- 支持 C++17 标准的编译器
- CUDA 10/11
- git
- 所需的库:
  - ffmpeg 4.x/5.x libs (libavcodec*, libavformat*, libavfilter*, libavutil*, libswresample*, libavdevice*)
  - libass9
  - [可选] AvisynthPlus
  - [可选] VapourSynth

### 1. 安装构建工具

```Shell
sudo apt install build-essential git
sudo apt install libssl-dev curl
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --profile minimal \
  && . ~/.cargo/env \
  && cargo install cargo-c
```

### 2. 安装 NVIDIA 驱动

### 3. 安装 CUDA
```Shell
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt-get update
sudo apt-get -y install --no-install-recommends cuda-toolkit
export CUDA_PATH=/usr/local/cuda
```

### 4. 安装所需的库

安装ffmpeg和其他需要的库
```Shell
sudo apt install ffmpeg \
  libavcodec-extra libavcodec-dev libavutil-dev libavformat-dev libswresample-dev libavfilter-dev libavdevice-dev \
  libass-dev libx11-dev libplacebo-dev
```

### 5. [可选] 安装 AvisynthPlus 
只有在需要支持 AvisynthPlus(avs) reader 时才需要安装AvisynthPlus

如果不需要avs reader，请跳转到 [7. 构建 NVEncC]

<details><summary>如何构建 AvisynthPlus</summary>

#### 5.1 安装 AvisynthPlus 的构建工具
```Shell
sudo apt install cmake
```

#### 5.2 安装 AvisynthPlus
```Shell
git clone https://github.com/AviSynth/AviSynthPlus.git
cd AviSynthPlus
mkdir avisynth-build && cd avisynth-build 
cmake ../
make && sudo make install
cd ../..
```
</details>

### 6. [可选] 安装 VapourSynth
只有在需要支持 VapourSynth(vpy) reader 时才需要安装 VapourSynth 

如果不需要 vpy reader，请跳转到 [7. 构建 NVEncC]

<details><summary>如何构建 VapourSynth</summary>

#### 6.1 安装 VapourSynth 的构建工具
```Shell
sudo apt install python3-pip cython3 autoconf automake libtool meson libzimg-dev
```

#### 6.2 安装 VapourSynth
```Shell
git clone https://github.com/vapoursynth/vapoursynth.git
cd vapoursynth
./autogen.sh
./configure
make && sudo make install

# Make sure vapoursynth could be imported from python
# Change "python3.x" depending on your environment
sudo ln -s /usr/local/lib/python3.x/site-packages/vapoursynth.so /usr/lib/python3.x/lib-dynload/vapoursynth.so
sudo ldconfig
```

#### 6.3 检查 VapourSynth 是否正确安装
确认得到了版本号且没有产生报错
```Shell
LD_LIBRARY_PATH=/usr/local/lib vspipe --version
```

</details>

### 7. 构建 NVEncC
```Shell
git clone https://github.com/rigaya/NVEnc --recursive
cd NVEnc
./configure
make
```
检查是否正常执行
```Shell
./nvencc --check-hw
```

应可以得到当前NVENC支持的编解码器列表
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


## Linux (Ubuntu 20.04)

### 0. 环境需求

- GPU 驱动版本 435.21 或更高
- 支持 C++17 标准的编译器
- CUDA 10/11
- git
- 所需的库:
  - ffmpeg 4.x libs (libavcodec58, libavformat58, libavfilter7, libavutil56, libswresample3, libavdevice58)
  - libass9
  - [可选] AvisynthPlus
  - [可选] VapourSynth

### 1. 安装构建工具

```Shell
sudo apt install build-essential git
sudo apt install libssl-dev curl
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --profile minimal \
  && . ~/.cargo/env \
  && cargo install cargo-c
```

### 2. 安装 NVIDIA 驱动

获取可安装的驱动程序版本
```Shell
ubuntu-drivers devices
```

应可以获得以下输出
```Shell
== /sys/devices/pci0000:00/0000:00:03.1/0000:0d:00.0 ==
modalias : pci:v000010DEd00001B80sv000019DAsd00001426bc03sc00i00
vendor   : NVIDIA Corporation
model    : GP104 [GeForce GTX 1080]
driver   : nvidia-driver-390 - distro non-free
driver   : nvidia-driver-460 - distro non-free recommended
driver   : nvidia-driver-450-server - distro non-free
driver   : nvidia-driver-418-server - distro non-free
driver   : nvidia-driver-450 - distro non-free
driver   : xserver-xorg-video-nouveau - distro free builtin
```

选择安装最新版本的驱动
```Shell
sudo apt install nvidia-driver-460
sudo reboot
```

重启后，检查驱动是否正确安装
```Shell
rigaya@rigaya6-linux:~$ nvidia-smi
Sun Feb 21 13:49:17 2021
+-----------------------------------------------------------------------------+
| NVIDIA-SMI 460.32.03    Driver Version: 460.32.03    CUDA Version: 11.2     |
|-------------------------------+----------------------+----------------------+
| GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
|                               |                      |               MIG M. |
|===============================+======================+======================|
|   0  GeForce GTX 1080    Off  | 00000000:0D:00.0  On |                  N/A |
|  0%   33C    P8     8W / 230W |     46MiB /  8111MiB |      0%      Default |
|                               |                      |                  N/A |
+-------------------------------+----------------------+----------------------+

+-----------------------------------------------------------------------------+
| Processes:                                                                  |
|  GPU   GI   CI        PID   Type   Process name                  GPU Memory |
|        ID   ID                                                   Usage      |
|=============================================================================|
|    0   N/A  N/A      1076      G   /usr/lib/xorg/Xorg                 36MiB |
|    0   N/A  N/A      1274      G   /usr/bin/gnome-shell                7MiB |
+-----------------------------------------------------------------------------+
```

### 3. 安装 CUDA
```Shell
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt-get update
sudo apt-get -y install --no-install-recommends cuda-toolkit
export CUDA_PATH=/usr/local/cuda
```

### 4. 安装所需的库

安装ffmpeg和其他需要的库
```Shell
sudo apt install ffmpeg \
  libavcodec-extra libavcodec-dev libavutil-dev libavformat-dev libswresample-dev libavfilter-dev libavdevice-dev \
  libass9 libass-dev
```

### 5. [可选] 安装 AvisynthPlus 
只有在需要支持 AvisynthPlus(avs) reader 时才需要安装AvisynthPlus  

如果不需要avs reader，请跳转到 [7. 构建 NVEncC]

<details><summary>如何构建 AvisynthPlus</summary>

#### 5.1 安装 AvisynthPlus 的构建工具
```Shell
sudo apt install cmake
```

#### 5.2 安装 AvisynthPlus
```Shell
git clone https://github.com/AviSynth/AviSynthPlus.git
cd AviSynthPlus
mkdir avisynth-build && cd avisynth-build 
cmake ../
make && sudo make install
cd ../..
```

#### 5.3 [可选] 安装 lsmashsource
```Shell
# Install lsmash
git clone https://github.com/l-smash/l-smash.git
cd l-smash
./configure --enable-shared
make && sudo make install
cd ..
 
# Install vslsmashsource
git clone https://github.com/HolyWu/L-SMASH-Works.git
cd L-SMASH-Works
# Use older version to meet libavcodec lib version requirements
git checkout -b 20200531 refs/tags/20200531
cd VapourSynth
meson build
cd build
ninja && sudo ninja install
cd ../../../
```
</details>

### 6. [可选] 安装 VapourSynth
只有在需要支持 VapourSynth(vpy) reader 时才需要安装 VapourSynth 

如果不需要 vpy reader，请跳转到 [7. 构建 NVEncC]

<details><summary>如何构建 VapourSynth</summary>

#### 6.1 安装 VapourSynth 的构建工具
```Shell
sudo apt install python3-pip autoconf automake libtool meson
```

#### 6.2 安装 zimg
```Shell
git clone https://github.com/sekrit-twc/zimg.git --recursive
cd zimg
./autogen.sh
./configure
make && sudo make install
cd ..
```

#### 6.3 安装 cython
```Shell
sudo pip3 install Cython
```

#### 6.4 安装 VapourSynth
```Shell
git clone https://github.com/vapoursynth/vapoursynth.git
cd vapoursynth
./autogen.sh
./configure
make && sudo make install

# Make sure vapoursynth could be imported from python
# Change "python3.x" depending on your environment
sudo ln -s /usr/local/lib/python3.x/site-packages/vapoursynth.so /usr/lib/python3.x/lib-dynload/vapoursynth.so
sudo ldconfig
```

#### 6.5 检查 VapourSynth 是否正确安装
确认得到了版本号且没有产生报错
```Shell
LD_LIBRARY_PATH=/usr/local/lib vspipe --version
```

#### 6.6 [可选] 构建 vslsmashsource
```Shell
# Install lsmash
git clone https://github.com/l-smash/l-smash.git
cd l-smash
./configure --enable-shared
make && sudo make install
cd ..
 
# Install vslsmashsource
git clone https://github.com/HolyWu/L-SMASH-Works.git
cd L-SMASH-Works
# Use older version to meet libavcodec lib version requirements
git checkout -b 20200531 refs/tags/20200531
cd VapourSynth
meson build
cd build
ninja && sudo ninja install
cd ../../../
```

</details>

### 7. 构建 NVEncC
```Shell
git clone https://github.com/rigaya/NVEnc --recursive
cd NVEnc
./configure
make
```
检查是否正常执行
```Shell
./nvencc --check-hw
```

应可以得到当前NVENC支持的编解码器列表
```
#0: GeForce GTX 1080 (2560 cores, 1822 MHz)[PCIe3x16][460.32]
Avaliable Codec(s)
H.264/AVC
H.265/HEVC
```


## Linux (Ubuntu 18.04)

### 0. 所需环境

- GPU 驱动版本 435.21 或更高
- 支持 C++17 标准的编译器
- CUDA 10
- git
- 所需的库:
  - ffmpeg 4.x libs (libavcodec58, libavformat58, libavfilter7, libavutil56, libswresample3, libavdevice58)
  - libass9
  - [可选] VapourSynth

### 1. 安装构建工具

```Shell
sudo apt install git g++-8
sudo apt install libssl-dev curl
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --profile minimal \
  && . ~/.cargo/env \
  && cargo install cargo-c
```

### 2. 安装 NVIDIA 驱动和 CUDA 10.2
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

重启后，检查驱动是否已正确安装(在AWS g3s.xlarge中测试)

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

### 3. 安装所需的库

安装 ffmpeg 4.x 库
```Shell
sudo add-apt-repository ppa:jonathonf/ffmpeg-4
sudo apt update
sudo apt install ffmpeg \
  libavcodec-extra58 libavcodec-dev libavutil56 libavutil-dev libavformat58 libavformat-dev libavdevice58 libavdevice-dev \
  libswresample3 libswresample-dev libavfilter-extra7 libavfilter-dev libass9 libass-dev
```

### 4. [可选] 安装 VapourSynth
只有在需要支持 VapourSynth(vpy) reader 时才需要安装 VapourSynth 

如果不需要 vpy reader，请跳转到 [6. 构建 NVEncC]

<details><summary>如何构建 VapourSynth</summary>

#### 4.1 安装 VapourSynth 的构建工具
```Shell
sudo apt install python3-pip autoconf automake libtool meson
```

#### 4.2 安装 zimg
```Shell
git clone https://github.com/sekrit-twc/zimg.git --recursive
cd zimg
./autogen.sh
./configure
make && sudo make install
cd ..
```

#### 4.3 安装 cython
```Shell
sudo pip3 install Cython
```

#### 4.4 安装 VapourSynth
```Shell
git clone https://github.com/vapoursynth/vapoursynth.git
cd vapoursynth
./autogen.sh
./configure
make && sudo make install

# Make sure vapoursynth could be imported from python
# Change "python3.x" depending on your environment
sudo ln -s /usr/local/lib/python3.x/site-packages/vapoursynth.so /usr/lib/python3.x/lib-dynload/vapoursynth.so
sudo ldconfig
```

#### 4.5 检查 VapourSynth 是否已正确安装
Make sure you get version number without errors.
```Shell
vspipe --version
```

#### 4.6 [可选] 构建 vslsmashsource
```Shell
# Install lsmash
git clone https://github.com/l-smash/l-smash.git
cd l-smash
./configure --enable-shared
make && sudo make install
cd ..
 
# Install vslsmashsource
git clone https://github.com/HolyWu/L-SMASH-Works.git
cd L-SMASH-Works/VapourSynth
meson build
cd build
ninja && sudo ninja install
cd ../../../
```

</details>

### 5. 构建 NVEncC
```Shell
git clone https://github.com/rigaya/NVEnc --recursive
cd NVEnc
./configure --cxx=g++-8
make
```
检查是否正确执行
```Shell
./nvencc --check-hw
```

应可以得到当前NVENC支持的编解码器列表 (在AWS g3s.xlarge进行测试)
```
#0: Tesla M60 (2048 cores, 1177 MHz)[PCIe3x16][440.33]
Avaliable Codec(s)
H.264/AVC
H.265/HEVC
```


## Linux (Fedora 33)

### 0. 所需环境
- 支持 C++17 标准的编译器
- CUDA 11
- git
- 所需的库:
  - ffmpeg 4.x libs (libavcodec58, libavformat58, libavfilter7, libavutil56, libswresample3, libavdevice58)
  - libass9
  - [可选] AvisynthPlus
  - [可选] VapourSynth

### 1. 安装构建工具

```Shell
sudo dnf install @development-tools
sudo apt install libssl-dev curl
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --profile minimal \
  && . ~/.cargo/env \
  && cargo install cargo-c
```

### 2. 准备安装 CUDA 和 NVIDIA driver

```Shell
sudo dnf update
sudo dnf upgrade
sudo dnf clean all
sudo dnf install kernel-devel
sudo dnf install make pciutils acpid libglvnd-devel
sudo dnf install dkms
```

### 3. 安装 CUDA 和 NVIDIA driver
安装 CUDA 和 包括在 CUDA package 中的 NVIDIA driver
```Shell
wget https://developer.download.nvidia.com/compute/cuda/11.2.1/local_installers/cuda-repo-fedora33-11-2-local-11.2.1_460.32.03-1.x86_64.rpm
sudo rpm -ivh cuda-repo-fedora33-11-2-local-11.2.1_460.32.03-1.x86_64.rpm
sudo dnf clean all
sudo dnf install cuda
reboot
```
CUDA 会安装在 /usr/local/cuda.

重启后，检查驱动是否已正确安装
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

### 4. 安装所需的库

安装ffmpeg和其他所需的库
```Shell
sudo dnf install https://download1.rpmfusion.org/free/fedora/rpmfusion-free-release-$(rpm -E %fedora).noarch.rpm
sudo dnf install ffmpeg ffmpeg-devel libass libass-devel
```

### 5. [可选] 安装 AvisynthPlus
只有在需要支持 AvisynthPlus(avs) reader 时才需要安装AvisynthPlus

如果不需要avs reader，请跳转到 [7. 构建 NVEncC]

<details><summary>如何构建 AvisynthPlus</summary>

#### 5.1 安装 AvisynthPlus 的构建工具
```Shell
sudo dnf install cmake
```

#### 5.2 安装 AvisynthPlus
```Shell
git clone git://github.com/AviSynth/AviSynthPlus.git
cd AviSynthPlus
mkdir avisynth-build && cd avisynth-build 
cmake ../
make && sudo make install
cd ../..
```

#### 5.3 [可选] 安装 lsmashsource
```Shell
# Build lsmash
git clone https://github.com/l-smash/l-smash.git
cd l-smash
./configure --enable-shared
make && sudo make install
cd ..

# Build lsmashsource
git clone https://github.com/HolyWu/L-SMASH-Works.git
cd L-SMASH-Works
git checkout -b 20200531 refs/tags/20200531
cd AviSynth
meson build
cd build
ninja && sudo ninja install
cd ../../../
```

</details>


### 6. [可选] 安装 VapourSynth
只有在需要支持 VapourSynth(vpy) reader 时才需要安装 VapourSynth 

如果不需要 vpy reader，请跳转到 [7. 构建 NVEncC]

<details><summary>如何构建 VapourSynth</summary>

#### 6.1 安装 VapourSynth 的构建工具
```Shell
sudo dnf install zimg zimg-devel meson autotools automake libtool python3-devel ImageMagick
```

#### 6.2 安装 VapourSynth
```Shell
sudo pip3 install Cython --install-option="--no-cython-compile"
```

#### 6.3 构建 VapourSynth
```Shell
git clone https://github.com/vapoursynth/vapoursynth.git
cd vapoursynth
./autogen.sh
./configure
make && sudo make install

# Make sure vapoursynth could be imported from python
# Change "python3.x" depending on your environment
sudo ln -s /usr/local/lib/python3.x/site-packages/vapoursynth.so /usr/lib/python3.x/lib-dynload/vapoursynth.so
sudo ldconfig
```

#### 6.4 Check if VapourSynth has been installed properly
确认得到了版本号且没有产生报错
```Shell
vspipe --version
```

#### 6.5 [可选] 构建 vslsmashsource
```Shell
# Build lsmash (Not required if already done in 5.3)
git clone https://github.com/l-smash/l-smash.git
cd l-smash
./configure --enable-shared
make && sudo make install
cd ..
 
# Build vslsmashsource
git clone https://github.com/HolyWu/L-SMASH-Works.git
cd L-SMASH-Works
git checkout -b 20200531 refs/tags/20200531
cd VapourSynth
meson build
cd build
ninja && sudo ninja install
cd ../../../
```

</details>

### 7. 构建 NVEncC
```Shell
git clone https://github.com/rigaya/NVEnc --recursive
cd NVEnc
./configure
make
```

检查是否正常执行
```Shell
./nvencc --check-hw
```

应可以得到当前NVENC支持的编解码器列表
```
#0: GeForce GTX 1080 (2560 cores, 1822 MHz)[PCIe3x16][460.32]
Avaliable Codec(s)
H.264/AVC
H.265/HEVC
```