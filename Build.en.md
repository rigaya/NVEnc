
# How to build NVEnc

- [Windows](./Build.en.md#windows)
- [Linux (Ubuntu 20.04 - 24.04)](./Build.en.md#linux-ubuntu-2004---2404)
- [Linux (Ubuntu 18.04)](./Build.en.md#linux-ubuntu-1804)
- [Linux (Fedora 33)](./Build.en.md#linux-fedora-33)

## Windows

### 0. Requirements
To build NVEnc, components below are required.

- Visual Studio 2019
- CUDA 10.1 or later (x64)
- CUDA 11.0 or later (x86)
- [Avisynth](https://github.com/AviSynth/AviSynthPlus) SDK
- [VapourSynth](http://www.vapoursynth.com/) SDK

### 1. Install build tools.

Install Avisynth+ and VapourSynth, with the SDKs.

Then, "avisynth_c.h" of the Avisynth+ SDK and "VapourSynth.h" of the VapourSynth SDK should be added to the include path of Visual Studio.

These include path can be passed by environment variables "AVISYNTH_SDK" and "VAPOURSYNTH_SDK".

With default installation, environment variables could be set as below.
```Batchfile
setx AVISYNTH_SDK "C:\Program Files (x86)\AviSynth+\FilterSDK"
setx VAPOURSYNTH_SDK "C:\Program Files (x86)\VapourSynth\sdk"
```

### 2. Download source code

```Batchfile
git clone https://github.com/rigaya/NVEnc --recursive
cd NVEnc
curl -s -o ffmpeg_lgpl.7z -L https://github.com/rigaya/ffmpeg_dlls_for_hwenc/releases/download/20241102/ffmpeg_dlls_for_hwenc_20241102.7z
7z x -offmpeg_lgpl -y ffmpeg_lgpl.7z
```

### 3. Build NVEncC.exe / NVEnc.auo

Finally, open NVEnc.sln, and start build of NVEnc by Visual Studio.

|   | For Debug build | For Release build |
|:---------------------|:------|:--------|
|NVEncC(64).exe | DebugStatic | RelStatic |
|NVEnc.auo (win32 only) | Debug | Release |
|cufilters.auf (win32 only) | DebugFilters | RelFilters |


## Linux (Ubuntu 20.04 - 24.04)

### 0. Requirements

- GPU Driver 435.21 or later
- C++17 Compiler
- CUDA 10-12
- rust + cargo-c
- git
- libraries
  - ffmpeg 4.x-7.x libs (libavcodec*, libavformat*, libavfilter*, libavutil*, libswresample*, libavdevice*)
  - libass9
  - [Optional] AvisynthPlus
  - [Optional] VapourSynth

### 1. Install build tools

- Install build tools

  ```Shell
  sudo apt install build-essential git
  ```

- Install rust + cargo-c (for libdovi build)

  ```Shell
  sudo apt install libssl-dev curl
  curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --profile minimal \
    && . ~/.cargo/env \
    && cargo install cargo-c
  ```

### 2. Install NVIDIA driver

### 3. Install CUDA
Download CUDA deb file for each OS.
```Shell
# For Ubuntu 20.04
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-keyring_1.1-1_all.deb
# For Ubuntu 22.04
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
```

Install CUDA.
```Shell
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt-get update
sudo apt-get -y install --no-install-recommends cuda-toolkit
export CUDA_PATH=/usr/local/cuda
```

> [!NOTE]
> When isntall failes on Ubuntu 24.04  using the command line above, selecting only the required packages as below would work.
> CUDA_VER_MAJOR and CUDA_VER_MINOR might be needed to be chaged.
> ```
> CUDA_VER_MAJOR=12
> CUDA_VER_MINOR=4
> apt-get -y install cuda-drivers cuda-compiler-${CUDA_VER_MAJOR}-${CUDA_VER_MINOR} \
>   cuda-cudart-dev-${CUDA_VER_MAJOR}-${CUDA_VER_MINOR} cuda-driver-dev-${CUDA_VER_MAJOR}-${CUDA_VER_MINOR} \
>   cuda-nvrtc-dev-${CUDA_VER_MAJOR}-${CUDA_VER_MINOR} libcurand-dev-${CUDA_VER_MAJOR}-${CUDA_VER_MINOR} \
>   libnpp-dev-${CUDA_VER_MAJOR}-${CUDA_VER_MINOR} cuda-nvml-dev-${CUDA_VER_MAJOR}-${CUDA_VER_MINOR}
> ```

### 4. Install required libraries

Install ffmpeg and other required libraries.
```Shell
sudo apt install ffmpeg \
  libavcodec-extra libavcodec-dev libavutil-dev libavformat-dev libswresample-dev libavfilter-dev libavdevice-dev \
  libass-dev libx11-dev libplacebo-dev
```

### 5. [Optional] Install AvisynthPlus
AvisynthPlus is required only if you need AvisynthPlus(avs) reader support.  

Please go on to [7. Build NVEncC] if you don't need avs reader.

<details><summary>How to build AvisynthPlus</summary>

#### 5.1 Install build tools for AvisynthPlus
```Shell
sudo apt install cmake
```

#### 5.2 Install AvisynthPlus
```Shell
git clone https://github.com/AviSynth/AviSynthPlus.git
cd AviSynthPlus
mkdir avisynth-build && cd avisynth-build 
cmake ../
make && sudo make install
cd ../..
```
</details>

### 6. [Optional] Install VapourSynth
VapourSynth is required only if you need VapourSynth(vpy) reader support.  

Please go on to [7. Build NVEncC] if you don't need vpy reader.

<details><summary>How to build VapourSynth</summary>

#### 6.1 Install build tools for VapourSynth
```Shell
sudo apt install python3-pip cython3 autoconf automake libtool meson libzimg-dev
```

#### 6.2 Install VapourSynth
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

#### 6.3 Check if VapourSynth has been installed properly
Make sure you get version number without errors.
```Shell
LD_LIBRARY_PATH=/usr/local/lib vspipe --version
```

</details>

### 7. Build NVEncC
```Shell
git clone https://github.com/rigaya/NVEnc --recursive
cd NVEnc
./configure
make
```
Check if it works properly.
```Shell
./nvencc --check-hw
```

You shall get information of the avaliable codecs supported by NVENC.
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

### 0. Requirements

- GPU Driver 435.21 or later
- C++17 Compiler
- CUDA 10
- git
- libraries
  - ffmpeg 4.x libs (libavcodec58, libavformat58, libavfilter7, libavutil56, libswresample3, libavdevice58)
  - libass9
  - [Optional] VapourSynth

### 1. Install build tools

- Install build tools

  ```Shell
  sudo apt install git g++-8
  ```

- Install rust + cargo-c (for libdovi build)

  ```Shell
  sudo apt install libssl-dev curl
  curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --profile minimal \
    && . ~/.cargo/env \
    && cargo install cargo-c
  ```

### 2. Install NVIDIA driver and CUDA 10.2
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

After reboot, check if the driver has been installed properly. (Tested on AWS g3s.xlarge)

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

### 3. Install required libraries

Install ffmpeg 4.x libraries.
```Shell
sudo add-apt-repository ppa:jonathonf/ffmpeg-4
sudo apt update
sudo apt install ffmpeg \
  libavcodec-extra58 libavcodec-dev libavutil56 libavutil-dev libavformat58 libavformat-dev libavdevice58 libavdevice-dev \
  libswresample3 libswresample-dev libavfilter-extra7 libavfilter-dev libass9 libass-dev
```

### 4. [Optional] Install VapourSynth
VapourSynth is required only if you need VapourSynth(vpy) reader support.  

Please go on to [6. Build NVEncC] if you don't need vpy reader.

<details><summary>How to build VapourSynth</summary>

#### 4.1 Install build tools for VapourSynth
```Shell
sudo apt install python3-pip autoconf automake libtool meson
```

#### 4.2 Install zimg
```Shell
git clone https://github.com/sekrit-twc/zimg.git --recursive
cd zimg
./autogen.sh
./configure
make && sudo make install
cd ..
```

#### 4.3 Install cython
```Shell
sudo pip3 install Cython
```

#### 4.4 Install VapourSynth
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

#### 4.5 Check if VapourSynth has been installed properly
Make sure you get version number without errors.
```Shell
vspipe --version
```

#### 4.6 [Option] Build vslsmashsource
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

### 5. Build NVEncC
```Shell
git clone https://github.com/rigaya/NVEnc --recursive
cd NVEnc
./configure --cxx=g++-8
make
```
Check if it works properly.
```Shell
./nvencc --check-hw
```

You shall get information of the avaliable codecs supported by NVENC. (Tested on AWS g3s.xlarge)
```
#0: Tesla M60 (2048 cores, 1177 MHz)[PCIe3x16][440.33]
Avaliable Codec(s)
H.264/AVC
H.265/HEVC
```


## Linux (Fedora 33)

### 0. Requirements
- C++17 compiler
- CUDA 11
- git
- libraries
  - ffmpeg 4.x libs (libavcodec58, libavformat58, libavfilter7, libavutil56, libswresample3, libavdevice58)
  - libass9
  - [Option] AvisynthPlus
  - [Option] VapourSynth

### 1. Install build tools

- Install build tools

  ```Shell
  sudo dnf install @development-tools
  ```

- Install rust + cargo-c (for libdovi build)

  ```Shell
  sudo apt install libssl-dev curl
  curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --profile minimal \
    && . ~/.cargo/env \
    && cargo install cargo-c
  ```

### 2. Getting ready to install CUDA and NVIDIA driver

```Shell
sudo dnf update
sudo dnf upgrade
sudo dnf clean all
sudo dnf install kernel-devel
sudo dnf install make pciutils acpid libglvnd-devel
sudo dnf install dkms
```

### 3. Install CUDA and NVIDIA driver
Install CUDA and the NVIDIA driver included in CUDA package.
```Shell
wget https://developer.download.nvidia.com/compute/cuda/11.2.1/local_installers/cuda-repo-fedora33-11-2-local-11.2.1_460.32.03-1.x86_64.rpm
sudo rpm -ivh cuda-repo-fedora33-11-2-local-11.2.1_460.32.03-1.x86_64.rpm
sudo dnf clean all
sudo dnf install cuda
reboot
```
CUDA was installed in /usr/local/cuda.

After reboot, check if the driver has been installed properly.
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

### 4. Install required libraries

Install ffmpeg and other required libraries.
```Shell
sudo dnf install https://download1.rpmfusion.org/free/fedora/rpmfusion-free-release-$(rpm -E %fedora).noarch.rpm
sudo dnf install ffmpeg ffmpeg-devel libass libass-devel
```

### 5. [Optional] Install AvisynthPlus
AvisynthPlus is required only if you need AvisynthPlus(avs) reader support.  

Please go on to [7. Build NVEncC] if you don't need avs reader.

<details><summary>How to build AvisynthPlus</summary>

#### 5.1 Install build tools for AvisynthPlus
```Shell
sudo dnf install cmake
```

#### 5.2 Install AvisynthPlus
```Shell
git clone git://github.com/AviSynth/AviSynthPlus.git
cd AviSynthPlus
mkdir avisynth-build && cd avisynth-build 
cmake ../
make && sudo make install
cd ../..
```

#### 5.3 [Option] Build lsmashsource
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


### 6. [Option] Install VapourSynth
VapourSynth is required only if you need VapourSynth(vpy) reader support.  

Please go on to [7. Build NVEncC] if you don't need vpy reader.

<details><summary>How to build VapourSynth</summary>

#### 6.1 Install build tools for VapourSynth
```Shell
sudo dnf install zimg zimg-devel meson autotools automake libtool python3-devel ImageMagick
```

#### 6.2 Install cython
```Shell
sudo pip3 install Cython --install-option="--no-cython-compile"
```

#### 6.3 Build VapourSynth
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
Make sure you get version number without errors.
```Shell
vspipe --version
```

#### 6.5 [Option] Build vslsmashsource
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

### 7. Build NVEncC
```Shell
git clone https://github.com/rigaya/NVEnc --recursive
cd NVEnc
./configure
make
```

Check if it works properly.
```Shell
./nvencc --check-hw
```

You shall get information of the avaliable codecs supported by NVENC.
```
#0: GeForce GTX 1080 (2560 cores, 1822 MHz)[PCIe3x16][460.32]
Avaliable Codec(s)
H.264/AVC
H.265/HEVC
```