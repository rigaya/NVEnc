
# How to build NVEnc

- [Windows](./Build.en.md#windows)
- [Linux (Ubuntu 19.10)](./Build.en.md#linux-ubuntu-1910)
- [Linux (Ubuntu 18.04)](./Build.en.md#linux-ubuntu-1804)

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

You will also need source code of [Caption2Ass_PCR](https://github.com/maki-rxrz/Caption2Ass_PCR).

```Batchfile
git clone https://github.com/maki-rxrz/Caption2Ass_PCR <path-to-clone>
setx CAPTION2ASS_SRC "<path-to-clone>/src"
```

### 2. Download source code

```Batchfile
git clone https://github.com/rigaya/NVEnc --recursive
```

### 3. Build NVEncC.exe / NVEnc.auo

Finally, open NVEnc.sln, and start build of NVEnc by Visual Studio.

|   | For Debug build | For Release build |
|:---------------------|:------|:--------|
|NVEncC(64).exe | DebugStatic | RelStatic |
|NVEnc.auo (win32 only) | Debug | Release |
|cufilters.auf (win32 only) | DebugFilters | RelFilters |


## Linux (Ubuntu 19.10)

### 0. Requirements

- GPU Driver 435.21 or later
- C++14 Compiler
- CUDA 10
- git
- libraries
  - ffmpeg 4.x libs (libavcodec58, libavformat58, libavfilter7, libavutil56, libswresample3)
  - libass9
  - [Optional] VapourSynth

### 1. Install build tools

```Shell
sudo apt install build-essential git
```

### 2. Install NVIDIA driver

Check driver version which could be installed.
```Shell
sudo add-apt-repository ppa:graphics-drivers/ppa
sudo apt update
ubuntu-drivers devices
```

You shall get the output like below.
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

Select and install the latest driver.
```Shell
sudo apt install nvidia-driver-440
sudo reboot
```

After rebbot, check if the driver has been installed properly.
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

### 3. Install CUDA
```Shell
sudo apt install nvidia-cuda-toolkit
```

### 4. Install required libraries

ffmpegと関連ライブラリを導入します。
```Shell
sudo apt install ffmpeg \
  libavcodec-extra libavcodec-dev libavutil-dev libavformat-dev libswresample-dev libavfilter-dev \
  libass9 libass-dev
```

### 5. [Optional] Install VapourSynth
VapourSynth is required only if you need VapourSynth(vpy) reader support.  

Please go on to [6. Build NVEncC] if you don't need vpy reader.

<details><summary>How to build VapourSynth</summary>

#### 5.1 Install build tools for VapourSynth
```Shell
sudo apt install python3-pip autoconf automake libtool meson
```

#### 5.2 Install zimg
```Shell
git clone https://github.com/sekrit-twc/zimg.git
cd zimg
./autogen.sh
./configure
sudo make install -j16
cd ..
```

#### 5.3 Install cython
```Shell
sudo pip3 install Cython
```

#### 5.4 Install VapourSynth
```Shell
git clone https://github.com/vapoursynth/vapoursynth.git
cd vapoursynth
./autogen.sh
./configure
make -j16
sudo make install

# Make sure vapoursynth could be imported from python
# Change "python3.x" depending on your encironment
sudo ln -s /usr/local/lib/python3.x/site-packages/vapoursynth.so /usr/lib/python3.x/lib-dynload/vapoursynth.so
sudo ldconfig
```

#### 5.5 Check if VapourSynth has been installed properly
Make sure you get version number without errors.
```Shell
vspipe --version
```

#### 5.6 [Option] Build vslsmashsource
```Shell
# Install lsmash
git clone https://github.com/l-smash/l-smash.git
cd l-smash
./configure --enable-shared
sudo make install -j16
cd ..
 
# Install vslsmashsource
git clone https://github.com/HolyWu/L-SMASH-Works.git
cd L-SMASH-Works/VapourSynth
meson build
cd build
sudo ninja install
cd ../../../
```

</details>

### 6. Build NVEncC
```Shell
git clone https://github.com/rigaya/NVEnc --recursive
cd NVEnc
./configure
make -j16
```
Check if it works properly.
```Shell
./nvencc --check-hw
```

You shall get information of the avaliable codecs supported by NVENC.
```
#0: GeForce GTX 1080 (2560 cores, 1822 MHz)[PCIe3x16][440.82]
Avaliable Codec(s)
H.264/AVC
H.265/HEVC
```


## Linux (Ubuntu 18.04)

### 0. Requirements

- GPU Driver 435.21 or later
- C++14 Compiler
- CUDA 10
- git
- libraries
  - ffmpeg 4.x libs (libavcodec58, libavformat58, libavfilter7, libavutil56, libswresample3)
  - libass9
  - [Optional] VapourSynth

### 1. Install build tools

```Shell
sudo apt install git
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

After rebbot, check if the driver has been installed properly. (Tested on AWS g3s.xlarge)

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
  libavcodec-extra58 libavcodec-dev libavutil56 libavutil-dev libavformat58 libavformat-dev \
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
git clone https://github.com/sekrit-twc/zimg.git
cd zimg
./autogen.sh
./configure
sudo make install -j4
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
make -j4
sudo make install

# Make sure vapoursynth could be imported from python
# Change "python3.x" depending on your encironment
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
sudo make install -j4
cd ..
 
# Install vslsmashsource
git clone https://github.com/HolyWu/L-SMASH-Works.git
cd L-SMASH-Works/VapourSynth
meson build
cd build
sudo ninja install
cd ../../../
```

</details>

### 5. Build NVEncC
```Shell
git clone https://github.com/rigaya/NVEnc --recursive
cd NVEnc
./configure
make -j4
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