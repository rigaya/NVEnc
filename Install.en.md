
# How to install NVEncC

- [Windows 10](./Install.en.md#windows)
- Linux
  - [Linux (Ubuntu 20.04)](./Install.en.md#linux-ubuntu-2004)
  - [Linux (Ubuntu 18.04)](./Install.en.md#linux-ubuntu-1804)
  - Other Linux OS  
    For other Linux OS, building from source will be needed. Please check the [build instrcutions](./Build.en.md).


## Windows 10

### 1. Install [NVIDIA Graphics driver](https://www.nvidia.com/Download/index.aspx)
### 2. Download Windows binary  
Windows binary can be found from [this link](https://github.com/rigaya/NVEnc/releases). NVEncC_x.xx_Win32.7z contains 32bit exe file, NVEncC_x.xx_x64.7z contains 64bit exe file.

NVEncC could be run directly from the extracted directory.
  
## Linux (Ubuntu 20.04)

### 1. Install [NVIDIA Graphics driver](https://www.nvidia.com/Download/index.aspx)  
Select "Linux 64bit" and download "run" file. A install file named like "NVIDIA-Linux-x86_64-460.84.run" shall be downloaded, which can be run to install the driver.  
```Shell
chmod u+x NVIDIA-Linux-x86_64-460.84.run
sudo ./NVIDIA-Linux-x86_64-460.84.run
```

### 2. Install nvencc
Download deb package from [this link](https://github.com/rigaya/NVEnc/releases), and install running the following command line. Please note "x.xx" should be replaced to the target version name.

```Shell
sudo apt install ./nvencc_x.xx_Ubuntu20.04_amd64.deb
```

### 3. Addtional Tools

There are some features which require additional installations.  

| Feature | Requirements |
|:--      |:--           |
| --vpp-colorspace | CUDA Toolkit (libnvrtc.so is required)                   |
| avs reader       | [AvisynthPlus](https://github.com/AviSynth/AviSynthPlus) |
| vpy reader       | [VapourSynth](https://www.vapoursynth.com/)              |

  
## Linux (Ubuntu 18.04)

### 1. Install [NVIDIA Graphics driver](https://www.nvidia.com/Download/index.aspx)  
Select "Linux 64bit" and download "run" file. A install file named like "NVIDIA-Linux-x86_64-460.84.run" shall be downloaded, which can be run to install the driver.  
```Shell
chmod u+x NVIDIA-Linux-x86_64-xxx.xx.run
sudo ./NVIDIA-Linux-x86_64-xxx.xx.run
```

### 2. Install nvencc
Download deb package from [this link](https://github.com/rigaya/NVEnc/releases), and install running the following command line. Please note "x.xx" should be replaced to the target version name.

```Shell
# for ffmpeg 4.xx
sudo add-apt-repository ppa:jonathonf/ffmpeg-4
sudo apt update
# Install nvencc
sudo apt install ./nvencc_x.xx_Ubuntu18.04_amd64.deb
```

### 3. Addtional Tools

There are some features which require additional installations.  
  

| Feature | Requirements |
|:--      |:--           |
| --vpp-colorspace | CUDA Toolkit (libnvrtc.so is required)                   |
| avs reader       | [AvisynthPlus](https://github.com/AviSynth/AviSynthPlus) |
| vpy reader       | [VapourSynth](https://www.vapoursynth.com/)              |

  