
# 安装 NVEncC

- [Windows 10](#windows)
- Linux
  - [Linux (Ubuntu 20.04)](#linux-ubuntu-2004)
  - [Linux (Ubuntu 18.04)](#linux-ubuntu-1804)
  - 其他Linux操作系统 
    对于其他Linux操作系统，请查看[构建说明](./Build.cn.md).


## Windows 10

### 1. 安装 [NVIDIA Graphics driver](https://www.nvidia.com/Download/index.aspx)
### 2. 下载Windows版二进制文件  
可从[这里](https://github.com/rigaya/NVEnc/releases)下载。
NVEncC_x.xx_Win32.7z 包含32位程序, NVEncC_x.xx_x64.7z 包含64位程序。

NVEncC可以在文件夹中直接运行

## Linux (Ubuntu 20.04)

### 1. 安装 [NVIDIA Graphics driver](https://www.nvidia.com/Download/index.aspx)  
选中"Linux 64bit"并下载文件。应得到格式为"NVIDIA-Linux-x86_64-xxx.xx.run"的安装文件，运行该文件以安装驱动程序。
```Shell
chmod u+x NVIDIA-Linux-x86_64-xxx.xx.run
sudo ./NVIDIA-Linux-x86_64-xxx.xx.run
```

### 2. 安装 nvencc
从[这里](https://github.com/rigaya/NVEnc/releases)下载deb包。
使用如下命令安装nvencc，注意"x.xx"应替换为对应版本的名称。

```Shell
sudo apt install ./nvencc_x.xx_Ubuntu20.04_amd64.deb
```

### 3. 其他工具
  
一些功能需要额外安装程序才能使用

| 功能 | 依赖 |
|:--      |:--           |
| --vpp-colorspace | CUDA Toolkit (需要libnvrtc.so)                   |
| avs reader       | [AvisynthPlus](https://github.com/AviSynth/AviSynthPlus) |
| vpy reader       | [VapourSynth](https://www.vapoursynth.com/)              |

  
## Linux (Ubuntu 18.04)

### 1. 安装 [NVIDIA Graphics driver](https://www.nvidia.com/Download/index.aspx)  
选中"Linux 64bit"并下载文件。应得到格式为"NVIDIA-Linux-x86_64-xxx.xx.run"的安装文件，运行该文件以安装驱动程序。

```Shell
chmod u+x NVIDIA-Linux-x86_64-xxx.xx.run
sudo ./NVIDIA-Linux-x86_64-xxx.xx.run
```

### 2. Install nvencc
从[这里](https://github.com/rigaya/NVEnc/releases)下载deb包。
使用如下命令安装nvencc，注意"x.xx"应替换为对应版本的名称。

```Shell
# for ffmpeg 4.xx
sudo add-apt-repository ppa:jonathonf/ffmpeg-4
sudo apt update
# Install nvencc
sudo apt install ./nvencc_x.xx_Ubuntu18.04_amd64.deb
```

### 3. Addtional Tools

一些功能需要额外安装程序才能使用

| 功能 | 依赖 |
|:--      |:--           |
| --vpp-colorspace | CUDA Toolkit (需要libnvrtc.so)                   |
| avs reader       | [AvisynthPlus](https://github.com/AviSynth/AviSynthPlus) |
| vpy reader       | [VapourSynth](https://www.vapoursynth.com/)              |

  