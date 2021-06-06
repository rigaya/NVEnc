
# NVEncCのインストール方法

- [Windows](./Install.ja.md#windows)
- Linux
  - [Linux (Ubuntu 20.04)](./Install.ja.md#linux-ubuntu-2004)
  - [Linux (Ubuntu 18.04)](./Install.ja.md#linux-ubuntu-1804)
  - その他のLinux OS  
    その他のLinux OS向けには、ソースコードからビルドする必要があります。ビルド方法については、[こちら](./Build.ja.md)を参照してください。


## Windows

### 1. [NVIDIAグラフィックスドライバ](https://www.nvidia.co.jp/Download/index.aspx?lang=jp)をインストールします。
### 2. Windows用実行ファイルをダウンロードして展開します。  
実行ファイルは[こちら](https://github.com/rigaya/NVEnc/releases)からダウンロードできます。NVEncC_x.xx_Win32.7z が 32bit版、NVEncC_x.xx_x64.7z が 64bit版です。通常は、64bit版を使用します。

実行時は展開したフォルダからそのまま実行できます。
  
## Linux (Ubuntu 20.04)

### 1. [NVIDIAグラフィックスドライバ](https://www.nvidia.co.jp/Download/index.aspx?lang=jp)をインストールします。  
OSで「Linux 64bit」を選択し、"run"ファイルをダウンロードします。(例えば、"NVIDIA-Linux-x86_64-460.84.run"などのファイルがダウンロードできます)

その後、runファイルを実行し、ドライバをインストールします。
```Shell
chmod u+x NVIDIA-Linux-x86_64-460.84.run
sudo ./NVIDIA-Linux-x86_64-460.84.run
```

### 2. nvenccのインストール
NVEncCのdebファイルを[こちら](https://github.com/rigaya/NVEnc/releases)からダウンロードします。

その後、下記のようにインストールします。"x.xx"はインストールするバージョンに置き換えてください。

```Shell
sudo apt install ./nvencc_x.xx_Ubuntu20.04_amd64.deb
```

### 3. 追加オプション
NVEncCの下記オプションを使用するには、追加でインストールが必要です。

- avs読み込み  
  [AvisynthPlus](https://github.com/AviSynth/AviSynthPlus)のインストールが必要です。
  
- vpy読み込み  
  [VapourSynth](https://www.vapoursynth.com/)のインストールが必要です。
  
- vpp-colorspace  
  CUDAのインストールが必要です。


## Linux (Ubuntu 18.04)

### 1. [NVIDIAグラフィックスドライバ](https://www.nvidia.co.jp/Download/index.aspx?lang=jp)をインストールします。  
OSで「Linux 64bit」を選択し、"run"ファイルをダウンロードします。(例えば、"NVIDIA-Linux-x86_64-460.84.run"などのファイルがダウンロードできます)

その後、runファイルを実行し、ドライバをインストールします。
```Shell
chmod u+x NVIDIA-Linux-x86_64-xxx.xx.run
sudo ./NVIDIA-Linux-x86_64-xxx.xx.run
```

### 2. nvenccのインストール
NVEncCのdebファイルを[こちら](https://github.com/rigaya/NVEnc/releases)からダウンロードします。

その後、下記のようにインストールします。"x.xx"はインストールするバージョンに置き換えてください。

```Shell
# ffmpeg 4.xx が必要なので
sudo add-apt-repository ppa:jonathonf/ffmpeg-4
sudo apt update
# nvencc のインストール
sudo apt install ./nvencc_x.xx_Ubuntu18.04_amd64.deb
```

### 3. 追加オプション
- avs読み込み  
  [AvisynthPlus](https://github.com/AviSynth/AviSynthPlus)のインストールが必要です。
  
- vpy読み込み  
  [VapourSynth](https://www.vapoursynth.com/)のインストールが必要です。
  
- --vpp-colorspace  
  CUDAのインストールが必要です。(libnvrtc.so が必要です。)
















