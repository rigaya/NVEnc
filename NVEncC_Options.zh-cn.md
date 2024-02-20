
# NVEncC 选项列表

**[日本語版はこちら＞＞](./NVEncC_Options.ja.md)**

- [命令行示例](#命令行示例)
  - [基本命令](#基本命令)
  - [更多示例](#更多示例)
    - [使用 hw (cuvid) 解码器](#使用-hw-cuvid-解码器)
    - [使用 hw (cuvid) 解码器(交错)](#使用-hw-cuvid-解码器交错)
    - [Avisynth 示例 (avs 和 vpy 均可通过 vfw 读取)](#Avisynth-示例-avs-和-vpy-均可通过-vfw-读取)
    - [管道输入示例](#管道输入示例)
    - [从 FFmpeg 管道输入](#从-FFmpeg-管道输入)
    - [从 FFmpeg 传递视频和音频](#从-FFmpeg-传递视频和音频)
- [选项格式](#选项格式)
- [显示选项](#显示选项)
  - [-h, -? --help](#-h-----help)
  - [-v, --version](#-v---version)
  - [--option-list](#--option-list)
  - [--check-device](#--check-device)
  - [--check-hw \[\<int\>\]](#--check-hw-int)
  - [--check-features \[\<int\>\]](#--check-features-int)
  - [--check-environment](#--check-environment)
  - [--check-codecs, --check-decoders, --check-encoders](#--check-codecs---check-decoders---check-encoders)
  - [--check-profiles \<string\>](#--check-profiles-string)
  - [--check-formats](#--check-formats)
  - [--check-protocols](#--check-protocols)
  - [--check-avdevices](#--check-avdevices)
  - [--check-filters](#--check-filters)
  - [--check-avversion](#--check-avversion)
- [基本编码选项](#基本编码选项)
  - [-d, --device \<int\>](#-d---device-int)
  - [-c, --codec \<string\>](#-c---codec-string)
  - [-o, --output \<string\>](#-o---output-string)
  - [-i, --input \<string\>](#-i---input-string)
  - [--raw](#--raw)
  - [--y4m](#--y4m)
  - [--avi](#--avi)
  - [--avs](#--avs)
  - [--vpy](#--vpy)
  - [--avsw](#--avsw)
  - [--avhw](#--avhw)
  - [--interlace \<string\>](#--interlace-string)
  - [--video-track \<int\>](#--video-track-int)
  - [--crop \<int\>,\<int\>,\<int\>,\<int\>](#--crop-intintintint)
  - [--frames \<int\>](#--frames-int)
  - [--fps \<int\>/\<int\> or \<float\>](#--fps-intint-or-float)
  - [--input-res \<int\>x\<int\>](#--input-res-intxint)
  - [--output-res \<int\>x\<int\>\[,\<string\>=\<string\>\]](#--output-res-intxintstringstring)
  - [--input-csp \<string\>](#--input-csp-string)
- [编码模式选项](#编码模式选项)
  - [--qvbr  \<float\>](#--qvbr--float)
  - [--cbr \<int\>](#--cbr-int)
  - [--vbr \<int\>](#--vbr-int)
  - [--cqp \<int\> or \<int\>:\<int\>:\<int\>](#--cqp-int-or-intintint)
- [其他适用于编码器的选项](#其他适用于编码器的选项)
  - [-u, --preset](#-u---preset)
  - [--output-depth \<int\>](#--output-depth-int)
  - [--output-csp \<string\>](#--output-csp-string)
  - [--multipass \<string\>](#--multipass-string)
  - [--lossless  \[H.264/HEVC\]](#--lossless--h264hevc)
  - [--max-bitrate \<int\>](#--max-bitrate-int)
  - [--vbv-bufsize \<int\>](#--vbv-bufsize-int)
  - [--qp-init \<int\> or \<int\>:\<int\>:\<int\>](#--qp-init-int-or-intintint)
  - [--qp-min \<int\> or \<int\>:\<int\>:\<int\>](#--qp-min-int-or-intintint)
  - [--qp-max \<int\> or \<int\>:\<int\>:\<int\>](#--qp-max-int-or-intintint)
  - [--chroma-qp-offset \<int\>  \[H.264/HEVC\]](#--chroma-qp-offset-int--h264hevc)
  - [--vbr-quality \<float\>](#--vbr-quality-float)
  - [--dynamic-rc \<int\>:\<int\>:\<int\>\<int\>,\<param1\>=\<value1\>\[,\<param2\>=\<value2\>\],...](#--dynamic-rc-intintintintparam1value1param2value2)
  - [--lookahead \<int\>](#--lookahead-int)
  - [--no-i-adapt](#--no-i-adapt)
  - [--no-b-adapt](#--no-b-adapt)
  - [--strict-gop](#--strict-gop)
  - [--gop-len \<int\>](#--gop-len-int)
  - [-b, --bframes \<int\>](#-b---bframes-int)
  - [--ref \<int\>](#--ref-int)
  - [--multiref-l0 \<int\> \[H.264/HEVC\]](#--multiref-l0-int-h264hevc)
  - [--multiref-l1 \<int\> \[H.264/HEVC\]](#--multiref-l1-int-h264hevc)
  - [--weightp](#--weightp)
  - [--nonrefp](#--nonrefp)
  - [--aq](#--aq)
  - [--aq-temporal](#--aq-temporal)
  - [--aq-strength \<int\>](#--aq-strength-int)
  - [--bref-mode \<string\>](#--bref-mode-string)
  - [--direct \<string\> \[H.264\]](#--direct-string-h264)
  - [--(no-)adapt-transform \[H.264\]](#--no-adapt-transform-h264)
  - [--hierarchial-p \[H.264\]](#--hierarchial-p-h264)
  - [--hierarchial-b \[H.264\]](#--hierarchial-b-h264)
  - [--temporal-layers \<int\> \[H.264\]](#--temporal-layers-int-h264)
  - [--mv-precision \<string\>](#--mv-precision-string)
  - [--slices \<int\> \[H.264/HEVC\]](#--slices-int-h264hevc)
  - [--cabac \[H.264\]](#--cabac-h264)
  - [--cavlc \[H.264\]](#--cavlc-h264)
  - [--bluray \[H.264\]](#--bluray-h264)
  - [--(no-)deblock \[H.264\]](#--no-deblock-h264)
  - [--cu-max \<int\> \[HEVC\]](#--cu-max-int-hevc)
  - [--cu-min \<int\> \[HEVC\]](#--cu-min-int-hevc)
  - [--part-size-min \<int\> \[AV1\]](#--part-size-min-int-av1)
  - [--part-size-max \<int\> \[AV1\]](#--part-size-max-int-av1)
  - [--tile-columns \<int\> \[AV1\]](#--tile-columns-int-av1)
  - [--tile-rows \<int\> \[AV1\]](#--tile-rows-int-av1)
  - [--max-temporal-layers \<int\> \[AV1\]](#--max-temporal-layers-int-av1)
  - [--refs-forward \<int\> \[AV1\]](#--refs-forward-int-av1)
  - [--refs-backward \<int\> \[AV1\]](#--refs-backward-int-av1)
  - [--level \<string\>](#--level-string)
  - [--profile \<string\>](#--profile-string)
  - [--tier \<string\>  \[HEVC only\]](#--tier-string--hevc-only)
  - [--sar \<int\>:\<int\>](#--sar-intint)
  - [--dar \<int\>:\<int\>](#--dar-intint)
  - [--colorrange \<string\>](#--colorrange-string)
  - [--videoformat \<string\>](#--videoformat-string)
  - [--colormatrix \<string\>](#--colormatrix-string)
  - [--colorprim \<string\>](#--colorprim-string)
  - [--transfer \<string\>](#--transfer-string)
  - [--chromaloc \<int\> or "auto"](#--chromaloc-int-or-auto)
  - [--max-cll \<int\>,\<int\> or "copy" \[HEVC, AV1\]](#--max-cll-intint-or-copy-hevc-av1)
  - [--master-display \<string\> or "copy" \[HEVC, AV1\]](#--master-display-string-or-copy-hevc-av1)
  - [--atc-sei \<string\> or \<int\> \[HEVC only\]](#--atc-sei-string-or-int-hevc-only)
  - [--dhdr10-info \<string\> \[HEVC, AV1\]](#--dhdr10-info-string-hevc-av1)
  - [--dhdr10-info copy \[HEVC, AV1\]](#--dhdr10-info-copy-hevc-av1)
  - [--dolby-vision-profile \<float\>](#--dolby-vision-profile-float)
  - [--dolby-vision-rpu \<string\>](#--dolby-vision-rpu-string)
  - [--aud \[H.264/HEVC\]](#--aud-h264hevc)
  - [--repeat-headers](#--repeat-headers)
  - [--pic-struct \[H.264/HEVC\]](#--pic-struct-h264hevc)
  - [--split-enc \<string\>](#--split-enc-string)
  - [--ssim](#--ssim)
  - [--psnr](#--psnr)
  - [--vmaf \[\<param1\>=\<value1\>\]\[,\<param2\>=\<value2\>\],...](#--vmaf-param1value1param2value2)
- [输入输出 / 音频 / 字幕设置 ](#输入输出--音频--字幕设置)
  - [--input-analyze \<float\>](#--input-analyze-float)
  - [--input-probesize \<int\>](#--input-probesize-int)
  - [--trim \<int\>:\<int\>\[,\<int\>:\<int\>\]\[,\<int\>:\<int\>\]...](#--trim-intintintintintint)
  - [--seek \[\<int\>:\]\[\<int\>:\]\<int\>\[.\<int\>\]](#--seek-intintintint)
  - [--seekto \[\<int\>:\]\[\<int\>:\]\<int\>\[.\<int\>\]](#--seekto-intintintint)
  - [--input-format \<string\>](#--input-format-string)
  - [-f, --output-format \<string\>](#-f---output-format-string)
  - [--video-track \<int\>](#--video-track-int-1)
  - [--video-streamid \<int\>](#--video-streamid-int)
  - [--video-tag \<string\>](#--video-tag-string)
  - [--video-metadata \<string\> or \<string\>=\<string\>](#--video-metadata-string-or-stringstring)
  - [--audio-copy \[\<int/string\>;\[,\<int/string\>\]...\]](#--audio-copy-intstringintstring)
  - [--audio-codec \[\[\<int/string\>?\]\<string\>\[:\<string\>=\<string\>\[,\<string\>=\<string\>\]...\]...\]](#--audio-codec-intstringstringstringstringstringstring)
  - [--audio-bitrate \[\<int/string\>?\]\<int\>](#--audio-bitrate-intstringint)
  - [--audio-profile \[\<int/string\>?\]\<string\>](#--audio-profile-intstringstring)
  - [--audio-stream \[\<int/string\>?\]{\<string1\>}\[:\<string2\>\]](#--audio-stream-intstringstring1string2)
  - [--audio-samplerate \[\<int/string\>?\]\<int\>](#--audio-samplerate-intstringint)
  - [--audio-resampler \<string\>](#--audio-resampler-string)
  - [--audio-delay \[\<int/string\>?\]\<float\>](#--audio-delay-intstringfloat)
  - [--audio-file \[\<int/string\>?\]\[\<string\>\]\<string\>](#--audio-file-intstringstringstring)
  - [--audio-filter \[\<int/string\>?\]\<string\>](#--audio-filter-intstringstring)
  - [--audio-disposition \[\<int/string\>?\]\<string\>\[,\<string\>\]\[\]...](#--audio-disposition-intstringstringstring)
  - [--audio-metadata \[\<int/string\>?\]\<string\> or \[\<int/string\>?\]\<string\>=\<string\>](#--audio-metadata-intstringstring-or-intstringstringstring)
  - [--audio-bsf \[\<int/string\>?\]\<string\>](#--audio-bsf-intstringstring)
  - [--audio-ignore-decode-error \<int\>](#--audio-ignore-decode-error-int)
  - [--audio-source \<string\>\[:{\<int\>?}\[;\<param1\>=\<value1\>...\]/\[\]...\]](#--audio-source-stringintparam1value1)
  - [--chapter \<string\>](#--chapter-string)
  - [--chapter-copy](#--chapter-copy)
  - [--chapter-no-trim](#--chapter-no-trim)
  - [--key-on-chapter](#--key-on-chapter)
  - [--keyfile \<string\>](#--keyfile-string)
  - [--sub-source \<string\>\[:{\<int\>?}\[;\<param1\>=\<value1\>...\]/\[\]...\]](#--sub-source-stringintparam1value1)
  - [--sub-copy \[\<int/string\>;\[,\<int/string\>\]...\]](#--sub-copy-intstringintstring)
  - [--sub-disposition \[\<int/string\>?\]\<string\>](#--sub-disposition-intstringstring)
  - [--sub-metadata \[\<int/string\>?\]\<string\> or \[\<int/string\>?\]\<string\>=\<string\>](#--sub-metadata-intstringstring-or-intstringstringstring)
  - [--sub-bsf \[\<int/string\>?\]\<string\>](#--sub-bsf-intstringstring)
  - [--data-copy \[\<int\>\[,\<int\>\]...\]](#--data-copy-intint)
  - [--attachment-copy \[\<int\>\[,\<int\>\]...\]](#--attachment-copy-intint)
  - [--attachment-source \<string\>\[:{\<int\>?}\[;\<param1\>=\<value1\>\]...\]...](#--attachment-source-stringintparam1value1)
  - [--input-option \<string1\>:\<string2\>](#--input-option-string1string2)
  - [-m, --mux-option \<string1\>:\<string2\>](#-m---mux-option-string1string2)
  - [--metadata \<string\> or \<string\>=\<string\>](#--metadata-string-or-stringstring)
  - [--avsync \<string\>](#--avsync-string)
  - [--timecode \[\<string\>\]](#--timecode-string)
  - [--tcfile-in \<string\>](#--tcfile-in-string)
  - [--timebase \<int\>/\<int\>](#--timebase-intint)
  - [--input-hevc-bsf \<string\>](#--input-hevc-bsf-string)
  - [--allow-other-negative-pts](#--allow-other-negative-pts)
- [Vpp 设置](#Vpp-设置)
  - [Vpp 过滤顺序](#Vpp-过滤顺序)
  - [--vpp-colorspace \[\<param1\>=\<value1\>\]\[,\<param2\>=\<value2\>\],...](#--vpp-colorspace-param1value1param2value2)
  - [--vpp-delogo \<string\>\[,\<param1\>=\<value1\>\]\[,\<param2\>=\<value2\>\],...](#--vpp-delogo-stringparam1value1param2value2)
  - [--vpp-rff](#--vpp-rff)
  - [--vpp-deinterlace \<string\>](#--vpp-deinterlace-string)
  - [--vpp-afs \[\<param1\>=\<value1\>\]\[,\<param2\>=\<value2\>\],...](#--vpp-afs-param1value1param2value2)
  - [--vpp-nnedi \[\<param1\>=\<value1\>\]\[,\<param2\>=\<value2\>\],...](#--vpp-nnedi-param1value1param2value2)
  - [--vpp-yadif \[\<param1\>=\<value1\>\]](#--vpp-yadif-param1value1)
  - [--vpp-decimate \[\<param1\>=\<value1\>\]\[,\<param2\>=\<value2\>\],...](#--vpp-decimate-param1value1param2value2)
  - [--vpp-mpdecimate \[\<param1\>=\<value1\>\]\[,\<param2\>=\<value2\>\],...](#--vpp-mpdecimate-param1value1param2value2)
  - [--vpp-select-every \<int\>\[,\<param1\>=\<int\>\]](#--vpp-select-every-intparam1int)
  - [--vpp-rotate \<int\>](#--vpp-rotate-int)
  - [--vpp-transform \[\<param1\>=\<value1\>\]\[,\<param2\>=\<value2\>\],...](#--vpp-transform-param1value1param2value2)
  - [--vpp-convolution3d \[\<param1\>=\<value1\>\]\[,\<param2\>=\<value2\>\],...](#--vpp-convolution3d-param1value1param2value2)
  - [--vpp-nvvfx-denoise \[\<param1\>=\<value1\>\]\[,\<param2\>=\<value2\>\],...](#--vpp-nvvfx-denoise-param1value1param2value2)
  - [--vpp-nvvfx-artifact-reduction \[\<param1\>=\<value1\>\]\[,\<param2\>=\<value2\>\],...](#--vpp-nvvfx-artifact-reduction-param1value1param2value2)
  - [--vpp-smooth \[\<param1\>=\<value1\>\]\[,\<param2\>=\<value2\>\],...](#--vpp-smooth-param1value1param2value2)
  - [--vpp-denoise-dct \[\<param1\>=\<value1\>\]\[,\<param2\>=\<value2\>\],...](#--vpp-denoise-dct-param1value1param2value2)
  - [--vpp-knn \[\<param1\>=\<value1\>\]\[,\<param2\>=\<value2\>\],...](#--vpp-knn-param1value1param2value2)
  - [--vpp-pmd \[\<param1\>=\<value1\>\]\[,\<param2\>=\<value2\>\],...](#--vpp-pmd-param1value1param2value2)
  - [--vpp-gauss \<int\>](#--vpp-gauss-int)
  - [--vpp-subburn \[\<param1\>=\<value1\>\]\[,\<param2\>=\<value2\>\],...](#--vpp-subburn-param1value1param2value2)
  - [--vpp-resize \<string\> or \[\<param1\>=\<value1\>\]\[,\<param2\>=\<value2\>\],...](#--vpp-resize-string-or-param1value1param2value2)
  - [--vpp-unsharp \[\<param1\>=\<value1\>\]\[,\<param2\>=\<value2\>\],...](#--vpp-unsharp-param1value1param2value2)
  - [--vpp-edgelevel \[\<param1\>=\<value1\>\]\[,\<param2\>=\<value2\>\],...](#--vpp-edgelevel-param1value1param2value2)
  - [--vpp-warpsharp \[\<param1\>=\<value1\>\]\[,\<param2\>=\<value2\>\],...](#--vpp-warpsharp-param1value1param2value2)
  - [--vpp-curves \[\<param1\>=\<value1\>\]\[,\<param2\>=\<value2\>\],...](#--vpp-curves-param1value1param2value2)
  - [--vpp-tweak \[\<param1\>=\<value1\>\]\[,\<param2\>=\<value2\>\],...](#--vpp-tweak-param1value1param2value2)
  - [--vpp-curves \[\<param1\>=\<value1\>\]\[,\<param2\>=\<value2\>\],...](#--vpp-curves-param1value1param2value2)
  - [--vpp-deband \[\<param1\>=\<value1\>\]\[,\<param2\>=\<value2\>\],...](#--vpp-deband-param1value1param2value2)
  - [--vpp-pad \<int\>,\<int\>,\<int\>,\<int\>](#--vpp-pad-intintintint)
  - [--vpp-overlay \[\<param1\>=\<value1\>\]\[,\<param2\>=\<value2\>\],...](#--vpp-overlay-param1value1param2value2)
  - [--vpp-perf-monitor](#--vpp-perf-monitor)
  - [--vpp-nvvfx-model-dir \<string\>](#--vpp-nvvfx-model-dir-string)
- [其他设置](#其他设置)
  - [--cuda-schedule \<string\>](#--cuda-schedule-string)
  - [--disable-nvml \<int\>](#--disable-nvml-int)
  - [--output-buf \<int\>](#--output-buf-int)
  - [--output-thread \<int\>](#--output-thread-int)
  - [--log \<string\>](#--log-string)
  - [--log-level \[\<param1\>=\]\<value\>\[,\<param2\>=\<value\>\]...](#--log-level-param1valueparam2value)
  - [--log-opt \<param1\>=\<value\>\[,\<param2\>=\<value\>\]...](#--log-opt-param1valueparam2value)
  - [--log-framelist \[\<string\>\]](#--log-framelist-string)
  - [--log-packets \[\<string\>\]](#--log-packets-string)
  - [--log-mux-ts \[\<string\>\]](#--log-mux-ts-string)
  - [--thread-affinity \[\<string1\>=\]{\<string2\>\[#\<int\>\[:\<int\>\]...\] or 0x\<hex\>}](#--thread-affinity-string1string2intint-or-0xhex)
  - [--thread-priority \[\<string1\>=\]\<string2\>\[#\<int\>\[:\<int\>\]...\]](#--thread-priority-string1string2intint)
  - [--thread-throttling \[\<string1\>=\]\<string2\>\[#\<int\>\[:\<int\>\]...\]](#--thread-throttling-string1string2intint)
  - [--option-file \<string\>](#--option-file-string)
  - [--max-procfps \<int\>](#--max-procfps-int)
  - [--lowlatency](#--lowlatency)
  - [--avsdll \<string\>](#--avsdll-string)
  - [--process-codepage \<string\> \[Windows OS only\]](#--process-codepage-string-windows-os-only)
  - [--perf-monitor \[\<string\>\[,\<string\>\]...\]](#--perf-monitor-stringstring)
  - [--perf-monitor-interval \<int\>](#--perf-monitor-interval-int)


## 命令行示例


### 基本命令

```Batchfile
NVEncC.exe [Options] -i <filename> -o <filename>
```

### 更多示例
#### 使用 hw (cuvid) 解码器

```Batchfile
NVEncC --avhw -i "<mp4(H.264/AVC) file>" -o "<outfilename.264>"
```

#### 使用 hw (cuvid) 解码器(交错)

```Batchfile
NVEncC --avhw --interlace tff -i "<mp4(H.264/AVC) file>" -o "<outfilename.264>"
```

#### Avisynth 示例 (avs 和 vpy 均可通过 vfw 读取)

```Batchfile
NVEncC -i "<avsfile>" -o "<outfilename.264>"
```

#### 管道输入示例

```Batchfile
avs2pipemod -y4mp "<avsfile>" | NVEncC --y4m -i - -o "<outfilename.264>"
```

#### 从 FFmpeg 管道输入

```Batchfile
ffmpeg -y -i "<inputfile>" -an -pix_fmt yuv420p -f yuv4mpegpipe - | NVEncC --y4m -i - -o "<outfilename.264>"
```

#### 从 FFmpeg 传递视频和音频

--> 使用 "nut" 格式来通过管道传递视频和音频
```Batchfile
ffmpeg -y -i "<input>" <options for ffmpeg> -codec:a copy -codec:v rawvideo -pix_fmt yuv420p -f nut - | NVEncC --avsw -i - --audio-codec aac -o "<outfilename.mp4>"
```

## 选项格式

```
-<short option name>, --<option name> <argument>

参数类型：
- none
- <int>    ... 整数
- <float>  ... 浮点数
- <string> ... 字符串

带有 [] {} 的参数是可选的。

--(no-)xxx
名为 --no-xxx 的选项将会有和 --xxx 相反的效果。
示例 1: --xxx: 启用 xxx → --no-xxx: 禁用 xxx
示例 2: --xxx: 禁用 xxx → --no-xxx: 启用 xxx
```

## 显示选项

### -h, -? --help

显示帮助

### -v, --version

显示 NVEncC 版本

### --option-list
显示选项列表

### --check-device

显示可以被 NVEnc 识别的 GPU 设备列表

### --check-hw [&lt;int&gt;]

检测指定设备是否可以运行 NVEnc。

若DeviceID未指定则默认检查DeviceID为0的设备。

### --check-features [&lt;int&gt;]

显示指定设备可用的特性。

若DeviceID未指定则默认检查DeviceID为0的设备。

### --check-environment

显示 NVEncC 识别的环境信息

### --check-codecs, --check-decoders, --check-encoders

显示可用的音频编解码器名

### --check-profiles &lt;string&gt;

显示指定的编码器可用的音频 profile 列表

### --check-formats

显示可用的输出格式

### --check-protocols

显示可用的协议

### --check-avdevices
获取可用的设备列表(通过libavdevice)

### --check-filters

显示可用的音频滤镜

### --check-avversion

显示 ffmpeg dll 版本号

## 基本编码选项

### -d, --device &lt;int&gt;

指定 NVEnc 使用的 deviceId。deviceID 可以通过 [--check-device](#--check-device) 获得。

如果未指定，且当前环境有多个可用的GPU，则将会根据以下条件自动选择

- 设备是否支持指定的编码
- 如果启用 --avhw，检查设备是否支持硬件解码该输入文件
- 如果启用交错编码，检查设备是否硬件是否支持
- 视频引擎占用率（Video Engine Utilization）更低的设备将会被优先选择
- GPU 占用率更低的设备将被优先选择
- 更新的 GPU 将会被优先选择
- 更多核心的 GPU 将会被优先选择

视频引擎和 GPU 占用率在 x64 版本中使用 [NVML library](https://developer.nvidia.com/nvidia-management-library-nvml) 获取, 在 x86 版本中通过 执行 nvidia-smi.exe 获取。

nvidia-smi 通常与驱动一起安装在 "C:\Program Files\NVIDIA Corporation\NVSMI\nvidia-smi.exe"。


### -c, --codec &lt;string&gt;

指定输出编码
 - h264 (默认)
 - hevc
 - av1
 - raw

  ```-c raw``` 将不进行编码并直接输出raw帧。raw帧的格式默认为y4m。可以通过```-f raw```指定raw帧的格式.

### -o, --output &lt;string&gt;

设置输出文件名，使用 "-" 进行管道输出。

### -i, --input &lt;string&gt;

设置输入文件名，使用 "-" 进行管道输入。

下表展示了 NVEnc 支持的读取器。当输入格式没有被指定时，将会根据输入文件后缀名确定。

**输入读取器自动选择**  

| 读取器 | 目标扩展名 |
|:---|:---|          
| Avisynth 读取器    | avs |
| VapourSynth 读取器 | vpy |
| avi 读取器         | avi |
| y4m 读取器         | y4m |
| raw 读取器         | yuv |
| avhw/avsw 读取器 | 其他 |

**读取器支持的色彩格式**  

| 读取器 | yuv420 | yuy2 | yuv422 | yuv444 | rgb24 | rgb32 |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| raw    |   ◎   |      |   ◎   |   ◎   |       |       |
| y4m    |   ◎   |      |   ◎   |   ◎   |       |       |
| avi    |   ○   |  ○  |        |        |   ○  |   ○  |
| avs    |   ◎   |  ○  |   ◎   |   ◎   |   ○  |   ○  |
| vpy    |   ◎   |      |   ◎   |   ◎   |       |       |
| avhw   |   □   |      |        |   ◇   |       |       |
| avsw   |   ◎   |      |   ◎   |   ◎   |   ○  |   ○  |

◎ ... 支持 8bit / 9bit / 10bit / 12bit / 14bit / 16bit  
◇ ... 支持 8bit / 10bit / 12bit  
□ ... 支持 8bit / 10bit  
○ ... 只支持 8 bit  
未标记 ... 不支持  

### --raw

将输入格式指定为未处理格式（Raw）。必须指定输入分辨率和帧率。

### --y4m

将输入格式指定为 y4m (YUV4MPEG2) 。

### --avi

使用 avi 读取器读取 avi 文件。

### --avs

使用 avs 读取器读取 Avisynth 脚本文件。

NVEncC 默认使用 UTF-8 编码格式读取文件, 因此当 Avisynth 脚本文件存在非ASCII字符时，应使用UTF-8格式保存

当使用系统默认的编码格式保存的脚本时, 例如使用 ANSI,则需要添加 "[--process-codepage](#--process-codepage-string-windows-os-only) os" 选项使 NVEncC 也使用操作系统的编码格式

### --vpy

使用 vpy 读取器读取 VapourSynth 脚本文件。

### --avsw

使用 avformat 和 ffmpeg 的软件解码器读取文件.

### --avhw

使用 avformat 和 cuvid 的硬件解码器。使用该模式可以提供最佳性能，因为该模式下整个编解码过程均在 GPU 运行。


**avhw reader 支持的编码**  

| Codecs | Status |
|:---|:---:|
| MPEG1      | ○ |
| MPEG2      | ○ |
| H.264/AVC  | ○ |
| H.265/HEVC | ○ |
| VP8        | × |
| VP9        | ○ |
| AV1        | ○ |
| VC-1       | ○ |
| WMV3/WMV9  | × |

○ ... 支持  
× ... 不支持

### --interlace &lt;string&gt;

指定 **输入** 的交错标志。

通过 [--vpp-deinterlace](#--vpp-deinterlace-string) 或 [--vpp-afs](#--vpp-afs-param1value1param2value2) 可以进行反交错。如果未指定反交错，则将会进行交错编码。

- progressive ... 逐行扫描
- tff ... 上场优先
- bff ... 下场优先
- auto ... 根据各帧自动判断 (仅使用[avhw](#--avhw)/[avsw](#--avsw)时有效)

### --video-track &lt;int&gt;
设置需要编码的视频轨编号。使用 avhw/avsw 读取器时有效。

 - 1 (默认)  最高分辨率视频轨
 - 2            次高分辨率视频轨
    ...
 - -1           最低分辨率视频轨
 - -2           次低分辨率视频轨
    ...

### --crop &lt;int&gt;,&lt;int&gt;,&lt;int&gt;,&lt;int&gt;
从左、上、右、下方向裁剪视频的像素数。

### --frames &lt;int&gt;
输入的帧的数量(注意，基于输入，而不是基于输出)

### --fps &lt;int&gt;/&lt;int&gt; or &lt;float&gt;
设置输入帧率，未处理格式（Raw）输入时需要。

### --input-res &lt;int&gt;x&lt;int&gt;
设置输入分辨率，未处理格式（Raw）输入时需要。

### --output-res &lt;int&gt;x&lt;int&gt;
设置输出分辨率。当与输入分辨率不同时，将会自动启用硬件/GPU缩放器。

未指定时将会与输入分辨率相同（不缩放）。

- **使用特殊值**
  - 0 ... 与输入保持一致
  - 宽高其中一个为负值   
    调整尺寸以适合另一侧，同时保持长宽比。将会选择一个能被该负数整除的值。

- **参数**
  - preserve_aspect_ratio=&lt;string&gt;  
    根据指定的宽度**或者**高度调整尺寸, 同时保持长宽比。
    - increase ... 在保持长宽比的同时调整为比指定分辨率大的分辨率（外接于指定分辨率）
    - decrease ... 在保持长宽比的同时调整为比指定分辨率小的分辨率（包含在指定分辨率内）

- 例子
  ```
  输入分辨率为1280x720...
  --output-res 1024x576 -> 正常更改分辨率为1024x576
  --output-res 960x0    -> 更改分辨率为 960x720 (0 将用与输入相同的 720 代替)
  --output-res 1920x-2  -> 更改分辨率为 1920x1080 (计算出保持纵横比的分辨率)
  
  --output-res 1440x1440,preserve_aspect_ratio=increase -> 更改分辨率为 2560x1440
  --output-res 1440x1440,preserve_aspect_ratio=decrease -> 更改分辨率为 1440x810
  ```

### --input-csp &lt;string&gt;
为--raw 设定输入的色彩空间,默认为yv12
```
  yv12, nv12, p010, yuv420p9le, yuv420p10le, yuv420p12le, yuv420p14le, yuv420p16le
  yuv422p, yuv422p9le, yuv422p10le, yuv422p12le, yuv422p14le, yuv422p16le
  yuv444p, yuv444p9le, yuv444p10le, yuv444p12le, yuv444p14le, yuv444p16le
```

## 编码模式选项
默认选择为 QVBR （固定质量）。

### --qvbr  &lt;float&gt;
以固定质量模式编码 (0.0-51.0, 0 = 自动)

等效于 --vbr 0 --vbr-quality &lt;float&gt;.

### --cbr &lt;int&gt;
### --vbr &lt;int&gt;
设置码率，单位kbps。

### --cqp &lt;int&gt; or &lt;int&gt;:&lt;int&gt;:&lt;int&gt;
将 QP 值设定为 &lt;I 帧&gt;:&lt;P 帧&gt;:&lt;B 帧&gt;。

一般情况下，推荐将 QP 值设置为 I &lt; P &lt; B 的组合。

## 其他适用于编码器的选项

### -u, --preset
编码质量预设，P1~P7选择从API v10.0开始支持
P1为最快，P7为质量最高
- default
- performance
- quality
- P1 (= performance)
- P2
- P3
- P4 (= default)
- P5
- P6
- P7 (= quality)

### --output-depth &lt;int&gt;
设置输出位深度。
- 8 ... 8 bits (默认)
- 10 ... 10 bits

### --output-csp &lt;string&gt;
设置输出时使用的色彩空间
- yuv420 (默认)
- yuv444

  :::注意
  没有添加yuv422和rgb色彩空间的计划
  :::

### --multipass &lt;string&gt;
多重编码模式，只在--vbr和--cbr模式下有效。 [API v10.0 以后支持]  

在1pass模式下，编码器估计宏块所需的QP并立即编码宏块。


在2pass模式中，在1pass对整个视频进行一次编码，确定视频不同位置所需比特量的分布。在2pass中，根据其结果进行宏块的编码。这可以更适当地对不同位置设置合适的码率，特别是在CBR模式中。


- none  
  1pass模式。 (最快)

- 2pass-quarter  
  以1/4大小的分辨率进行1pass。由此，能够捕捉较大的运动矢量并传递到2pass。

- 2pass-full  
  1pass/2pass都以全分辨率进行。虽然性能下降，但可以将更详细的分析信息传递给2pass。

### --lossless [H.264/HEVC]
进行无损输出。 (默认：关)

### --max-bitrate &lt;int&gt;
最大码率，单位kbps。

### --vbv-bufsize &lt;int&gt;
设定 vbv buffer 大小 (单位为kbps)。 (默认: 自动)

### --qp-init &lt;int&gt; or &lt;int&gt;:&lt;int&gt;:&lt;int&gt;

设置初始 QP 值为 &lt;I 帧&gt;:&lt;P 帧&gt;:&lt;B 帧&gt;。CQP模式下将会被忽略。

这些值将会被在编码开始时被应用。如果希望调节视频起始段的画面质量请设置该值。在 CBR/VBR 模式下有时会不稳定。

### --qp-min &lt;int&gt; or &lt;int&gt;:&lt;int&gt;:&lt;int&gt;

设置最小 QP 值为 &lt;I 帧&gt;:&lt;P 帧&gt;:&lt;B 帧&gt;。CQP模式下将会被忽略。

可被用于限制浪费在部分静止画面的码率。

### --qp-max &lt;int&gt; or &lt;int&gt;:&lt;int&gt;:&lt;int&gt;

设置最大 QP 值为 &lt;I 帧&gt;:&lt;P 帧&gt;:&lt;B 帧&gt;。CQP模式下将会被忽略。

可用于在视频的任何部分保持一定的图像质量，即使这样做可能超过指定的码率。

### --chroma-qp-offset &lt;int&gt;  [H.264/HEVC]
色度分量的QP偏移。 (默认: 0)

### --vbr-quality &lt;float&gt;

当使用 VBR 模式时设置输出质量。 (0.0-51.0, 0 表示自动)

### --dynamic-rc &lt;int&gt;:&lt;int&gt;:&lt;int&gt;&lt;int&gt;,&lt;param1&gt;=&lt;value1&gt;[,&lt;param2&gt;=&lt;value2&gt;],...  
改变"开始帧编号:结束帧编号"之间使用的码率控制方法。可以指定的参数有码率控制方法、最大码率和目标质量（vbr-quality）。

**必要参数**   
必须指定以下参数之一。
- [cqp](./NVEncC_Options.zh-cn.md#--cqp-int-or-intintint)=&lt;int&gt; or cqp=&lt;int&gt;:&lt;int&gt;:&lt;int&gt;  
- [cbr](./NVEncC_Options.zh-cn.md#--cbr-int)=&lt;int&gt;   
- [vbr](./NVEncC_Options.zh-cn.md#--vbr-int)=&lt;int&gt;   

**追加参数**
- [max-bitrate](./NVEncC_Options.zh-cn.md#--max-bitrate-int)=&lt;int&gt;  
- [vbr-quality](./NVEncC_Options.zh-cn.md#--vbr-quality-float)=&lt;float&gt;  
- [multipass](./NVEncC_Options.zh-cn.md#--multipass-string)=&lt;string&gt;  

```
例1: 3000-3999 帧使用vbr模式12000kbps编码、
     5000-5999 帧使用固定质量29.0编码、
     其他部分使用固定质量25.0编码。
  --vbr 0 --vbr-quality=25.0 --dynamic-rc 3000:3999,vbrhq=12000 --dynamic-rc 5000:5999,vbr=0,vbr-quality=29.0

例2: 3000 帧之前使用vbrhq模式6000kbps编码、
     3000 帧之后使用vbrhq模式12000kbps编码。
  --vbrhq 6000 --dynamic-rc start=3000,vbr=12000
```

### --lookahead &lt;int&gt;

使用 lookahead 并指定其目标范围的帧数。 (0 - 32) 

对于提高画面质量很有效，允许自适应插入 I 帧和 B帧。

### --no-i-adapt

当 lookahead 启用时禁用自适应 I 帧插入。

### --no-b-adapt

当 lookahead 启用时禁用自适应 B 帧插入。

### --strict-gop

强制固定 GOP 长度。

### --gop-len &lt;int&gt;

设置最大 GOP 长度。当 lookahead 未启用时，将总是使用该值。
(固定 GOP，非可变)

### -b, --bframes &lt;int&gt;

设置连续 B 帧数量。

### --ref &lt;int&gt;

设置参考距离。（最大16）

### --multiref-l0 &lt;int&gt; [H.264/HEVC]  
### --multiref-l1 &lt;int&gt; [H.264/HEVC]  
设置L0和L1的最大参考帧数量(上限为7) [API v9.1以上支持]

### --weightp

启用带权 P 帧。

### --nonrefp
自动插入 non-reference P 帧。

### --aq

在帧内启用自适应量化（Adaptive Quantization）。（默认：关）

### --aq-temporal

在帧间启用自适应量化（Adaptive Quantization）。（默认：关）

### --aq-strength &lt;int&gt;

指定自适应量化强度（Adaptive Quantization Strength）。(1 (弱) - 15 (强), 0 = 自动)

### --bref-mode &lt;string&gt; [H.264]
指定 B 帧参考模式。

- auto (默认)
- disabled
- each
  将每一 B 帧作为参考
- middle
  只有第 (B帧数量)/2 个B帧会被作为参考  

### --direct &lt;string&gt; [H.264]

指定 H.264 B Direct 模式.
- auto (默认)
- disabled
- spatial
- temporal

### --(no-)adapt-transform [H.264]
启用（或禁用）H.264 的自适应变换模式（Adaptive Transform Mode）。

### --hierarchial-p [H.264]
启用hierarchial P帧。

### --hierarchial-b [H.264]
启用hierarchial B帧。

### --temporal-layers &lt;int&gt; [H.264]
指定用于hierarchial编码的temporal layers数量。

### --mv-precision &lt;string&gt;
运动向量（Motion Vector）准确度 / 默认：自动。

- auto ... 自动
- Q-pel ... 1/4 像素精度 (高精确度)
- half-pel ... 1/2 像素精度
- full-pel ... 1 像素精度 (低精确度)


### --slices &lt;int&gt; [H.264/HEVC]
设定slices值.

### --cabac [H.264]
使用 CABAC (默认: 开)

### --cavlc [H.264]
使用 CAVLC (默认: 关)

### --bluray [H.264]
Bluray 的输出 (默认: 关)

### --(no-)deblock [H.264]
启用去色块（Deblock）滤镜。 (默认: 开)

### --cu-max &lt;int&gt; [HEVC]
### --cu-min &lt;int&gt; [HEVC]
设置最大和最小编码单元（Coding Unit, CU）大小。可以设置8、16、32。

**由于已知这些设置会降低画面质量，不推荐使用这些设置**

### --part-size-min &lt;int&gt; [AV1]
指定亮度分量的最小编码块大小 (默认: 0 = auto)
```
  0 (auto), 4, 8, 16, 32, 64
```

### --part-size-max &lt;int&gt; [AV1]
指定亮度分量的最大编码块大小 (默认: 0 = auto)
```
  0 (auto), 4, 8, 16, 32, 64
```

### --tile-columns &lt;int&gt; [AV1]
指定列方向的tile值 (默认: 0 = auto)

```
  0 (auto), 1, 2, 4, 8, 16, 32, 64
```

### --tile-rows &lt;int&gt; [AV1]
指定行方向的tile值 (默认: 0 = auto)

```
  0 (auto), 1, 2, 4, 8, 16, 32, 64
```

### --max-temporal-layers &lt;int&gt; [AV1]
指定用于hierarchial编码的temporal layers最大数量。

### --refs-forward &lt;int&gt; [AV1]
指定用于帧预测的前向参考帧的最大数目。(默认: 0 = auto)

可在1-4之间指定(Last, Last2, last3 and Golden)。注意，并非总是遵循此值。

### --refs-backward &lt;int&gt; [AV1]
指定用于帧预测的L1列表参考帧的最大数目。(默认: 0 = auto)

可在1-3之间指定(Backward, Altref2, Altref)。注意，并非总是遵循此值。


### --level &lt;string&gt;

设置编码器等级（Level）。如果未指定，将会自动设置。

```
h264: auto, 1, 1 b, 1.1, 1.2, 1.3, 2, 2.1, 2.2, 3, 3.1, 3.2, 4, 4.1, 4.2, 5, 5.1, 5.2
hevc: auto, 1, 2, 2.1, 3, 3.1, 4, 4.1, 5, 5.1, 5.2, 6, 6.1, 6.2
av1 :  auto, 2, 2.1, 2.2, 2.3, 3, 3.1, 3.2, 3.3, 4, 4.1, 4.2, 4.3, 5, 5.1, 5.2, 5.3, 6, 6.1, 6.2, 6.3, 7, 7.1, 7.2, 7.3
```

### --profile &lt;string&gt;

设置编码器 profile。如果未指定，将会自动设置。

```
h264:  auto, baseline, main, high, high444
hevc:  auto, main, main10, main444
av1 :  auto, main, high
```

### --tier &lt;string&gt;  [仅在 HEVC 下有效]

设置编码器 tier。
```
hevc:  main, high
av1 :  0, 1
```

### --sar &lt;int&gt;:&lt;int&gt;

设置 SAR 比例（Pixel Aspect Ratio）。

### --dar &lt;int&gt;:&lt;int&gt;

设置 DAR 比例 (Screen Aspect Ratio)。

### --colorrange &lt;string&gt;   
"--colorrange full"与"--fullrange"相同。   
指定为"auto"时、与输入文件保持一致。(仅当使用[avhw](#--avhw)/[avsw](#--avsw)时有效)
```
  limited, full, auto
```

### --videoformat &lt;string&gt;   
```
  undef, ntsc, component, pal, secam, mac
```
### --colormatrix &lt;string&gt;   
指定为"auto"时、与输入文件保持一致。(仅当使用[avhw](#--avhw)/[avsw](#--avsw)时有效)
```
  undef, auto, bt709, smpte170m, bt470bg, smpte240m, YCgCo, fcc, GBR, bt2020nc, bt2020c
```
### --colorprim &lt;string&gt;   
指定为"auto"时、与输入文件保持一致。(仅当使用[avhw](#--avhw)/[avsw](#--avsw)时有效)
```
  undef, auto, bt709, smpte170m, bt470m, bt470bg, smpte240m, film, bt2020
```
### --transfer &lt;string&gt;   
指定为"auto"时、与输入文件保持一致。(仅当使用[avhw](#--avhw)/[avsw](#--avsw)时有效)
```
  undef, auto, bt709, smpte170m, bt470m, bt470bg, smpte240m, linear,
  log100, log316, iec61966-2-4, bt1361e, iec61966-2-1,
  bt2020-10, bt2020-12, smpte2084, smpte428, arib-std-b67
```

### --chromaloc &lt;int&gt; or "auto"
指定为"auto"时、与输入文件保持一致。(仅当使用[avhw](#--avhw)/[avsw](#--avsw)时有效)

为输出流设置色度位置标志（Chroma Location Flag），从0到5。
 
默认: 0 = 未指定

### --max-cll &lt;int&gt;,&lt;int&gt; or "auto" [HEVC, AV1]

设置 MaxCLL 和 MaxFall，单位nits。如设定为copy则与输入文件保持一致。(仅当使用[avhw](#--avhw)/[avsw](#--avsw)时有效)

注意，此选项将自动启用 [--repeat-headers](#--repeat-headers)

```
例1：--max-cll 1000,300
例2: --max-cll copy  # copy values from source
```

### --master-display &lt;string&gt; or "auto" [HEVC, AV1]

设置 Mastering display 数据。如设定为copy则与输入文件保持一致。(仅当使用[avhw](#--avhw)/[avsw](#--avsw)时有效)

注意，此选项将自动启用 [--repeat-headers](#--repeat-headers)

```
例1: --master-display G(13250,34500)B(7500,3000)R(34000,16000)WP(15635,16450)L(10000000,1)
例2: --master-display copy # 从输入文件复制
```

### --atc-sei &lt;string&gt; or &lt;int&gt; [HEVC only]
设置 alternative transfer characteristics SEI，使用下述字符串或整数指定。
```
  undef, auto, bt709, smpte170m, bt470m, bt470bg, smpte240m, linear,
  log100, log316, iec61966-2-4, bt1361e, iec61966-2-1,
  bt2020-10, bt2020-12, smpte2084, smpte428, arib-std-b67
```  

### --dhdr10-info &lt;string&gt; [HEVC, AV1]
从指定JSON文件导入HDR10+的动态范围信息。额外依赖[hdr10plus_gen.exe](https://github.com/rigaya/hdr10plus_gen)。

### --dhdr10-info copy [HEVC, AV1]
从输入文件复制HDR10+的动态范围信息。

使用 avhw 读取文件时，需要使用时间戳对帧进行排序，因此无法取得时间戳的raw ES等输入文件无法使用。

这种情况下请使用 avsw 读取文件。

### --dolby-vision-profile &lt;float&gt;
指定以杜比视界格式输出
```
5.0, 8.1, 8.2, 8.4
```

### --dolby-vision-rpu &lt;string&gt;
将指定杜比视界的RPU文件中包含的metadata插入输出文件。


在当前，使用此选项输出的视频文件不会由MediaInfo检测到Dolby Vision信息。为了使MediaInfo可以检测Dolby Vision信息，需要使用[tsMuxeR](https://github.com/justdan96/tsMuxer/releases) (nightly)重新封装。

### --aud [H.264/HEVC]
插入Access Unit Delimiter NAL。

### --repeat-headers
为每个 IDR frame输出 VPS, SPS and PPS。

### --pic-struct
插入 Picture Timing SEI。

### --split-enc &lt;string&gt;
- **参数**
  - auto  
    禁用split frame forced 模式，启用 auto 模式。

  - auto_forced  
    并启用split frame forced模式，由驱动自动选择最佳的strips数量

  - forced_2  
    指定使用 2-strip split frame 编码(如果NVENC数量大于1，则使用1-strip编码)

  - forced_3  
    指定使用 3-strip split frame 编码(如果NVENC数量大于2，则使用其他数量的strip编码)
    
  - disable  
    split frame 的 forced 模式和 auto 模式均禁用。

### --ssim  
计算编码结果的SSIM。

### --psnr   
计算编码结果的PSNR。

### --vmaf [&lt;param1&gt;=&lt;value1&gt;][,&lt;param2&gt;=&lt;value2&gt;],...
计算编码结果的VMAF。需要注意的是VMAF通过libvmaf在cpu上计算。这一过程很可能成为性能瓶颈，导致编码变慢。

目前仅适用于Windows x64

- **参数**

  - model=&lt;string&gt;  
    设置libvmaf的内部模型版本或外部模型文件路径。默认为内部的"vmaf_v0.6.1"。
    当使用模型文件，从[连接](https://github.com/Netflix/vmaf/tree/master/model)下载json格式模型，并使用此选项设定路径。

  - threads=&lt;int&gt;  (默认: 0)  
    使用多少进程用于计算VMAF。默认使用所有物理核。

  - subsample=&lt;int&gt;  (默认: 1)  
    指定计算VMAF的帧子采样间隔。

  - phone_model=&lt;bool&gt;  (默认: false)  
    使用phone模型，这可以产生更高的vmaf
    
  - enable_transform=&lt;bool&gt;  (默认: false)  
    计算vmaf时启用transform
    

```
例子: --vmaf model=vmaf_v0.6.1.json
```

## 输入输出 / 音频 / 字幕设置 

### --input-analyze &lt;int&gt;

设置 libav 分析视频时使用的视频长度，单位为秒。默认为5秒。如果音频 / 字幕轨等没有被正确检测，尝试增加该值（如60）。

### --input-probesize &lt;int&gt;
指定libav读取时分析的最大大小(单位为byte)


### --trim &lt;int&gt;:&lt;int&gt;[,&lt;int&gt;:&lt;int&gt;][,&lt;int&gt;:&lt;int&gt;]...

只编码指定范围内的帧。

```
示例1: --trim 0:1000,2000:3000    (编码第0~1000帧和2000~3000帧)
示例2: --trim 2000:0              (编码第2000帧到最后)
```

### --seek [&lt;int&gt;:][&lt;int&gt;:]&lt;int&gt;[.&lt;int&gt;]

格式为 hh:mm:ss.ms。"hh" 或 "mm" 可以省略。转码将从这一指定的视频时间开始。

与[--trim](#--trim-intintintintintint)相比，这一设置不那么精确但更快。如果你需要精确，请使用[--trim](#--trim-intintintintintint)。

```
示例 1: --seek 0:01:15.400
示例 2: --seek 1:15.4
示例 3: --seek 75.4
```

### --seekto [&lt;int&gt;:][&lt;int&gt;:]&lt;int&gt;[.&lt;int&gt;]
格式为 hh:mm:ss.ms。"hh" 或 "mm" 可以省略。设定编码的结束时间。

这可能不够精确，所以如果需要精确的帧数进行编码，请使用[--trim](#--trim-intintintintintint)


```
示例 1: --seekto 0:01:15.400
示例 2: --seekto 1:15.4
示例 3: --seekto 75.4
```

### --input-format &lt;string&gt;
为 avhw / avsw 读取器指定输入格式。

### -f, --output-format &lt;string&gt;

- 对于常规编码器

  为混流器指定输出格式。

  由于输出格式可以通过输出文件的扩展名自动确定，通常情况下无需指定，但你可以使用该选项强行指定输出格式。

  可用的格式可以通过[--check-formats](#--check-formats)查询。

- 对于 raw 输出 (使用```-c raw```)

  指定输出格式为raw帧

  - 参数
    - y4m (默认)
    - raw

### --video-track &lt;int&gt;

选择要编码的视频轨道。仅在使用avsw/avhw reader时有效。

 - 1 (默认)  最高分辨率的视频轨道
 - 2            第二高分辨率的视频轨道
    ...
 - -1           最低分辨率的视频轨道
 - -2           第二低分辨率的视频轨道
    ...
    
### --video-streamid &lt;int&gt;
使用stream id选择要编码的视频轨道。

### --video-tag  &lt;string&gt;
指定视频标签。
```
 -o test.mp4 -c hevc --video-tag hvc1
```

### --video-metadata &lt;string&gt; or &lt;string&gt;=&lt;string&gt;
设定视频轨道的metadata
  - copy  ... 如果可行，从输入复制metadata
  - clear ... 不复制metadata (默认)


```
例1: 从输入文件复制metadata
--video-metadata 1?copy
  
例2: 清空输入文件的metadata
--video-metadata 1?clear
  
例3: 设定metadata
--video-metadata 1?title="video title" --video-metadata 1?language=jpn
 ```

### --audio-copy [&lt;int&gt;[,&lt;int&gt;]...]

将音频轨复制到输出文件。仅当使用 avhw / avsw 读取器时有效。

如果工作异常，尝试使用[--audio-codec](#--audio-codec-intstring)编码音频，它更加稳定。

你也可以指定抽取并复制的音频轨（1,2,...）。

```
示例: 复制全部音频轨
--audio-copy

示例: 抽取并复制#1和#2音频轨
--audio-copy 1,2
```

### --audio-codec [[&lt;int/string&gt;?]&lt;string&gt;[:&lt;string&gt;=&lt;string&gt;[,&lt;string&gt;=&lt;string&gt;]...]...]

使用指定的编码器编码音频轨。如果没有设定编码器，将会自动使用最合适的编码器。可用的编码器可以通过[--check-encoders](#--check-codecs---check-decoders---check-encoders)查询。

你也可以使用[&lt;int&gt;]选择要抽取的音频轨（1,2,...），或者使用[&lt;string&gt;]选择对应语言的音频轨。

你可以在":"后指定编码器参数，在"#"后指定解码器参数。
```
示例 1: 把所有音频轨编码为mp3
--audio-codec libmp3lame

示例 2: 把第二根音频轨编码为aac
--audio-codec 2?aac

示例 3: 将英语音频轨道编码为acc
--audio-codec eng?aac
  
示例 4: 将英语和汉语音频轨道编码为acc
--audio-codec eng?aac --audio-codec chs?aac

示例 5: 为 "aac_coder" 添加 "twoloop" 参数可以提升低码率下的音频质量。
--audio-codec aac:aac_coder=twoloop
```

### --audio-bitrate [&lt;int/string&gt;?]&lt;int&gt;
指定音频码率，单位为kbps。

你可以使用[&lt;int&gt;]选择对应的音频轨（1,2,...），或者使用[&lt;string&gt;]选择对应语言的音频轨。

```
示例 1: --audio-bitrate 192 (设置音频轨码率为 192kbps)
示例 2: --audio-bitrate 2?256 (设置第二根音频轨的码率为 256kbps)
```

### --audio-profile [&lt;int/string&gt;?]&lt;string&gt;
指定音频编码器的profile。

你可以使用[&lt;int&gt;]选择对应的音频轨（1,2,...），或者使用[&lt;string&gt;]选择对应语言的音频轨。

### --audio-stream [&lt;int&gt;?][&lt;string1&gt;][:&lt;string2&gt;]

分离或合并音频声道。

该选项指定的音频轨将总是被编码（不能使用复制）。

使用半角逗号（","）分隔，你可以给同一输入音频轨生成多个输出音频轨。

**格式**

使用&lt;int&gt;指定要处理的轨道。

使用&lt;string1&gt;指定要使用的声道，如果不指定，则使用全部输入声道。

使用&lt;string2&gt;指定输出声道格式。如果不指定，&lt;string1&gt;指定的全部声道将会被使用。

```
示例 1: --audio-stream FR,FL
把双声道音频轨的左声道和右声道分离到两个单声道音频轨。

示例 2: --audio-stream :stereo
把任何音频轨转换为立体声。

示例 3: --audio-stream 2?5.1,5.1:stereo
把输入文件的第二根5.1ch音频轨编码为5.1ch，另一个立体声下混（downmixed）音频轨道将从同一源音频轨道生成。
```

**可用符号**
```
mono       = FC
stereo     = FL + FR
2.1        = FL + FR + LFE
3.0        = FL + FR + FC
3.0(back)  = FL + FR + BC
3.1        = FL + FR + FC + LFE
4.0        = FL + FR
4.0        = FL + FR + FC + BC
quad       = FL + FR + BL + BR
quad(side) = FL + FR + SL + SR
5.0        = FL + FR + FC + SL + SR
5.1        = FL + FR + FC + LFE + SL + SR
6.0        = FL + FR + FC + BC + SL + SR
6.0(front) = FL + FR + FLC + FRC + SL + SR
hexagonal  = FL + FR + FC + BL + BR + BC
6.1        = FL + FR + FC + LFE + BC + SL + SR
6.1(front) = FL + FR + LFE + FLC + FRC + SL + SR
7.0        = FL + FR + FC + BL + BR + SL + SR
7.0(front) = FL + FR + FC + FLC + FRC + SL + SR
7.1        = FL + FR + FC + LFE + BL + BR + SL + SR
7.1(wide)  = FL + FR + FC + LFE + FLC + FRC + SL + SR
```

### --audio-samplerate [&lt;int/string&gt;?]&lt;int&gt;

设定音频采样率，单位Hz。

你可以使用[&lt;int&gt;]选择对应的音频轨（1,2,...），或者使用[&lt;string&gt;]选择对应语言的音频轨。

```
示例 1: --audio-bitrate 44100 (把音频转换为 44100Hz)
示例 2: --audio-bitrate 2?22050 (把第二根音频轨的音频转换为 22050Hz)
```

### --audio-resampler &lt;string&gt;

指定用于混合音频声道和采样率转换的引擎。

- swr ... swresampler (默认)
- soxr ... sox resampler (libsoxr)

### --audio-delay [&lt;int/string&gt;?]&lt;float&gt; 
设置音频延迟，单位ms。

你可以使用[&lt;int&gt;]选择对应的音频轨（1,2,...），或者使用[&lt;string&gt;]选择对应语言的音频轨。

### --audio-file [&lt;int/string&gt;?][&lt;string&gt;]&lt;string&gt;

把音频轨抽取到指定的路径。输出格式由输出文件后缀名自动确定。仅当使用 avhw / avsw 读取器时有效。

你可以使用[&lt;int&gt;]选择对应的音频轨（1,2,...），或者使用[&lt;string&gt;]选择对应语言的音频轨。

```
示例: 把第二根音频轨的音频抽取到"test_out2.aac"
--audio-file 2?"test_out2.aac"
```

[&lt;string&gt;] 允许你指定输出格式.
```
示例: 不带后缀名的情况下以 adts 格式输出
--audio-file 2?adts:"test_out2"  
```

### --audio-filter [&lt;int&gt;?]&lt;string&gt;

为音频轨应用滤镜。滤镜可以到[link](https://ffmpeg.org/ffmpeg-filters.html#Audio-Filters)选择。

你也可以指定抽取并应用滤镜的音频轨（1,2,...）。

```
示例 1: --audio-filter volume=0.2  (降低音量)
示例 2: --audio-filter 2?volume=-4db (降低第二根音频轨的音量)
```

### --audio-disposition [&lt;int/string&gt;?]&lt;string&gt;[,&lt;string&gt;][]...
指定默认语音。

你可以使用[&lt;int&gt;]选择对应的音频轨（1,2,...），或者使用[&lt;string&gt;]选择对应语言的音频轨。

- 可用于设置默认倾向的参数列表
  ```
  default
  dub
  original
  comment
  lyrics
  karaoke
  forced
  hearing_impaired
  visual_impaired
  clean_effects
  attached_pic
  captions
  descriptions
  dependent
  metadata
  copy
  ```

```
例子:
--audio-disposition 2?default,forced
```

### --audio-metadata [&lt;int/string&gt;?]&lt;string&gt; or [&lt;int/string&gt;?]&lt;string&gt;=&lt;string&gt;

设定音频轨道的metadata
  - copy  ... 如果可行，从输入复制metadata (默认)
  - clear ... 不复制metadata

你可以使用[&lt;int&gt;]选择对应的音频轨（1,2,...），或者使用[&lt;string&gt;]选择对应语言的音频轨。

```
例子 1: 从输入文件复制metadata
--audio-metadata 1?copy
  
例子 2: 清空输入文件的metadata
--audio-metadata 1?clear
  
例子 3: 设定metadata
--audio-metadata 1?title="audio title" --audio-metadata 1?language=jpn
```

### --audio-bsf [&lt;int/string&gt;?]&lt;string&gt;
将[bitstream filter](https://ffmpeg.org/ffmpeg-bitstream-filters.html)应用于音频轨道。

### --audio-ignore-decode-error &lt;int&gt;

忽略持续的音频解码错误，在阈值允许范围内继续转码。无法被正确的解码的音频部分将会使用空白音频替代。

默认值为10。
```
Example1: 五个连续音频解码错误后退出转码
--audio-ignore-decode-error 5

Example2: 任何解码错误后退出转码
--audio-ignore-decode-error 0
```

### --audio-source &lt;string&gt;[:{&lt;int&gt;?}[;&lt;param1&gt;=&lt;value1&gt;...]/[]...]

混流指定的外部音频文件。

- **文件参数**
  - format=&lt;string&gt;  
    指定输入文件的格式。

  - input_opt=&lt;string&gt;  
    指定输入文件的选项。

**轨道参数**

  - copy  
    直接复制音频轨。

  - codec=&lt;string&gt;  
    使用指定编码器编码音频轨。

  - profile=&lt;string&gt;  
    指定编码音频时使用的profile。

  - bitrate=&lt;int&gt;  
    指定音频编码时使用的码率，单位kbps。

  - samplerate=&lt;int&gt;  
    指定音频编码时使用的采样率，单位Hz。

  - delay=&lt;int&gt;  
    指定音频延迟 (单位为毫秒)。
  
  - dec_prm=&lt;string&gt;  
    指定音频解码参数。

  - enc_prm=&lt;string&gt;  
    指定音频编码参数。

  - filter=&lt;string&gt;  
    指定音频编码滤镜。

  - disposition=&lt;string&gt;  
    指定默认音频。
    
  - metadata=&lt;string1&gt;=&lt;string2&gt;  
    指定音频轨道的metadata。
    
  - bsf=&lt;string&gt;  
    指定用于音频轨道的bitstream过滤器。

```
例1: --audio-source "<audio_file>:copy"
例2: --audio-source "<audio_file>:codec=aac"
例3: --audio-source "<audio_file>:1?codec=aac;bitrate=256/2?codec=aac;bitrate=192;metadata=language=chs;disposition=default,forced"
例4: --audio-source "hw:1:format=alsa/codec=aac;bitrate=256"
```

### --chapter &lt;string&gt;

使用章节文件设置章节信息。章节文件可以是 nero、apple 或 matroska 格式。无法与 --chapter-copy 同时使用。


nero格式
```
CHAPTER01=00:00:39.706
CHAPTER01NAME=chapter-1
CHAPTER02=00:01:09.703
CHAPTER02NAME=chapter-2
CHAPTER03=00:01:28.288
CHAPTER03NAME=chapter-3
```

apple格式 (utf-8)
```
<?xml version="1.0" encoding="UTF-8" ?>
  <TextStream version="1.1">
   <TextStreamHeader>
    <TextSampleDescription>
    </TextSampleDescription>
  </TextStreamHeader>
  <TextSample sampleTime="00:00:39.706">chapter-1</TextSample>
  <TextSample sampleTime="00:01:09.703">chapter-2</TextSample>
  <TextSample sampleTime="00:01:28.288">chapter-3</TextSample>
  <TextSample sampleTime="00:01:28.289" text="" />
</TextStream>
```

matroska格式 (utf-8)

[其他例子](https://github.com/nmaier/mkvtoolnix/blob/master/examples/example-chapters-1.xml)
```
<?xml version="1.0" encoding="UTF-8"?>
<Chapters>
  <EditionEntry>
    <ChapterAtom>
      <ChapterTimeStart>00:00:00.000</ChapterTimeStart>
      <ChapterDisplay>
        <ChapterString>chapter-0</ChapterString>
      </ChapterDisplay>
    </ChapterAtom>
    <ChapterAtom>
      <ChapterTimeStart>00:00:39.706</ChapterTimeStart>
      <ChapterDisplay>
        <ChapterString>chapter-1</ChapterString>
      </ChapterDisplay>
    </ChapterAtom>
    <ChapterAtom>
      <ChapterTimeStart>00:01:09.703</ChapterTimeStart>
      <ChapterDisplay>
        <ChapterString>chapter-2</ChapterString>
      </ChapterDisplay>
    </ChapterAtom>
    <ChapterAtom>
      <ChapterTimeStart>00:01:28.288</ChapterTimeStart>
      <ChapterTimeEnd>00:01:28.289</ChapterTimeEnd>
      <ChapterDisplay>
        <ChapterString>chapter-3</ChapterString>
      </ChapterDisplay>
    </ChapterAtom>
  </EditionEntry>
</Chapters>
```

### --chapter-copy

从输入文件复制章节信息。

### --chapter-no-trim

读取章节时不应用--trim

### --key-on-chapter

在章节分割处设置关键帧。

### --keyfile &lt;string&gt;

由文件指定关键帧位置（从0,1,2,...起）。文件应一行一个帧序号。

### --sub-source &lt;string&gt;[:{&lt;int&gt;?}[;&lt;param1&gt;=&lt;value1&gt;...]/[]...]
读取指定字幕文件并混流。

- **文件参数**
  - format=&lt;string&gt;  
    指定输入文件的格式。

  - input_opt=&lt;string&gt;  
    指定输入文件的选项。

- **轨道参数**
  - disposition=&lt;string&gt;  
    设置默认字幕
    
  - metadata=&lt;string1&gt;=&lt;string2&gt;  
    指定字幕轨的metadata

    
  - bsf=&lt;string&gt;  
    指定用于字幕轨的bitstream过滤器。
  
```
例1: --sub-source "<sub_file>"
例2: --sub-source "<sub_file>:disposition=default,forced;metadata=language=chs"
  ```

### --sub-copy [&lt;int&gt;[,&lt;int&gt;]...]

从输入文件复制字幕轨。仅当使用 avhw / avsw 读取器时有效。

你也可以指定需要抽取并复制的字幕轨（1,2,...）。

支持 PGS / srt / txt / ttxt 格式字幕。

```
示例1: 复制所有字幕轨
--sub-copy
示例2: 复制第一、第二根字幕轨
--sub-copy 1,2
示例3: 复制标记了英语和汉语的字幕轨
--sub-copy eng,chs
```

### --sub-disposition [&lt;int/string&gt;?]&lt;string&gt;
将选中的字幕轨设为默认

- 可用于设置默认倾向的参数列表
  ```
   default
   dub
   original
   comment
   lyrics
   karaoke
   forced
   hearing_impaired
   visual_impaired
   clean_effects
   attached_pic
   captions
   descriptions
   dependent
   metadata
   copy
  ```


### --sub-metadata [&lt;int/string&gt;?]&lt;string&gt; or [&lt;int/string&gt;?]&lt;string&gt;=&lt;string&gt;
指定字幕轨的metadata
  - copy  ... 如果可行，从输入复制metadata (默认)
  - clear ... 不复制metadata

```
例1: 从输入文件复制metadata
--sub-metadata 1?copy
  
例2: 清空输入文件的metadata
--sub-metadata 1?clear
  
例3: 设定metadata
--sub-metadata 1?title="subtitle title" --sub-metadata 1?language=jpn
```

### --sub-bsf [&lt;int/string&gt;?]&lt;string&gt;
将[bitstream filter](https://ffmpeg.org/ffmpeg-bitstream-filters.html)应用于字幕轨。

### --data-copy [&lt;int&gt;[,&lt;int&gt;]...]   
复制 Data 流，使用avhw/avsw时有效。

### --attachment-copy [&lt;int&gt;[,&lt;int&gt;]...]   
复制输入文件的附加文件流，使用avhw/avsw时有效。

### --attachment-source &lt;string&gt;[:{&lt;int&gt;?}[;&lt;param1&gt;=&lt;value1&gt;]...]...
从指定文件读取为附加文件并混流。
- **参数** 
  - metadata=&lt;string1&gt;=&lt;string2&gt;  
    指定附加文件的metadata，必须要设定mimetype
  
```
例1: --attachment-source <png_file>:metadata=mimetype=image/png
例2: --attachment-source <font_file>:metadata=mimetype=application/x-truetype-font
```


### --input-option &lt;string1&gt;:&lt;string2&gt;   
使用 avsw/avhw 读取视频时透传的参数。&lt;string1&gt;为参数名，&lt;string2&gt;为参数值。

```
示例: 读取BD的Playlist 1
-i bluray:D:\ --input-option playlist:1
```

### -m, --mux-option &lt;string1&gt;:&lt;string2&gt;

为混流器传递附加参数。用&lt;string1&gt;指定参数名，用&lt;string2&gt;指定参数值。

```
示例: 输出 HLS
-i <input> -o test.m3u8 -f hls -m hls_time:5 -m hls_segment_filename:test_%03d.ts --gop-len 30

示例: 在没有设定为“default”的字幕轨的情况下，抑制自动赋予"default"（仅mkv）
  -m default_mode:infer_no_subs
```

### --metadata &lt;string&gt; or &lt;string&gt;=&lt;string&gt;
为输出文件设定全局metadata
  - copy  ... 如果可行，从输入复制metadata (默认)
  - clear ... 不复制metadata

```
例1: 从输入文件复制metadata
--metadata copy
  
例2: 清空输入文件的metadata
--metadata clear
  
例3: 设定metadata
--metadata title="video title" --metadata language=jpn
```

### --avsync &lt;string&gt;
  - cfr (默认)  
    输入文件将会被认为是固定帧率，输入的PTS（Presentation Time Stamp）将不会被检查。

  - forcecfr  
    检查输入文件的PTS（Presentation Time Stamp），重复或者移除帧来保持固定帧率，以维持与音频的同步。无法和 --trim 一起使用。

  - vfr  
    遵循输入文件的时间戳并启用可变帧率输出。仅当使用 avsw/avhw 读取器时有效。

### --timecode [&lt;string&gt;]  
将时间码文件保存到指定路径，如果未设置路径，将保存为"&lt;output file path&gt;.timecode.txt"。


### --tcfile-in &lt;string&gt;  
读取timecode文件从而设置输入帧的时间戳，适用于avhw以外的读取器

### --timebase &lt;int&gt;/&lt;int&gt;  
设定时间刻度。也用于读取timecode文件时的时间刻度。

### --input-hevc-bsf &lt;string&gt;  
对于硬件解码器的输入，切换hevc bitstream过滤器。(用于调试目的)

- 参数

  - internal  
    使用内部实现。 (默认)

  - libavcodec  
    使用 hevc_mp4toannexb bitstream 过滤器.

### --allow-other-negative-pts  
允许语音字幕有着负timestamp。原则上只用于调试。

## Vpp 设置

用于在编码前添加过滤的选项。

### Vpp 过滤顺序

vpp过滤器的应用顺序是固定的，与命令行的顺序无关，将按以下顺序应用:

- [--vpp-deinterlace](#--vpp-deinterlace-string)
- [--vpp-colorspace](#--vpp-colorspace-param1value1param2value2)
- [--vpp-rff](#--vpp-rff)
- [--vpp-delogo](#--vpp-delogo-stringparam1value1param2value2)
- [--vpp-afs](#--vpp-afs-param1value1param2value2)
- [--vpp-nnedi](#--vpp-nnedi-param1value1param2value2)
- [--vpp-yadif](#--vpp-yadif-param1value1)
- [--vpp-decimate](#--vpp-decimate-param1value1param2value2)
- [--vpp-mpdecimate](#--vpp-mpdecimate-param1value1param2value2)
- [--vpp-select-every](#--vpp-select-every-intparam1int)
- [--vpp-transform/rotate](#--vpp-rotate-int)
- [--vpp-convolution3d](#--vpp-convolution3d-param1value1param2value2)
- [--vpp-nvvfx-denoise](#--vpp-nvvfx-denoise-param1value1param2value2)
- [--vpp-nvvfx-artifact-reduction](#--vpp-nvvfx-artifact-reduction-param1value1param2value2)
- [--vpp-smooth](#--vpp-smooth-param1value1param2value2)
- [--vpp-denoise-dct](#--vpp-denoise-dct-param1value1param2value2)
- [--vpp-knn](#--vpp-knn-param1value1param2value2)
- [--vpp-pmd](#--vpp-pmd-param1value1param2value2)
- [--vpp-gauss](#--vpp-gauss-int)
- [--vpp-subburn](#--vpp-subburn-param1value1param2value2)
- [--vpp-resize](#--vpp-resize-string-or-param1value1param2value2)
- [--vpp-unsharp](#--vpp-unsharp-param1value1param2value2)
- [--vpp-edgelevel](#--vpp-edgelevel-param1value1param2value2)
- [--vpp-warpsharp](#--vpp-warpsharp-param1value1param2value2)
- [--vpp-tweak](#--vpp-tweak-param1value1param2value2)
- [--vpp-deband](#--vpp-deband-param1value1param2value2)
- [--vpp-padding](#--vpp-pad-intintintint)
- [--vpp-overlay](#--vpp-overlay-param1value1param2value2)

### --vpp-colorspace [&lt;param1&gt;=&lt;value1&gt;][,&lt;param2&gt;=&lt;value2&gt;],...  
对视频进行颜色空间变换。仅x64版可用。

当参数设置为"input"时，将参考输入文件的色彩空间。(仅当使用avhw/avsw时有效)

- **参数**
  - matrix=&lt;from&gt;:&lt;to&gt;  
    
  ```
    bt709, smpte170m, bt470bg, smpte240m, YCgCo, fcc, GBR, bt2020nc, bt2020c, auto
  ```
  
  - colorprim=&lt;from&gt;:&lt;to&gt;  
  ```
    bt709, smpte170m, bt470m, bt470bg, smpte240m, film, bt2020, auto
  ```
  
  - transfer=&lt;from&gt;:&lt;to&gt;  
  ```
    bt709, smpte170m, bt470m, bt470bg, smpte240m, linear,
    log100, log316, iec61966-2-4, iec61966-2-1,
    bt2020-10, bt2020-12, smpte2084, arib-std-b67, auto
  ```
  
  - range=&lt;from&gt;:&lt;to&gt;  
  ```
    limited, full, auto
  ```
  
  - lut3d=&lt;string&gt;  
    对输入的文件应用3D LUT，目前只支持.cube文件
    
  - lut3d_interp=&lt;string&gt;  
    ```
    nearest, trilinear, tetrahedral, pyramid, prism
    ```
  
  - hdr2sdr=&lt;string&gt;   
    指定tone-mapping算法将HDR10转换成SDR
  
    - none (默认)  
      禁止进行hdr到sdr的转换
    
    - hable  
      试图保留明亮和黑暗的细节，但画面会很暗。
      可为下面的hable色调映射函数指定参数(a,b,c,d,e,f)。

      hable(x) = ( (x * (a*x + c*b) + d*e) / (x * (a*x + b) + d*f) ) - e/f  
      output = hable( input ) / hable( (source_peak / ldr_nits) )

      默认值：a=0.22,b=0.3,c=0.1,d=0.2,e=0.01,f=0.3
  
    - mobius  
      能够尽量保留画面的亮度和对比度，但可能损坏亮部的细节。
      - transition=&lt;float&gt;  (默认: 0.3)  
        由线性变换改用 mobius 色调映射的临界点。
      - peak=&lt;float&gt;  (默认: 1.0)  
        参考峰值亮度。
    
    - reinhard  
      - contrast=&lt;float&gt;  (默认: 0.5)  
        局部对比度系数。
      - peak=&lt;float&gt;  (默认: 1.0)  
        参考峰值亮度。
        
    - bt2390  
      BT.2390中规定的色调映射(EETF)
  
  - source_peak=&lt;float&gt;  (默认: 1000.0)  
  
  - ldr_nits=&lt;float&gt;  (默认: 100.0)  
    hdr2sdr的目标亮度
    
  - desat_base=&lt;float&gt;  (默认: 0.18)  
    在hdr2sdr中使用的desaturation curve的偏移。
  
  - desat_strength=&lt;float&gt;  (默认: 0.75)  
    hdr2sr中使用的desaturation curve强度。
    0.0将禁用desaturation，1.0将使过于明亮的颜色趋向于白色。
  
  - desat_exp=&lt;float&gt;  (默认: 1.5)  
    hdr2sdr中使用的desaturation curve的指数，控制从多少亮度开始进行处理。
    较低的值表示更积极地进行处理。

```
例1: BT.601 -> BT.709 的变换
--vpp-colorspace matrix=smpte170m:bt709
  
例2: 使用 hdr2sdr (hable色调映射)
--vpp-colorspace hdr2sdr=hable,source_peak=1000.0,ldr_nits=100.0
  
例3: 使用 hdr2sdr (hable色调映射) 并设定coefs (例子中的参数是默认参数)
--vpp-colorspace hdr2sdr=hable,source_peak=1000.0,ldr_nits=100.0,a=0.22,b=0.3,c=0.1,d=0.2,e=0.01,f=0.3
  
例4: 使用 lut3d
--vpp-colorspace lut3d="example.cube",lut3d_interp=trilinear
```

### --vpp-delogo &lt;string&gt;[,&lt;param1&gt;=&lt;value1&gt;][,&lt;param2&gt;=&lt;value2&gt;],...
指定需要消除的Logo的Logo文件及设置。Logo文件支持".lgd"、".ldp"、".ldp2"格式。

- **参数**
  - select=&lt;string&gt;  
    对于logo包，通过以下任一项指定要使用的logo:
    - Logo 名称
    - 编号 (1, 2, ...)
    - 通过 ini 文件自动选择
      ```
       [LOGO_AUTO_SELECT]
       logo<num>=<pattern>,<logo name>
      ```
      
      例子:
      ```ini
      [LOGO_AUTO_SELECT]
      logo1= (NHK-G).,NHK総合 1440x1080
      logo2= (NHK-E).,NHK-E 1440x1080
      logo3= (MX).,TOKYO MX 1 1440x1080
      logo4= (CTC).,チバテレビ 1440x1080
      logo5= (NTV).,日本テレビ 1440x1080
      logo6= (TBS).,TBS 1440x1088
      logo7= (TX).,TV東京 50th 1440x1080
      logo8= (CX).,フジテレビ 1440x1088
      logo9= (BSP).,NHK BSP v3 1920x1080
      logo10= (BS4).,BS日テレ 1920x1080
      logo11= (BSA).,BS朝日 1920x1080
      logo12= (BS-TBS).,BS-TBS 1920x1080
      logo13= (BSJ).,BS Japan 1920x1080
      logo14= (BS11).,BS11 1920x1080 v3
      ```
  
  
  - pos &lt;int&gt;:&lt;int&gt;
    在x:y方向上以1/4像素精度调整Logo位置。 
  
  - depth &lt;int&gt;
    调整Logo透明度。(默认: 128) 
  
  - y=&lt;int&gt;  
  - cb=&lt;int&gt;  
  - cr=&lt;int&gt;  
    调整Logo各颜色成分。
  
  - auto_fade=&lt;bool&gt;  
    根据Logo的实际深度自动调整淡入度值 (默认: false) 
    
  - auto_nr=&lt;bool&gt;  
    动态调整降噪强度 (默认: false)  
  
  - nr_area=&lt;int&gt;  
    水印附近的降噪范围. (默认: 0 (关闭), 0 - 3)  
  
  - nr_value=&lt;int&gt;  
    水印附近的降噪强度. (默认: 0 (关闭), 0 - 4)
  
  - log=&lt;bool&gt;  
    使用auto_fade、auto_nr时，输出淡入淡出值变化日志。

```
例子:
--vpp-delogo logodata.ldp2,select=delogo.auf.ini,auto_fade=true,auto_nr=true,nr_value=3,nr_area=1,log=true
```

### --vpp-rff
RFF（Reflect the Repeat Field）标记。可以解决由于 RFF 引发的 avsync 错误。仅当使用[--avhw](#--avhw-string)时有效。

2或以上的值不被支持（仅支持 rff = 1）。同时，无法与[--trim](#--trim-intintintintintint)和[--vpp-deinterlace](#--vpp-deinterlace-string)一起使用。

### --vpp-deinterlace &lt;string&gt;

激活硬件反交错器。仅当使用[--avhw](#--avhw-string)(硬件解码)时有效，并且需要为[--interlace](#--interlace-string)选项指定tff或bff。

- none ... 不反交错 (默认)
- normal ... 标准 60i → 30p 反交错.
- adaptive ... 与 normal 相同
- bob ... 60i → 60p 交错.

对于 IT(inverse telecine), 使用 [--vpp-afs](#--vpp-afs-param1value1param2value2).

### --vpp-afs [&lt;param1&gt;=&lt;value1&gt;][,&lt;param2&gt;=&lt;value2&gt;],...

激活自动场偏移（Activate Auto Field Shift, AFS）反交错。

**参数**
- top=&lt;int&gt;
- bottom=&lt;int&gt;
- left=&lt;int&gt;
- right=&lt;int&gt;
  裁剪出场偏移的范围

- method_switch=&lt;int&gt;  (0 - 256)  
  切换场偏移算法的阈值

- coeff_shift=&lt;int&gt;  (0 - 256)  
  场偏移阈值，更大的值会导致更多的场偏移

- thre_shift=&lt;int&gt;  (0 - 1024)  

  条纹检测（stripe detection）的阈值，将用于偏移决策。较低的值将导致更多的条纹检测。

- thre_deint=&lt;int&gt;   (0 - 1024)  
  反交错时使用的条纹检测（stripe detection）阈值。较低的值将导致更多的条纹检测。

- thre_motion_y=&lt;int&gt;  (0 - 1024)  
- thre_motion_c=&lt;int&gt;  (0 - 1024)  
  运动检测阈值。较低的值会导致更多的运动检测。

- level=&lt;int&gt;  (0 - 4)  
  选择如何移除条纹。

| level | 处理方法 | 目标 | 描述 |
|:---|:---|:---|:---|
| 0 | none  | | 不移除条纹。<br>将会输出场偏移生成的新帧。|
| 1 | triplication | 所有像素 | 将前一场混合到场偏移生成的新帧中。<br>运动引起的条纹将会变成残像。 |
| 2 | duplicate | 检测到条纹的像素 | 仅在检测到条纹的帧，将前一场混合到场偏移生成的新帧中。<br>适合运动较少的影片。 |
| 3 (默认) | duplicate  | 检测到运动的像素 | 仅在检测到运动的帧，将前一场混合到场偏移生成的新帧中。<br>该模式与2相比可以保留更多边缘和小字（small letters?） | 
| 4 | interpolate | 检测到运动的像素 | 在检测到运动的像素，丢弃一个场，并从另一个场插值来生成像素。<br>这不会导致残像，但是运动的像素的垂直分辨率将减半。 |

- shift=&lt;bool&gt;  
  启用场偏移（Field Shift）。

- drop=&lt;bool&gt;  
  丢弃显示时间小于1帧的帧。

  注意：启用该选项会生成可变帧率视频。当混流由 NVEncC 完成时，时间码（timecode）将会被自动应用。
  
  但当使用未处理输出（Raw）时，你需要为 vpp-afs 添加 "timecode=true" 来输出时间码文件，然后混流。

- smooth=&lt;bool&gt;  
  平滑图像显示时间

- 24fps=&lt;bool&gt;  
  强制 30fps -> 24fps 转换.

- tune=&lt;bool&gt;  
  当该选项设置为 true ，输出将会是运动和条纹检测结果，由下表颜色指示

| 颜色 | 描述 |
|:---:|:---|
| 暗蓝色 | 检测到运动 |
| 灰色 | 检测到条纹 |
| 亮蓝色 | 检测到运动和条纹 |

- rff=&lt;bool&gt;   
  当该选项设置为 true，输入的 RFF 标记将会被检查，当有RFF编码的逐行扫描帧时，反交错将不会被使用。

- log=&lt;bool&gt;  
  为每一帧生成AFS状态日志（调试用）。

- preset=&lt;string&gt;  
  参数如下表

|preset name   | default | triple | double | anime<br>cinema | min_afterimg |  24fps  | 30fps |
|:---          |:---:| :---:| :---:|:---:|:---:|:---:| :---:|
|method_switch |     0   |    0   |     0  |       64        |       0      |    92   |   0   |
|coeff_shift   |   192   |  192   |   192  |      128        |     192      |   192   |  192  |
|thre_shift    |   128   |  128   |   128  |      128        |     128      |   448   |  128  |
|thre_deint    |    48   |   48   |    48  |       48        |      48      |    48   |   48  |
|thre_motion_y |   112   |  112   |   112  |      112        |     112      |   112   |  112  |
|thre_motion_c |   224   |  224   |   224  |      224        |     224      |   224   |  224  |
|level         |     3   |    1   |     2  |        3        |       4      |     3   |    3  |
|shift         |    on   |  off   |    on  |       on        |      on      |    on   |  off  |
|drop          |   off   |  off   |    on  |       on        |      on      |    on   |  off  |
|smooth        |   off   |  off   |    on  |       on        |      on      |    on   |  off  |
|24fps         |   off   |  off   |   off  |      off        |     off      |    on   |  off  |
|tune          |   off   |  off   |   off  |      off        |     off      |   off   |  off  |
|rff           |   off   |  off   |   off  |      off        |     off      |   off   |  off  |

```
示例: same as --vpp-afs preset=24fps
--vpp-afs preset=anime,method_switch=92,thre_shift=448,24fps=true
```

### --vpp-nnedi [&lt;param1&gt;=&lt;value1&gt;][,&lt;param2&gt;=&lt;value2&gt;],...

使用 nnedi 进行反交错。丢弃其中一个场，再使用神经网络进行轮廓修正重新构建另一个场来去除交错，十分慢。

**参数**
- field  
  去除交错的方法。
  - auto (默认)  
    自动选择保持不变的场
  - top  
    保持上场不变
  - bottom  
    保持下场不变

- nns  (默认: 32)  
  神经网络的神经元数量。
  - 16, 32, 64, 128, 256

- nsize  (默认: 32x4)  
  神经网络中参照的近邻区域大小。
  - 8x6, 16x6, 32x6, 48x6, 8x4, 16x4, 32x4

- quality (默认: fast)  
  设定品质。

  - fast
  - slow  
    slow即将fast的神经网络的输出，与另一神经网络的输出进行混合来提升输出质量（当然，会变得更慢）。

- prescreen (默认: new_block)  
  进行预处理来决定是进行简单的补间还是使用神经网络进行修正。一般来说，只有边缘附近会被作为神经网络修正的对象，降低了使用神经网络的频率使得处理速度上升。

  - none  
    不进行预处理，将所有的像素使用神经网络进行重新构建。

  - original
  - new  
    进行预处理，在必要的地方使用神经网络进行修正。original和new在处理方式上不同，new要更快一些。

  - original_block
  - new_block  
    original/new的 GPU 优化版。不使用像素而使用区域作为判定单位。

- errortype (默认: abs)  
  选择神经网络的权重参数。
  - abs  
    使用训练过的权重参数让绝对误差最小。
  - square  
    使用训练过的权重参数让二乘误差最小。

- prec (默认: auto)  
  选择运算精度。
  - auto  
    当fp16可用并且使用能获得更快的速度的时候，将自动选择fp16。

    当前 Turing 架构的 GPU 将会自动使用 fp16。Pascal 架构的 GPU 虽然可以使用 fp16 但是太慢了所以不会使用。
    - fp16  
      强制使用fp16。只适用于x64。
    
    - fp32  
      强制使用fp32。

- weightfile (默认: 使用内置文件)  
  指定权重参数文件。不指定的时候将会使用内置的数据。

```
示例：--vpp-nnedi field=auto,nns=64,nsize=32x6,quality=slow,prescreen=none,prec=fp32
```

### --vpp-yadif [&lt;param1&gt;=&lt;value1&gt;]
使用 yadif 进行反交错。

**参数**
- mode

  - auto (default)  
    自动选择保持不变的场
  - tff  
    保持上场不变
  - bff  
    保持下场不变
  - bob   
    处理成60fps（场序自动选择）
  - bob_tff   
    处理成60fps（上场优先）
  - bob_bff  
    处理成60fps（下场优先）

### --vpp-decimate [&lt;param1&gt;=&lt;value1&gt;][,&lt;param2&gt;=&lt;value2&gt;],...  
删除重复帧。

**参数**
  - cycle=&lt;int&gt;  (默认: 5)  
    丢弃帧的周期。从每该设置的值的帧中丢弃1帧。

  - drop=&lt;int&gt;  (默认: 1)  
    一个周期内丢弃的帧数

  - thredup=&lt;float&gt;  (默认: 1.1,  0.0 - 100.0)  
    重复帧判断阈值。

  - thresc=&lt;float&gt;   (默认: 15.0,  0.0 - 100.0)  
    场景变化判断阈值。

  - blockx=&lt;int&gt;  
  - blocky=&lt;int&gt;  
    判定重复时计算使用的块大小。默认：32。
    块大小可以设置为 16、32、64。

  - chroma=&lt;bool&gt;  
    考虑色度成分进行判断。(默认: on)

  - log=&lt;bool&gt;  
    输出判断结果日志。 (默认: off)

### --vpp-mpdecimate [&lt;param1&gt;=&lt;value1&gt;][,&lt;param2&gt;=&lt;value2&gt;],...  
通过删除连续的重复帧，制作VFR动画，这有助于提高编码速度和压缩率。
注意，此过滤器将自动启用[--avsync](./NVEncC_Options.cn.md#--avsync-string) vfr。

- **参数**
  - hi=&lt;int&gt;  (默认: 768, 8x8x12)  
    作为是否丢弃的阈值。如果8x8块中任意一个块的差异大于"hi"，则从丢弃对象中排除。

  - lo=&lt;int&gt;  (默认: 320, 8x8x5)  
  - frac=&lt;float&gt;  (默认: 0.33)  
    对于8x8块，如果差异小于"lo"的块的数量大于"frac"，则帧可能会被丢弃。

  - max=&lt;int&gt;  (默认: 0)  
    可以丢弃的最大连续帧数 (如果为正数)。
    丢弃帧之间的最小间隔 (如果为负数)。
    
  - log=&lt;bool&gt;  
    输出日志文件 (默认: off)

### --vpp-select-every &lt;int&gt;[,&lt;param1&gt;=&lt;int&gt;]

每隔特定数量的帧，选取一帧进行输出。

**参数**
- step=&lt;int&gt;
- offset=&lt;int&gt; (默认：0)

```
示例一： (即 "select even"): --vpp-select-every 2
示例二： (即 "select odd "): --vpp-select-every 2,offset=1
```

### --vpp-rotate &lt;int&gt;   
旋转视频。可以旋转90、180、270度。

### --vpp-transform [&lt;param1&gt;=&lt;value1&gt;][,&lt;param2&gt;=&lt;value2&gt;],...   

**参数**
- flip_x=&lt;bool&gt;

- flip_y=&lt;bool&gt;

- transpose=&lt;bool&gt;

### --vpp-convolution3d [&lt;param1&gt;=&lt;value1&gt;][,&lt;param2&gt;=&lt;value2&gt;],...
3维空间降噪

- **参数**
  - matrix=&lt;string&gt;  (default=original)  
    选择要使用的矩阵。  

    - 标准
      ```
      1 2 1 2 4 2 1 2 1 
      2 4 1 4 8 4 2 4 1 
      1 2 1 2 4 2 1 2 1 
      ```
    - 简单
      ```
      1 1 1 1 1 1 1 1 1 
      1 1 1 1 1 1 1 1 1 
      1 1 1 1 1 1 1 1 1 
      ```
    
  - fast=&lt;bool&gt  (默认: false)  
    使用简化计算的快速模式。
  
  - ythresh=&lt;float&gt;  (默认: 3, 0-255)  
    以空间方向的亮度成分的阈值进行轮廓的保护。值越大，则噪声去除越强，但轮廓可能会变得模糊。
  
  - cthresh=&lt;float&gt;  (默认: 4, 0-255)  
    以空间方向的色度分量的阈值进行轮廓的保护。值越大，则噪声去除越强，但轮廓可能会变得模糊。
  
  - t_ythresh=&lt;float&gt;  (默认: 3, 0-255)  
    通过时间方向的亮度成分的阈值，防止场景改变中的残像。值越大，则噪声去除越强，但在场景变化中容易产生残像。推荐10以下的值。
  
  - t_cthresh=&lt;float&gt;  (默认: 4, 0-255)  
    通过时间方向的色度分量的阈值，防止场景变化中的残像。值越大，则噪声去除越强，但在场景变化中容易产生残像。推荐10以下的值。
  

```
例子: 使用简单矩阵
--vpp-convolution3d matrix=simple
```

### --vpp-nvvfx-denoise [&lt;param1&gt;=&lt;value1&gt;][,&lt;param2&gt;=&lt;value2&gt;],...
[NVIDIA MAXINE VideoEffects SDK](https://github.com/NVIDIA/MAXINE-VFX-SDK),提供的摄像头降噪过滤器，仅x64版本支持此功能。

从摄像机视频中去除低亮度摄像机噪声，同时保留纹理细节，

支持80p到1080p之间的分辨率。

这一过滤器支持 Turing 架构(RTX20xx)及更新的显卡，如果要使用它，需要下载并安装 [Video Effect models and runtime dependencies](https://www.nvidia.com/broadcast-sdk-resources)。

- **参数**
  - strength=&lt;int&gt;
    - 0  
      较弱的效果，更重视保护纹理细节

    - 1  
      较强的效果，更重视去除噪声

### --vpp-nvvfx-artifact-reduction [&lt;param1&gt;=&lt;value1&gt;][,&lt;param2&gt;=&lt;value2&gt;],...
[NVIDIA MAXINE VideoEffects SDK](https://github.com/NVIDIA/MAXINE-VFX-SDK)提供的过滤器。在保存原始动画的信息的同时，去除视频编码时产生的压缩劣化效果。

注意，仅支持90p-1080p的分辨率，执行需要x64版的执行文件和 Turing架构(RTX20xx)以后的GPU。要使用该过滤器，请下载并安装[Video Effect models and runtime dependencies](https://www.nvidia.com/broadcast-sdk-resources)

- **参数**
  - mode=&lt;int&gt;
    - 0 (默认)  
      去除较少的压缩劣化，保留更多的信息。适用于原始文件有较高码率的情况。

    - 1  
      更强的效果，适用于原始文件码率较低的情况

### --vpp-smooth [&lt;param1&gt;=&lt;value1&gt;][,&lt;param2&gt;=&lt;value2&gt;],...

**参数**
- quality=&lt;int&gt;  (default=3, 1-6)  
  过滤器的目标质量，值越大精度越高速度越慢。

- qp=&lt;int&gt;  (default=12, 1 - 63)    
  滤镜强度。较高的值欧哲较强的去早点效果但会导致模糊不清。

- prec (默认: auto)  
  选择计算精度。
  - auto  
    如可以使用fp16且fp16似乎更快，则自动选择fp16。   
    当前对Turing架构的GPU自动使用fp16。   
    Pascal的GPU虽然可以使用fp16但速度很慢默认不使用。

  - fp16 (仅64位版本)  
    主要使用半精度浮点数进行计算。在某些环境下速度很快。Maxwell以前的GPU和32位版本无法使用。

  - fp32  
    使用单精度浮点数进行计算。

### --vpp-denoise-dct [&lt;param1&gt;=&lt;value1&gt;][,&lt;param2&gt;=&lt;value2&gt;],...

- **参数**
  - step=&lt;int&gt;  

    影响过滤器的质量，较小的值会产生较高的质量，但有着较低的速度

    - 1 (高质量，慢)
    - 2 (默认)
    - 4
    - 8 (快)
  
  - sigma=&lt;float&gt;  (default=4.0)    

    过滤器的强度，更大的值有着较高的降噪效果，但会导致模糊
    
  - block_size=&lt;int&gt;  (default=8)  
    - 8
    - 16 (慢)
    
### --vpp-knn [&lt;param1&gt;=&lt;value1&gt;][,&lt;param2&gt;=&lt;value2&gt;],...
强降噪滤镜。

**参数**
- radius=&lt;int&gt;  (默认=3, 1-5)   
  滤镜半径

- strength=&lt;float&gt;  (默认=0.08, 0.0 - 1.0)   
  滤镜强度

- lerp=&lt;float&gt;   (默认=0.2, 0.0 - 1.0)  
  原始像素与降噪像素的混合程度

- th_lerp=&lt;float&gt;  (默认=0.8, 0.0 - 1.0)  
  边缘检测阈值

```
示例: slightly stronger than default
--vpp-knn radius=3,strength=0.10,lerp=0.1
```

### --vpp-pmd [&lt;param1&gt;=&lt;value1&gt;][,&lt;param2&gt;=&lt;value2&gt;],...

由修改过的 pmd 方法提供的较弱的降噪，尝试在降噪的同时保留边缘。

**参数**
- apply_count=&lt;int&gt;  (默认=2, 1- )  
  应用滤镜的次数

- strength=&lt;float&gt;  (默认=100, 0-100)  
  滤镜强度

- threshold=&lt;float&gt;  (默认=100, 0-255)  
  边缘检测阈值。较小的值会导致更多的（像素）被识别为边缘从而保留。

```
示例: 比默认更弱一点点
--vpp-pmd apply_count=2,strength=90,threshold=120
```

### --vpp-gauss &lt;int&gt;

设置高斯滤镜的大小。可用3、5、7。

需要NVEncC64所在文件夹下存在nppc64_10.dll, nppif64_10.dll, nppig64_10.dll。只在x64版本支持。

npp dll可以在[这里](https://github.com/rigaya/NVEnc/releases/tag/7.00) (npp64_10_dll_7zip.7z)下载。

### --vpp-subburn [&lt;param1&gt;=&lt;value1&gt;][,&lt;param2&gt;=&lt;value2&gt;],...
将指定字幕压入。文本格式的字幕使用[libass](https://github.com/libass/libass)渲染。

**参数**
- track=&lt;int&gt;  
  压入输入文件的指定字幕轨（仅当使用--avhw、--avsw时有效，字幕轨从1起编号）

- filename=&lt;string&gt;  
  压入指定字幕文件。

- charcode=&lt;string&gt;  
  指定字幕的文本编码。（当字幕为文本格式时有效）

- shaping=&lt;string&gt;  
  指定字幕的渲染质量。（当字幕为文本格式时有效）
  - simple
  - complex (默认)

- scale=&lt;float&gt; (默认=0.0 (auto))  
  bitmap格式字幕缩放倍率。  

- transparency=&lt;float&gt; (デフォルト=0.0, 0.0 - 1.0)   
  字幕透明度。  

- brightness=&lt;float&gt; (デフォルト=0.0, -1.0 - 1.0)   
  字幕亮度调整。  

- contrast=&lt;float&gt; (デフォルト=1.0, -2.0 - 2.0)   
  字幕对比度调整。  

- vid_ts_offset=&lt;bool&gt;  
  为字幕轨增加偏移使其与视频的起始时间戳相合。 (默认=on)   
  使用"track"时该设置总是有效。

- ts_offset=&lt;float&gt; (默认=0.0)   
  字幕时间戳偏移，单位秒。

- fontsdir=&lt;string&gt;  
  使用的字体目录
    
- forced_subs_only=&lt;bool&gt;  
  强制仅渲染子对象 (默认: off).

```
例1: 将输入文件的第1字幕轨压入
--vpp-subburn track=1
例2: 压入PGS字幕
--vpp-subburn filename="subtitle.sup"
例3: 压入Shift-JIS编码的ass字幕文件
--vpp-subburn filename="subtitle.sjis.ass",charcode=sjis,shaping=complex
```

### --vpp-resize &lt;string&gt;

设置缩放算法。

- **选项**
  - algo=&lt;string&gt;  
    选择使用的算法
    | 选项名 | 描述 | 需要的dll |
    |:---|:---|:---:|
    | auto  | 自动选择 | |
    | bilinear | 线性插值 | |
    | bicubic  | 双三次插值 | |
    | spline16 | 4x4 样条曲线插值 | |
    | spline36 | 6x6 样条曲线插值 | |
    | spline64 | 8x8 样条曲线插值 | |
    | lanczos2 | 4x4 lanczos插值 | |
    | lanczos3 | 6x6 lanczos插值 | |
    | lanczos4 | 8x8 lanczos插值 | |
    | nn            | 近邻法 | ○ |
    | npp_linear    | NPP 库提供的线性插值 | ○ |
    | cubic         | 4x4 立方插值 | ○ |
    | super         | NPP 库提供的所谓的 "super sampling"  | ○ |
    | lanczos       | Lanczos 插值                    | ○ |
    | nvvfx-superres | 基于nvvfx库的超分辨率(仅适用于放大) |   |

  - superres-mode=&lt;int&gt;  
    选择nvvfx-superres的模式
    - 0 ... 保守 (default)
    - 1 ... 激进
  - superres-strength=&lt;float&gt;  
    nvvfx-superres的强度(0.0 - 1.0)

- 注意事项
  - 标记为"○"的算法需要[NPP library](https://developer.nvidia.com/npp)，仅在 x64 版本支持。要使用这些算法，需要另外下载 nppc64_10.dll, nppif64_10.dll, nppig64_10.dll并把它和 NVEncC64.exe 放置在同一目录。
    这些npp dll可以在[这里](https://github.com/rigaya/NVEnc/releases/tag/7.00) (npp64_10_dll_7zip.7z)下载。
  - ```nvvfx-superres``` 是来自[NVIDIA MAXINE VideoEffects SDK](https://github.com/NVIDIA/MAXINE-VFX-SDK)的超分辨率过滤器, 仅在x64版本支持。
    这一模式支持 Turing 架构(RTX20xx)及更新的显卡，如果要使用这一模式，需要下载并安装 [Video Effect models and runtime dependencies](https://www.nvidia.com/broadcast-sdk-resources).


```
例子: 使用 spline64
--vpp-resize spline64

例子: 使用 spline64
--vpp-resize algo=spline64 

例子: 使用 nvvfx-superres, 模式为激进
--vpp-resize algo=nvvfx-superres,superres-mode=1
  ```
### --vpp-unsharp [&lt;param1&gt;=&lt;value1&gt;][,&lt;param2&gt;=&lt;value2&gt;],...
反锐化滤镜，用于边缘和细节增强。

**参数**
- radius=&lt;int&gt; (默认=3, 1-9)  
  边缘和细节检测阈值

- weight=&lt;float&gt; (默认=0.5, 0-10)  
  边缘和细节强调强度。较大的值会导致更强的效果

- threshold=&lt;float&gt;  (默认=10.0, 0-255)  
  边缘和细节检测阈值

```
示例: 稍强的unsharp
--vpp-unsharp weight=1.0
```

### --vpp-edgelevel [&lt;param1&gt;=&lt;value1&gt;][,&lt;param2&gt;=&lt;value2&gt;],...
边缘等级调整滤镜，用于锐化边缘。


**参数**
- strength=&lt;float&gt; (默认=5.0, -31 - 31)  
  边缘锐化强度。较大的值会导致更强的边缘锐化。

- threshold=&lt;float&gt;  (默认=20.0, 0 - 255)  
  噪点阈值以避免增强噪点。较大的值会将更大的亮度变化视作噪点。

- black=&lt;float&gt;  (默认=0.0, 0-31)  
  增强边缘暗部的强度

- white=&lt;float&gt;  (默认=0.0, 0-31)  
  增强边缘亮部的强度

```
示例: 稍强的边缘等级调整 (Aviutl 默认)
--vpp-edgelevel strength=10.0,threshold=16.0,black=0,white=0

Example: 增强边缘的暗部
--vpp-edgelevel strength=5.0,threshold=24.0,black=6.0
```

### --vpp-warpsharp [&lt;param1&gt;=&lt;value1&gt;][,&lt;param2&gt;=&lt;value2&gt;],...
边缘细化(锐化)过滤器。用于轮廓调整。

- **参数**
  - threshold=&lt;float&gt;  (默认=128.0, 0 - 255)  
    检测轮廓的阈值。值越高，过滤器的效果越强。
  
  - blur=&lt;int&gt;  (默认=2)  
    模糊的次数。模糊次数越多，锐化越弱。
  
  - type=&lt;int&gt;  (默认=0)  
    - 0 ... 进行 13x13 大小的模糊处理
    - 1 ... 进行 5x5 大小的模糊处理。会产生更高的质量，但需要更多次数的模糊。
    
  - depth=&lt;float&gt;  (默认=16.0, -128.0 - 128.0)  
    细化的深度，增加该值会有着更强的锐化效果。
    
  - chroma=&lt;int&gt;  (默认=0)  
    设定处理色度通道的方式。
    - 0 ... 将亮度的轮廓检测结果直接应用于色度通道。
    - 1 ... 对各色度通道分别进行轮廓检测。
  
```
例子: 使用 type 1
--vpp-warpsharp threshold=128,blur=3,type=1
```


### --vpp-tweak [&lt;param1&gt;=&lt;value1&gt;][,&lt;param2&gt;=&lt;value2&gt;],...
- **参数**
  - brightness=&lt;float&gt; (default=0.0, -1.0 - 1.0)  
  
  - contrast=&lt;float&gt; (default=1.0, -2.0 - 2.0)  
  
  - gamma=&lt;float&gt; (default=1.0, 0.1 - 10.0)  
  
  - saturation=&lt;float&gt; (default=1.0, 0.0 - 3.0)  
  
  - hue=&lt;float&gt; (default=0.0, -180 - 180)  
  
  - swapuv=&lt;bool&gt;  (default=false)
  
- 
```
例子:
--vpp-tweak brightness=0.1,contrast=1.5,gamma=0.75
```

### --vpp-curves [&lt;param1&gt;=&lt;value1&gt;][,&lt;param2&gt;=&lt;value2&gt;],...  
使用指定变换曲线调整颜色的过滤器。

- **参数**
  - preset=&lt;float&gt;  
    - none
    - color_negative
    - process
    - darker
    - lighter
    - increase_contrast
    - linear_contrast
    - medium_contrast
    - strong_contrast
    - negative
    - vintage
  
  - m=&lt;string&gt;  
    指定用于亮度调整的曲线。将在RGB处理后作为后处理执行。

  - r=&lt;string&gt;  
    指定应用于红色分量的曲线。会覆盖之前的配置。
  
  - g=&lt;string&gt;  
    指定应用于绿色分量的曲线。会覆盖之前的配置。
  
  - b=&lt;string&gt;  
    指定应用于蓝色分量的曲线。会覆盖之前的配置。
  
  - all=&lt;string&gt;  
    指定所有分量的曲线。在r、g、b的曲线未指定的情况下使用。会覆盖之前的配置。

```
例子:
--vpp-curves r="0/0.11 0.42/0.51 1/0.95":g="0/0 0.50/0.48 1/1":b="0/0.22 0.49/0.44 1/0.8"
  ```

### --vpp-deband [&lt;param1&gt;=&lt;value1&gt;][,&lt;param2&gt;=&lt;value2&gt;],...

**参数**
- range=&lt;int&gt; (默认=15, 0-127)  
  模糊范围。用于模糊的样本在此范围内选取

- sample=&lt;int&gt; (默认=1, 0-2)  
  - sample = 0
    通过参考“范围”内的像素来执行模糊处理。

  - sample = 1
    通过参考总计2像素，包括一“范围”内的像素及其点对称像素来执行模糊处理

  - sample = 2
    通过参考总计4像素，包括两个“范围”内的像素及其点对称像素来执行模糊处理

- thre=&lt;int&gt; (为 y, cb 和 cr 设置相同阈值)
- thre_y=&lt;int&gt; (默认=15, 0-31)
- thre_cb=&lt;int&gt; (默认=15, 0-31)
- thre_cr=&lt;int&gt; (默认=15, 0-31)  
  为 y, cb, cr 模糊设定阈值。较高的值会导致更强的滤镜强度，但线条和边缘有可能消失

- dither=&lt;int&gt;   (set same dither for y & c)
- dither_y=&lt;int&gt; (default=15, 0-31)
- dither_c=&lt;int&gt; (default=15, 0-31)  
  y 和 c 的抖动强度

- seed=&lt;int&gt;  
  随机数种子

- blurfirst (默认=off)  
  首先处理模糊以达到更强的效果。副作用也可能更明显，细线条可能会消失。

- rand_each_frame (默认=off)  
  每一帧改变用于滤镜的随机数

```
示例:
--vpp-deband range=31,dither=12,rand_each_frame
```

### --vpp-pad &lt;int&gt;,&lt;int&gt;,&lt;int&gt;,&lt;int&gt;

为左、上、右、下边缘添加内边距，单位像素。

### --vpp-overlay [&lt;param1&gt;=&lt;value1&gt;][,&lt;param2&gt;=&lt;value2&gt;],...
将指定的图像(图片或视频)覆盖在视频上。

- **参数**
  - file=&lt;string&gt;  
    文件的路径。
    当文件为视频时，视频的帧率需要等于要覆盖的视频的帧率。
  
  - pos=&lt;int&gt;x&lt;int&gt;  
    添加图像的位置
  
  - size=&lt;int&gt;x&lt;int&gt;  
    图像大小
  
  - alpha=&lt;float&gt; (默认: 1.0 (0.0 - 1.0))  
    图像的透明度(alpha值)，1.0表示不透明，0.0表示完全透明
  
  - alpha_mode=&lt;string&gt;  
    - override ... 直接使用alpha值覆盖
    - mul      ... 使用alpha值进行乘算
    - lumakey  ... 根据亮度作为alpha值进行覆盖
  
  - lumakey_threshold=&lt;float&gt; (默认: 0.0 (dark: 0.0 - 1.0 :bright))  
    亮度值作为透明度
  
  - lumakey_tolerance=&lt;float&gt; (默认: 0.1 (0.0 - 1.0))  
    指定亮度值的透明度范围。
  
  - lumakey_softness=&lt;float&gt; (默认: 0.0 (0.0 - 1.0))  
    指定亮度值的softness范围。

```
例子:
--vpp-overlay file=logo.png,pos=1620x780,size=300x300
--vpp-overlay file=logo.mp4,pos=0x800,alpha_mode=lumakey,lumakey_threshold=0.0,lumakey_tolerance=0.1
```

### --vpp-perf-monitor
监视每个vpp滤镜的性能，输出应用的滤镜处理每帧的平均时间。开启该选项可能会对整体编码性能产生轻微影响。

### --vpp-nvvfx-model-dir &lt;string&gt;
设置Video Effect module模型文件的路径

## 其他设置

### --cuda-schedule &lt;string&gt;
  调整当等待 GPU 任务完成时 CPU 的表现。默认为 auto。

- auto (默认)
  将模式选择交由 CUDA 驱动。

- spin
  总是使 CPU 监视 GPU 任务的完成情况。同步的延迟将最小化，会总是使一个逻辑核的占用率达到100%

- yeild
  与 spin 基本相同，但允许切换到另一运行中的线程

- sync
  睡眠线程直到 GPU 任务完成。性能可能下降，但会减少 CPU 占用率，尤其是使用硬件解码时。

### --disable-nvml &lt;int&gt;
禁用 NVML GPU 监视器

- **参数**
  - 0 (默认)  
    启用 NVML。

  - 1
    当系统存在一个 CUDA 设备时禁用 NVML。

  - 2
    总是禁用 NVML。

### --output-buf &lt;int&gt;

指定输出缓冲区大小。单位为 MB，默认为 8，最大为 128。

输出缓冲区会存储输出数据，当数据量达到缓冲区上限时，数据将会被一次性写入。这可以带来更高的性能和更少的磁盘文件碎片。

此外，缓冲区太大可能会降低性能，因为向磁盘写入大量数据将会花费更长的时间。一般来说，默认值是较好的选择。

如果输出不是文件，缓冲区不会被使用。

### --output-thread &lt;int&gt;

是否使用单独线程输出。

- -1 ... 自动 (默认)
- 0 ... 不使用输出线程
- 1 ... 使用输出线程

使用输出线程会增加内存占用，但有时可以提高编码性能。

### --log &lt;string&gt;

把日志输出到指定文件。

### --log-level &lt;string&gt;

指定日志输出等级。

- **等级**
  - trace ... 每一帧都输出信息 (慢)
  - debug ... 输出更多信息，主要用于调试
  - info ... 显示编码信息 (默认)
  - warn ... 输出错误和警告
  - error ...只输出错误
  - quiet ... 不显示日志

- **目标**  
  日志的目标类别，默认为 all
  - all ... 所有目标
  - app ... 除了libav, libass, perfmonitor, amf以外的所有目标
  - device ... 设备初始化
  - core ... core的日志，包括 core_progress 和 core_result
  - core_progress ... 进度指示器
  - core_result ... 编码结果
  - decoder ... 解码器日志
  - input ... 文件输入日志
  - output ... 文件输出日志
  - vpp ... vpp 过滤器日志
  - amf ... ofamf 库日志
  - opencl ... opencl 日志
  - libav ... 内部 libav 库日志
  - libass ... ass 库日志
  - perfmonitor ... 性能监视器日志

```
例子: 启用 debug 日志
--log-level debug
  
例子: 仅显示 application 的 debug 日志
--log-level app=debug
  
例子: 仅显示 progress 的日志
--log-level error,core_progress=info
```


### --log-opt &lt;param1&gt;=&lt;value&gt;[,&lt;param2&gt;=&lt;value&gt;]...
关于日志输出的其他选项
- **参数**
  - addtime (默认=off)  
    日志信息包含时间

### --log-framelist [&lt;string&gt;]
只用于调试
输出avsw/avhw reader的日志

### --log-packets [&lt;string&gt;]
只用于调试
输出avsw/avhw reader的packets read日志

### --log-mux-ts [&lt;string&gt;]
只用于调试
输出packets written日志

### --thread-affinity [&lt;string1&gt;=]{&lt;string2&gt;[#&lt;int&gt;[:&lt;int&gt;]...] or 0x&lt;hex&gt;}
设置NVEncC的进程和线程的cpu核心亲和性。

- **目标** (&lt;string1&gt;)
  设置要设置线程亲和性的目标。默认为"all"。
  
  - all ... All targets
  - process ... process of NVEncC
  - main ... main thread
  - decoder ... avhw decode thread
  - csp ... colorspace conversion threads (CPU)
  - input ... input thread
  - output ... output thread
  - audio ... audio processing threads
  - perfmonitor ... performance monitoring threads
  - videoquality ... ssim/psnr/vmaf calculation thread

- **进程偏好** (&lt;string2&gt;)
  - all ... 所有核心(无限制)
  - pcore ... 性能核心(P核)(仅限混合体系架构)
  - ecore ... 能效核心(E核)(仅限混合体系架构)
  - logical ... "#"后指定的逻辑核心 (仅限windows)
  - physical ... "#"后指定的物理核心 (仅限windows)
  - cachel2 ... 使用了"#"后指定的L2缓存的核心，用法见例4 (仅限windows)
  - cachel3 ... 使用了"#"后指定的L3缓存的核心，用法见例4 (仅限windows)
  - <hex> ... set by 0x<hex> (same as "start /affinity")

```
例1: 设置进程亲和0,1,2,5,6号物理核心
--thread-affinity process=physical#0-2:5:6
  
例2: 设置进程亲和0,1,2,3号逻辑核心
--thread-affinity process=0x0f
--thread-affinity process=logical#0-3
--thread-affinity process=logical#0:1:2:3
  
例3: 设置性能监控进程亲和E核(在混合体系架构)
--thread-affinity perfmonitor=ecore
  
例4: 设置进程亲和Ryzen CPU的第一个CCX
--thread-affinity process=cachel3#0
```

### --thread-priority [&lt;string1&gt;=]&lt;string2&gt;[#&lt;int&gt;[:&lt;int&gt;]...]
设置进程或线程的优先级 [仅限Windows]

- **目标** (&lt;string1&gt;)
  设置要设置优先级的目标。默认为"all"。
  
  - all ... All targets below.
  - process ... whole process
  - main ... main thread
  - decoder ... avhw decode thread
  - csp ... colorspace conversion threads (CPU)
  - input ... input thread
  - encoder ... background encoder threads
  - output ... output thread
  - audio ... audio processing threads
  - perfmonitor ... performance monitoring threads
  - videoquality ... ssim/psnr/vmaf calculation thread
  
- **优先级** (&lt;string2&gt;)
  - background, idle, lowest, belownormal, normal (default), abovenormal, highest
  
```
例子: 将所有进程的优先级设为belownormal(低于正常)
--thread-priority process=belownormal
  
例子: 将output进程的优先级设为belownormal，background进程优先级设为performance
--thread-priority output=belownormal,perfmonitor=background
```

### --thread-throttling [&lt;string1&gt;=]&lt;string2&gt;[#&lt;int&gt;[:&lt;int&gt;]...]  
  设置进程或线程的效能模式 [仅限Windows]

- **目标** (&lt;string1&gt;)
  设置要设置功率限制模式的目标。默认为"all"。
  
  - all ... All targets below.
  - main ... main thread
  - decoder ... avhw decode thread
  - csp ... colorspace conversion threads (CPU)
  - input ... input thread
  - encoder ... background encoder threads
  - output ... output thread
  - audio ... audio processing threads
  - perfmonitor ... performance monitoring threads
  - videoquality ... ssim/psnr/vmaf calculation thread
  
- **模式** (&lt;string2&gt;)
  - unset (默认)    ... 根据编码目标自动设置
  - auto            ... 由操作系统决定
  - on              ... 偏好能效
  - off             ... 偏好性能
  
```
例子: 将output线程和性能监控线程设定为偏好能效
--thread-throttling output=on,perfmonitor=on
  
例子: 将main线程和input线程设定为偏好性能
--thread-throttling main=off,input=off
```

### --option-file &lt;string&gt;

从文件中载入选项列表

换行符被视为空格，因此一个选项或值不应拆分为多行。

### --max-procfps &lt;int&gt;

设置转码速度上限。默认为0（不限制）。

当你想要同时编码多个流，并且不想其中一个占用全部 CPU 或 GPU 资源时可以使用该选项。

```
示例: 限制最大转码速度为 90fps
--max-procfps 90
```

### --lowlatency   
降低编码延迟的模式。由于会降低最大编码速度（吞吐量），一般不会使用。

### --avsdll &lt;string&gt;
指定要使用的AviSynth DLL位置。未指定时，将使用默认的AviSynth.dll。

### --process-codepage &lt;string&gt; [仅限Windows]  
- **参数**  
  - utf8  
    使用utf-8作为编码方式(默认)
  
  - os  
    使用系统默认的编码方式
    
    这将允许AviSynth脚本文件使用非-ASCII字符。
    
    要应用此选项，需要更改执行文件中嵌入的名为manifest的信息。因此将自动复制执行文件，生成改写了manifest的临时执行文件，并执行该文件。
    

### --perf-monitor [&lt;string&gt;][,&lt;string&gt;]...

输出性能信息。可以从下表中选择要输出的信息的名字，默认为全部。

- **参数**
  ```
  all          ... 监视全部信息
  cpu_total    ... CPU 总占用 (%)
  cpu_kernel   ... CPU 核心占用 (%)
  cpu_main     ... CPU 核心线程占用 (%)
  cpu_enc      ... CPU 编码线程占用 (%)
  cpu_in       ... CPU 输入线程占用 (%)
  cpu_out      ... CPU 输出线程占用 (%)
  cpu_aud_proc ... cpu aud proc 线程占用 (%)
  cpu_aud_enc  ... cpu aud enc 线程占用 (%)
  cpu          ... 监视全部 CPU 信息
  gpu_load    ... GPU 占用 (%)
  gpu_clock   ... GPU 平均时钟频率
  vee_load    ... GPU 视频编码器占用 (%)
  ved_load    ... GPU 视频解码器占用 (%)
  gpu         ... 监视全部 GPU 信息
  queue       ... 队列占用
  mem_private ... 私有内存 (MB)
  mem_virtual ... 虚拟内存 (MB)
  mem         ... 监视全部内存信息
  io_read     ... 读取速度  (MB/s)
  io_write    ... 写入速度 (MB/s)
  io          ... 监视全部I/O信息
  fps         ... 编码速度 (fps)
  fps_avg     ... 平均编码速度 (fps)
  bitrate     ... 编码码率 (kbps)
  bitrate_avg ... 平均编码码率 (kbps)
  frame_out   ... 已写入的帧数
  ```

### --perf-monitor-interval &lt;int&gt;
指定[--perf-monitor](#--perf-monitor-stringstring)性能监视的间隔，单位ms（应为50或更高）。默认为500。