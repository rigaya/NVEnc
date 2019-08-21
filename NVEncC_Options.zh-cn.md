
# NVEncC 选项列表

**[日本語版はこちら＞＞](./NVEncC_Options.ja.md)**


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

#### 使用 hw (cuvid) 解码器（交错）

```Batchfile
NVEncC --avhw --interlace tff -i "<mp4(H.264/AVC) file>" -o "<outfilename.264>"
```

#### Avisynth 示例 (avs 和 vpy 均可通过 vfw 读取)

```Batchfile
NVEncC -i "<avsfile>" -o "<outfilename.264>"
```

#### 管道输入示例

```Batchfile
avs2pipemod -y4mp "<avsfile>" | NVEncC - y4m - i - - o "<outfilename.264>"
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

带有 [] 的参数是可选的。

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

### --check-filters

显示可用的音频滤镜

### --check-avversion

显示 ffmpeg dll 版本号

## Basic encoding options

### -d, --device &lt;int&gt;

指定 NVEnc 使用的 deviceId。deviceID 可以通过 [--check-device](#--check-device) 获得。

如果未指定，且当前环境有多个可用的GPU，则将会根据以下条件自动选择

- 设备是否支持指定的编码
- 如果 --avhw 启用，则检查设备是否支持硬件解码该输入文件
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
| raw | ○ |  |  |  |  |  |
| y4m | ◎ |  | ◎ | ◎ |  |  |
| avi | ○ | ○ |  |  | ○ | ○ |
| avs | ◎ | ○ | ◎ | ◎ |  |  |
| vpy | ◎ |  | ◎ | ◎ |  |  |
| avhw | ◎ |  |  |  |  |  |
| avsw | ◎ |  | ◎ | ◎ | ○ | ○ |

◎ ... 支持 8bit / 9bit / 10bit / 12bit / 14bit / 16bit   
○ ... 仅支持 8 bits

### --raw

将输入格式指定为未处理格式（Raw）。必须指定输入分辨率和帧率。

### --y4m

将输入格式指定为 y4m (YUV4MPEG2) 。

### --avi

使用 avi 读取器读取 avi 文件。

### --avs

使用 avs 读取器读取 Avisynth 脚本文件。

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
| VC-1       | × |
| WMV3/WMV9  | × |

○ ... 支持  
× ... 不支持

### --interlace &lt;string&gt;

指定 **输入** 的交错标志。

通过 [--vpp-deinterlace](#--vpp-deinterlace-string) 或 [--vpp-afs](#--vpp-afs-param1value1param2value2) 可以进行反交错。如果未指定反交错，则将会进行交错编码。

- none ... 逐行扫描
- tff ... 上场优先
- bff ... 下场优先

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

### --fps &lt;int&gt;/&lt;int&gt; or &lt;float&gt;

设置输入帧率，未处理格式（Raw）输入时需要。

### --input-res &lt;int&gt;x&lt;int&gt;

设置输入分辨率，未处理格式（Raw）输入时需要。

### --output-res &lt;int&gt;x&lt;int&gt;

设置输出分辨率。当与输入分辨率不同时，将会自动启用硬件/GPU缩放器。

未指定时将会与输入分辨率相同（不缩放）。


## 编码模式选项

默认选择为 CQP （固定质量）。

### --cqp &lt;int&gt; or &lt;int&gt;:&lt;int&gt;:&lt;int&gt;

将 QP 值设定为 &lt;I 帧&gt;:&lt;P 帧&gt;:&lt;B 帧&gt;。

一般情况下，推荐将 QP 值设置为 I &lt; P &lt; B 的组合。

### --cbr &lt;int&gt;
### --cbrhq &lt;int&gt;
### --vbr &lt;int&gt;
### --vbrhq &lt;int&gt;

设置码率，单位kbps。

## Other Options for Encoder

### --output-depth &lt;int&gt;

设置输出位深度。
- 8 ... 8 bits (默认)
- 10 ... 10 bits

### --lossless
进行无损输出。 (默认：关)

### --max-bitrate &lt;int&gt;
最大码率，单位kbps。

### --qp-init &lt;int&gt; 或 &lt;int&gt;:&lt;int&gt;:&lt;int&gt;

设置初始 QP 值为 &lt;I 帧&gt;:&lt;P 帧&gt;:&lt;B 帧&gt;。CQP模式下将会被忽略。

这些值将会被在编码开始时被应用。如果希望调节视频起始段的画面质量请设置该值。在 CBR/VBR 模式下有时会不稳定。

### --qp-min &lt;int&gt; or &lt;int&gt;:&lt;int&gt;:&lt;int&gt;

设置最小 QP 值为 &lt;I 帧&gt;:&lt;P 帧&gt;:&lt;B 帧&gt;。CQP模式下将会被忽略。

可被用于限制浪费在部分静止画面的码率。

### --qp-max &lt;int&gt; or &lt;int&gt;:&lt;int&gt;:&lt;int&gt;

设置最大 QP 值为 &lt;I 帧&gt;:&lt;P 帧&gt;:&lt;B 帧&gt;。CQP模式下将会被忽略。

可用于在视频的任何部分保持一定的图像质量，即使这样做可能超过指定的码率。

### --vbr-quality &lt;float&gt;

当使用 VBR 模式时设置输出质量。 (0.0-51.0, 0 = 自动)

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

设置参考距离。在硬件编码中，增加参考帧将会对画面质量和压缩率产生微小的影响。

### --weightp

启用带权 P 帧。[仅在 H.264 下有效]

### --aq

在帧内启用自适应量化（Adaptive Quantization）。（默认：关）

### --aq-temporal

在帧间启用自适应量化（Adaptive Quantization）。（默认：关）

### --aq-strength &lt;int&gt;

指定自适应量化强度（Adaptive Quantization Strength）。(1 (弱) - 15 (强), 0 = 自动)

### --bref-mode &lt;string&gt; [仅在 H.264 下有效]

指定 B 帧参考模式。

- disabled (默认)
- each
  将每一 B 帧作为参考
- middle
  只有第 (B帧数量)/2 个 B-frame 会被作为参考  

### --direct &lt;string&gt; [仅在 H.264 下有效]

指定 H.264 B Direct 模式.
- auto (默认)
- disabled
- spatial
- temporal

### --(no-)adapt-transform [仅在 H.264 下有效]

启用（或禁用）H.264 的自适应变换模式（Adaptive Transform Mode）。

### --mv-precision &lt;string&gt;
运动向量（Motion Vector）准确度 / 默认：自动。

- auto ... 自动
- Q-pel ... 1/4 像素精度 (高精确度)
- half-pel ... 1/2 像素精度
- full-pel ... 1 像素精度 (低精确度)

### --level &lt;string&gt;

设置编码器等级（Level）。如果未指定，将会自动设置。

```
h264: auto, 1, 1 b, 1.1, 1.2, 1.3, 2, 2.1, 2.2, 3, 3.1, 3.2, 4, 4.1, 4.2, 5, 5.1, 5.2
hevc: auto, 1, 2, 2.1, 3, 3.1, 4, 4.1, 5, 5.1, 5.2, 6, 6.1, 6.2
```

### --profile &lt;string&gt;

设置编码器 profile。如果未指定，将会自动设置。

```
h264:  auto, baseline, main, high, high444
hevc:  auto, main, main10, main444
```

### --tier &lt;string&gt;  [仅在 HEVC 下有效]

设置编码器 tier。
```
hevc:  main, high
```

### --sar &lt;int&gt;:&lt;int&gt;

设置 SAR 比例（Pixel Aspect Ratio）。

### --dar &lt;int&gt;:&lt;int&gt;

设置 DAR 比例 (Screen Aspect Ratio)。

### --fullrange

以全范围 YUV 编码.

### --videoformat &lt;string&gt;
```
  undef, ntsc, component, pal, secam, mac
```
### --colormatrix &lt;string&gt;
```
  undef, bt709, smpte170m, bt470bg, smpte240m, YCgCo, fcc, GBR, bt2020nc, bt2020c
```
### --colorprim &lt;string&gt;
```
  undef, bt709, smpte170m, bt470m, bt470bg, smpte240m, film, bt2020
```
### --transfer &lt;string&gt;
```
  undef, bt709, smpte170m, bt470m, bt470bg, smpte240m, linear,
  log100, log316, iec61966-2-4, bt1361e, iec61966-2-1,
  bt2020-10, bt2020-12, smpte2084, smpte428, arib-srd-b67
```

### --chromaloc &lt;int&gt;

为输出流设置色度位置标志（Chroma Location Flag），从0到5。
 
默认: 0 = 未指定

### --max-cll &lt;int&gt;,&lt;int&gt; [仅在 HEVC 下有效]

设置 MaxCLL and MaxFall，单位nits。

```
--max-cll 1000,300
```

### --master-display &lt;string&gt; [仅在 HEVC 下有效]

设置 Mastering display 数据。
```
示例: --master-display G(13250,34500)B(7500,3000)R(34000,16000)WP(15635,16450)L(10000000,1)
```

### --pic-struct
插入 Picture Timing SEI。

### --cabac [仅在 H.264 下有效]

使用 CABAC. （默认：开）

### --cavlc [仅在 H.264 下有效]

使用 CAVLC. （默认：关）

### --bluray [仅在 H.264 下有效]

Bluray输出。（默认：关）

### --(no-)deblock [仅在 H.264 下有效]

启用去色块（Deblock）滤镜。（默认：开）

### --cu-max &lt;int&gt; [仅在 HEVC 下有效]
### --cu-min &lt;int&gt; [仅在 HEVC 下有效]

设置最大和最小编码单元（Coding Unit, CU）大小。可以设置8、16、32。

**由于已知这些设置会降低画面质量，不推荐使用这些设置**

## 输入输出 / 音频 / 字幕设置 

### --input-analyze &lt;int&gt;

设置 libav 分析视频时使用的视频长度，单位为秒。默认为5秒。如果音频 / 字幕轨等没有被正确检测，尝试增加该值（如60）。

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

### --input-format &lt;string&gt;

为 avhw / avsw 读取器指定输入格式。

### -f, --output-format &lt;string&gt;

为混流器指定输出格式。

由于输出格式可以通过输出文件的扩展名自动确定，通常情况下无需指定，但你可以使用该选项强行指定输出格式。

可用的格式可以通过[--check-formats](#--check-formats)查询。

要将H.264 / HEVC输出为基本流，请指定“raw”。

### --video-tag  &lt;string&gt;
指定视频标签。
```
 -o test.mp4 -c hevc --video-tag hvc1
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

### --audio-codec [[&lt;int&gt;?]&lt;string&gt;[:&lt;string&gt;=&lt;string&gt;][,&lt;string&gt;=&lt;string&gt;],...]

使用指定的编码器编码音频轨。如果没有设定编码器，将会自动使用最合适的编码器。可用的编码器可以通过[--check-encoders](#--check-codecs---check-decoders---check-encoders)查询。

你也可以指定抽取并编码的音频轨（1,2,...）。

你也可以指定编码器参数。
```
示例 1: 把所有音频轨编码为mp3
--audio-codec libmp3lame

示例 2: 把第二根音频轨编码为aac
--audio-codec 2?aac

Example 3: 为 "aac_coder" 添加 "twoloop" 参数可以提升低码率下的音频质量。
--audio-codec aac:aac_coder=twoloop
```

### --audio-bitrate [&lt;int&gt;?]&lt;int&gt;

指定音频码率。

你也可以指定抽取并编码的音频轨（1,2,...）。

```
示例 1: --audio-bitrate 192 (设置音频轨码率为 192kbps)
示例 2: --audio-bitrate 2?256 (设置第二根音频轨的码率为 256kbps)
```

### --audio-profile [&lt;int&gt;?]&lt;string&gt;

指定音频编码器的profile。

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

### --audio-samplerate [&lt;int&gt;?]&lt;int&gt;

设定音频采样率，单位Hz。

你也可以指定抽取并编码的音频轨（1,2,...）。

```
示例 1: --audio-bitrate 44100 (把音频转换为 44100Hz)
示例 2: --audio-bitrate 2?22050 (把第二根音频轨的音频转换为 22050Hz)
```

### --audio-resampler &lt;string&gt;

指定用于混合音频声道和采样率转换的引擎。

- swr ... swresampler (默认)
- soxr ... sox resampler (libsoxr)

### --audio-file [&lt;int&gt;?][&lt;string&gt;]&lt;string&gt;

把音频轨抽取到指定的路径。输出格式由输出文件后缀名自动确定。仅当使用 avhw / avsw 读取器时有效。

你也可以指定抽取并编码的音频轨（1,2,...）。

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

### --audio-ignore-decode-error &lt;int&gt;

忽略持续的音频解码错误，在阈值允许范围内继续转码。无法被正确的解码的音频部分将会使用空白音频替代。

默认值为10。
```
Example1: 五个连续音频解码错误后退出转码
--audio-ignore-decode-error 5

Example2: 任何解码错误后退出转码
--audio-ignore-decode-error 0
```

### --audio-source &lt;string&gt;

混流指定的外部音频文件。

### --chapter &lt;string&gt;

使用章节文件设置章节信息。章节文件可以是 nero 或者 apple 格式。在 --chapter-copy 应用的情况下无法使用。


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

### --chapter-copy

从输入文件复制章节信息。

### --key-on-chapter

在章节分割处设置关键帧。

### --keyfile &lt;string&gt;

由文件指定关键帧位置（从0,1,2,...起）。文件应一行一个帧序号。

### --sub-copy [&lt;int&gt;[,&lt;int&gt;]...]

从输入文件复制字幕轨。仅当使用 avhw / avsw 读取器时有效。

你也可以指定需要抽取并复制的字幕轨（1,2,...）。

支持 PGS / srt / txt / ttxt 格式字幕。

```
示例: 复制第一、第二根字幕轨
--sub-copy 1,2
```

### --caption2ass

启用内部 caption2ass 处理。需要 Caption.dll。

输出格式需要为 mkv。

**支持格式**
- ass (默认)
- srt

### -m, --mux-option &lt;string1&gt;:&lt;string2&gt;

为混流器传递附加参数。用&lt;string1&gt;指定参数名，用&lt;string2&gt;指定参数值。

```
示例: 输出 HLS
-i <input> -o test.m3u8 -f hls -m hls_time:5 -m hls_segment_filename:test_%03d.ts --gop-len 30
```

### --avsync &lt;string&gt;
  - cfr (default)
    输入文件将会被认为是固定帧率，输入的PTS（Presentation Time Stamp）将不会被检查。

  - forcecfr
    检查输入文件的PTS（Presentation Time Stamp），重复或者移除帧来保持固定帧率，以维持与音频的同步。

  - vfr  
    遵循输入文件的时间戳并启用可变帧率输出。仅当使用 avsw/avhw 读取器时有效。无法和 --trim 一起使用。

## Vpp 设置

### --vpp-deinterlace &lt;string&gt;

激活硬件反交错器。仅当使用[--avhw](#--avhw-string)(硬件解码)时有效，并且需要为[--interlace](#--interlace-string)选项指定tff或bff。

- none ... 不反交错 (默认)
- normal ... 标准 60i → 30p 反交错.
- adaptive ... 与 normal 相同
- bob ... 60i → 60p 交错.

对于 IT(inverse telecine), 使用 [--vpp-afs](#--vpp-afs-param1value1param2value2).

### --vpp-rff

RFF（Reflect the Repeat Field）标记。可以解决由于 RFF 引发的 avsync 错误。仅当使用[--avhw](#--avhw-string)时有效。

2或以上的值不被支持（仅支持 rff = 1）。同时，无法与[--trim](#--trim-intintintintintint)和[--vpp-deinterlace](#--vpp-deinterlace-string)一起使用。

### --vpp-afs [&lt;param1&gt;=&lt;value1&gt;][,&lt;param2&gt;=&lt;value2&gt;],...

激活自动场偏移（Activate Auto Field Shift, AFS）反交错。
Activate Auto Field Shift (AFS) deinterlacer.

**参数**
- top=&lt;int&gt;
- bottom=&lt;int&gt;
- left=&lt;int&gt;
- right=&lt;int&gt;
  clip out the range to decide field shift.

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
  注意：启用该选项会生成可变帧率视频。当混流由 NVEncC 完成时，时间码（timecode）将会被自动应用。但当使用未处理输出（Raw）时，你需要为 vpp-afs 添加 "timecode=true" 来输出时间码文件，然后混流。

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

- weightfile (默认: 使用内置文件)  
  指定权重参数文件。不指定的时候将会使用内置的数据。

```
示例：--vpp-nnedi field=auto,nns=64,nsize=32x6,qual=slow,prescreen=none,prec=fp32
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

### --vpp-colorspace [&lt;param1&gt;=&lt;value1&gt;][,&lt;param2&gt;=&lt;value2&gt;],...  
进行色彩空间变换。

**参数**
- matrix=&lt;from&gt;:&lt;to&gt;  

```
  bt709, smpte170m, bt470bg, smpte240m, YCgCo, fcc, GBR, bt2020nc, bt2020c
```

- colorprim=&lt;from&gt;:&lt;to&gt;  
```
  bt709, smpte170m, bt470m, bt470bg, smpte240m, film, bt2020
```

- transfer=&lt;from&gt;:&lt;to&gt;  
```
  bt709, smpte170m, bt470m, bt470bg, smpte240m, linear,
  log100, log316, iec61966-2-4, iec61966-2-1,
  bt2020-10, bt2020-12, smpte2084, arib-srd-b67
```

- range=&lt;from&gt;:&lt;to&gt;  
```
  limited, full
```

- hdr2sdr=&lt;bool&gt;  
使用"Hable tone-mapping"将HDR10转换为SDR。 是从[hdr2sdr.py](https://gist.github.com/4re/34ccbb95732c1bef47c3d2975ac62395)移植的。

- source_peak=&lt;float&gt;  (默认: 1000.0)  

- ldr_nits=&lt;float&gt;  (默认: 100.0)  


```
例1: BT.709(fullrange) -> BT.601
--vpp-colorspace matrix=smpte170m:bt709,range=full:limited
例2: hdr2sdr
--vpp-colorspace hdr2sdr=true,source_peak=1000.0,ldr_nits=100.0
```

### --vpp-select-every &lt;int&gt;[,&lt;param1&gt;=&lt;int&gt;]

选取每隔特定数量的帧进行输出。

**参数**
- step=&lt;int&gt;
- offset=&lt;int&gt; (默认：0)

```
示例一： (即 "select even"): --vpp-select-every 2
示例二： (即 "select odd "): --vpp-select-every 2,offset=1
```

### --vpp-resize &lt;string&gt;

设置缩放算法。

标记为"○"的算法需要[NPP library](https://developer.nvidia.com/npp)（nppi64_10.dll），仅在 x64 版本支持。使用这些算法，你需要另外下载 nppi64_10.dll 并把它和 NVEncC64.exe 放置在同一目录。


| 选项名 | 描述 | 是否需要 nppi64_10.dll |
|:---|:---|:---:|
| default  | 自动选择 | |
| bilinear | 线性插值 | |
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

需要 nppi64_80.dll，仅在 x64 版本支持。


### --vpp-unsharp [&lt;param1&gt;=&lt;value1&gt;][,&lt;param2&gt;=&lt;value2&gt;],...
反锐化滤镜，用于边缘和细节增强。

**Parameters**
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

### --vpp-tweak [&lt;param1&gt;=&lt;value1&gt;][,&lt;param2&gt;=&lt;value2&gt;],...

**参数**
- brightness=&lt;float&gt; (默认=0.0, -1.0 - 1.0)  

- contrast=&lt;float&gt; (默认=1.0, -2.0 - 2.0)  

- gamma=&lt;float&gt; (默认=1.0, 0.1 - 10.0)  

- saturation=&lt;float&gt; (默认=1.0, 0.0 - 3.0)  

- hue=&lt;float&gt; (默认=0.0, -180 - 180)  

```
示例:
--vpp-tweak brightness=0.1,contrast=1.5,gamma=0.75
```

### --vpp-pad &lt;int&gt,&lt;int&gt,&lt;int&gt,&lt;int&gt

为左、上、右、下边缘添加内边距，单位像素。

### --vpp-delogo &lt;string&gt;[,&lt;param1&gt;=&lt;value1&gt;][,&lt;param2&gt;=&lt;value2&gt;],...

指定需要消除的Logo的Logo文件及设置。Logo文件支持".lgd"、".ldp"、".ldp2"格式。

**参数**
- select=&lt;string&gt;  

使用下列参数之一来指定使用Logo包中的Logo。

- Logo 名
- 编号 (1, 2, ...)
- ini 配置文件内的自动选择
```
 [LOGO_AUTO_SELECT]
 logo<num>=<pattern>,<logo name>
 ```

 示例:
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

调整Logo位置。（单位像素，精度1/4像素，格式X:Y）

- depth &lt;int&gt;

调整Logo透明度，默认为128。

- y=&lt;int&gt;  
- cb=&lt;int&gt;  
- cr=&lt;int&gt;  

调整Logo的颜色组成。

- auto_fade=&lt;bool&gt;  

动态调整淡入淡出值。默认为false。

- auto_nr=&lt;bool&gt;  

动态调整降噪强度。默认为false。

- nr_area=&lt;int&gt;  

Logo附近需要降噪的区域。（默认为0（关闭），取值为0-3）

- nr_value=&lt;int&gt;  

Logo附近降噪的强度。（默认为0（关闭），取值为0-4）

- log=&lt;bool&gt;  

使用auto_fade、auto_nr时，输出淡入淡出值变化日志。

```
示例:
--vpp-delogo logodata.ldp2,select=delogo.auf.ini,auto_fade=true,auto_nr=true,nr_value=3,nr_area=1,log=true
```

### --vpp-perf-monitor

监视每个vpp滤镜的性能，输出应用的滤镜处理每帧的平均时间。开启该选项可能会对整体编码性能产生轻微影响。


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

- error ... 只输出错误
- warn ... 输出错误和警告
- info ... 显示编码信息 (默认)
- debug ... 输出更多信息，主要用于调试
- trace ... 输出每一帧的信息（慢）

### --max-procfps &lt;int&gt;

设置转码速度上线。默认为0（不限制）。

当你想要同时编码多个流，并且不想其中一个占用全部 CPU 或 GPU 资源时可以使用该选项。

```
示例: 限制最大转码速度为 90fps
--max-procfps 90
```

### --perf-monitor [&lt;string&gt;][,&lt;string&gt;]...

输出性能信息。可以从下表中选择要输出的信息的名字，默认为全部。


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