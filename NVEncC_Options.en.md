
# NVEncC option list

**[日本語版はこちら＞＞](./NVEncC_Options.ja.md)**  
**[Chinese Version＞＞](./NVEncC_Options.zh-cn.md)**


## Command line example


### Basic commands
```Batchfile
NVEncC.exe [Options] -i <filename> -o <filename>
```

### More practical commands
#### example of using hw (cuvid) decoder
```Batchfile
NVEncC --avhw -i "<mp4(H.264/AVC) file>" -o "<outfilename.264>"
```

#### example of using hw (cuvid) decoder (interlaced)
```Batchfile
NVEncC --avhw --interlace tff -i "<mp4(H.264/AVC) file>" -o "<outfilename.264>"
```

#### avs (Avisynth) example (avs and vpy can also be read via vfw)
```Batchfile
NVEncC -i "<avsfile>" -o "<outfilename.264>"
```

#### example of pipe usage
```Batchfile
avs2pipemod -y4mp "<avsfile>" | NVEncC - y4m - i - - o "<outfilename.264>"
```

#### pipe usage from ffmpeg

```Batchfile
ffmpeg -y -i "<inputfile>" -an -pix_fmt yuv420p -f yuv4mpegpipe - | NVEncC --y4m -i - -o "<outfilename.264>"
```

#### Passing video & audio from ffmpeg
--> use "nut" to pass both video & audio thorough pipe.
```Batchfile
ffmpeg -y -i "<input>" <options for ffmpeg> -codec:a copy -codec:v rawvideo -pix_fmt yuv420p -f nut - | NVEncC --avsw -i - --audio-codec aac -o "<outfilename.mp4>"
```

## Option format

```
-<short option name>, --<option name> <argument>

The argument type is
- none
- <int>    ... use integer
- <float>  ... use decimal point
- <string> ... use character string

The argument with [] brackets are optional.

--(no-)xxx
If it is attached with --no-xxx, you get the opposite effect of --xxx.
Example 1: --xxx: enable xxx → --no-xxx: disable xxx
Example 2: --xxx: disable xxx → --no-xxx: enable xxx
```

## Display options

### -h, -? --help
Show help

### -v, --version
Show version of NVEncC

### --check-device
Show device of available GPU recognized by NVEnc

### --check-hw [&lt;int&gt;]
Check whether the specified device is able to run NVEnc. DeviceID: "0" will be checked if not specified.

### --check-features [&lt;int&gt;]
Show the information of features of the specified device. DeviceID: "0" will be checked if not specified.

### --check-environment
Show environment information recognized by NVEncC

### --check-codecs, --check-decoders, --check-encoders
Show available audio codec names

### --check-profiles &lt;string&gt;
Show profile names available for specified codec

### --check-formats
Show available output format

### --check-protocols
Show available protocols

### --check-filters
Show available audio filters

### --check-avversion
Show version of ffmpeg dll

## Basic encoding options

### -d, --device &lt;int&gt;
Specify the deviceId to be used with NVEnc. deviceID can be checked with [--check-device](#--check-device).

If unspecified, and you are running on multi-GPU environment, the device to be used will automatically selected, depending on following conditions...

- whether the device supports specified encoding
- if --avhw is specified, then check whether the device supports hw decoding for the input file
- if interlaced encoding is specified, then check if it is supported
- device with lower Video Engine Utilization will be favored
- device with lower GPU Utilization will be favored
- later generation GPU will be favored
- GPU with more cores will be favored

Utilization of the Video Engine and GPU is obtained using [NVML library](https://developer.nvidia.com/nvidia-management-library-nvml) in x64 version, and nvidia-smi.exe is executed in x86 version.

nvidia-smi is usually installed in "C:\Program Files\NVIDIA Corporation\NVSMI\nvidia-smi.exe" with the driver.


### -c, --codec &lt;string&gt;
Specify the output codec
 - h264 (default)
 - hevc

### -o, --output &lt;string&gt;
Set output file name, pipe output with "-"

### -i, --input &lt;string&gt;
Set input file name, pipe input with "-"

Table below shows the supported readers of NVEnc. When input format is not set,
reader used will be selected depending on the extension of input file.

**Auto selection of reader**  

| reader |  target extension |
|:---|:---|          
| Avisynth reader    | avs |
| VapourSynth reader | vpy |
| avi reader         | avi |
| y4m reader         | y4m |
| raw reader         | yuv |
| avhw/avsw reader | others |

**color format supported by reader**  

| reader | yuv420 | yuy2 | yuv422 | yuv444 | rgb24 | rgb32 |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| raw    |   ○   |      |        |        |       |       |
| y4m    |   ◎   |      |   ◎   |   ◎   |       |       |
| avi    |   ○   |  ○  |        |        |   ○  |   ○  |
| avs    |   ◎   |  ○  |   ◎   |   ◎   |   ○  |   ○  |
| vpy    |   ◎   |      |   ◎   |   ◎   |       |       |
| avhw   |   □   |      |        |   ◇   |       |       |
| avsw   |   ◎   |      |   ◎   |   ◎   |   ○  |   ○  |

◎ ... 8bit / 9bit / 10bit / 12bit / 14bit / 16bit supported  
◇ ... 8bit / 10bit / 12bit supported  
□ ... 8bit / 10bit supported  
○ ... support only 8 bits

### --raw
Set the input to raw format.
input resolution & input fps must also be set.

### --y4m
Read input as y4m (YUV4MPEG2) format.

### --avi
Read avi file using avi reader.

### --avs
Read Avisynth script file using avs reader.

### --vpy
Read VapourSynth script file using vpy reader.

### --avsw
Read input file using avformat + ffmpeg's sw decoder.

### --avhw
Read using avformat + cuvid hw decoder. Using this mode will provide maximum performance,
since entire transcode process will be run on the GPU.

**Codecs supported by avhw reader**  

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

○ ... supported  
× ... no support

### --interlace &lt;string&gt;
Set interlace flag of **input** frame.

Deinterlace is available through [--vpp-deinterlace](#--vpp-deinterlace-string) or [--vpp-afs](#--vpp-afs-param1value1param2value2). If deinterlacer is not activated for interlaced input, then interlaced encoding is performed.

- none ... progressive
- tff ... top field first
- bff ... Bottom Field First

### --video-track &lt;int&gt;
Set video track to encode in track id. Will be active when used with avhw/avsw reader.
 - 1 (default)  highest resolution video track
 - 2            next high resolution video track
    ...
 - -1           lowest resolution video track
 - -2           next low resolution video track
    ...

### --crop &lt;int&gt;,&lt;int&gt;,&lt;int&gt;,&lt;int&gt;
Number of pixels to cropped from left, top, right, bottom.

### --fps &lt;int&gt;/&lt;int&gt; or &lt;float&gt;
Set the input frame rate. Required for raw format.

### --input-res &lt;int&gt;x&lt;int&gt;
Set input resolution. Required for raw format.

### --output-res &lt;int&gt;x&lt;int&gt;
Set output resolution. When it is different from the input resolution, HW/GPU resizer will be activated automatically.

If not specified, it will be same as the input resolution. (no resize)


## Encode Mode Options

The default is CQP (Constant quantization).

### --cqp &lt;int&gt; or &lt;int&gt;:&lt;int&gt;:&lt;int&gt;
Set the QP value of &lt;I frame&gt;:&lt;P frame&gt;:&lt;B frame&gt;

Generally, it is recommended to set the QP value to be I &lt; P &lt; B.

### --cbr &lt;int&gt;
### --cbrhq &lt;int&gt;
### --vbr &lt;int&gt;
### --vbrhq &lt;int&gt;
Set bitrate in kbps.

Constant quality mode can be used by "--vbrhq 0 --vbr-quality &lt;float&gt;".

## Other Options for Encoder

### -u, --preset
Encode qulaity preset.
- default
- performance
- quality

### --output-depth &lt;int&gt;
Set output bit depth.
- 8 ... 8 bits (default)
- 10 ... 10 bits

### --lossless
Perform lossless output. (Default: off)

### --max-bitrate &lt;int&gt;
Maximum bitrate (in kbps).

### --qp-init &lt;int&gt; or &lt;int&gt;:&lt;int&gt;:&lt;int&gt;
Set the initial QP value with &lt;I frame&gt;:&lt;P frame&gt;:&lt;B frame&gt;. This option will be ignored in CQP mode.

These QP values will be applied at the beginning of encoding. 
Use this option when you want to adjust the image quality at the beginning of the movie,
which sometimes gets unstable in CBR/VBR modes.

### --qp-min &lt;int&gt; or &lt;int&gt;:&lt;int&gt;:&lt;int&gt;
Set the minimum QP value with &lt;I frame&gt;:&lt;P frame&gt;:&lt;B frame&gt;. This option will be ignored in CQP mode. 

It could be used to suppress bitrate being used unnecessarily to a portion of movie with still image.

### --qp-max &lt;int&gt; or &lt;int&gt;:&lt;int&gt;:&lt;int&gt;
Set the maximum QP value to &lt;I frame&gt;:&lt;P frame&gt;:&lt;B frame&gt;. This option will be ignored in CQP mode.

It could be used to maintain certain degree of image quality in any part of the video, even if doing so may exceed the specified bitrate.

### --vbr-quality &lt;float&gt;
Set target quality when using VBR mode. (0.0-51.0, 0 = automatic)

### --dynamic-rc &lt;int&gt;:&lt;int&gt;:&lt;int&gt;&lt;int&gt;,&lt;param1&gt;=&lt;value1&gt;[,&lt;param2&gt;=&lt;value2&gt;],...  
Change the rate control mode and rate control params within the specified range of output frames.

**required parameters**
It is required to specify one of the params below.
- [cqp](./NVEncC_Options.en.md#--cqp-int-or-intintint)=&lt;int&gt; or cqp=&lt;int&gt;:&lt;int&gt;:&lt;int&gt;  
- [cbr](./NVEncC_Options.en.md#--cbr-int)=&lt;int&gt;  
- [cbrhq](./NVEncC_Options.en.md#--cbrhq-int)=&lt;int&gt;  
- [vbr](./NVEncC_Options.en.md#--vbr-int)=&lt;int&gt;  
- [vbrhq](./NVEncC_Options.en.md#--vbrhq-int)=&lt;int&gt;  

**additional parameters**
- [max-bitrate](./NVEncC_Options.en.md#--max-bitrate-int)=&lt;int&gt;  
- [vbr-quality](./NVEncC_Options.en.md#--vbr-quality-float)=&lt;float&gt;  

```
Example1: Encode by vbrhq(12000kbps) in output frame range 3000-3999,
          encode by constant quality mode(29.0) in output frame range 5000-5999,
          and encode by constant quality mode(25.0) on other frame range.
  --vbrhq 0 --vbr-quality=25.0 --dynamic-rc 3000:3999,vbrhq=12000 --dynamic-rc 5000:5999,vbrhq=0,vbr-quality=29.0

Example2: Encode by vbrhq(6000kbps) to output frame number 2999,
          and encode by vbrhq(12000kbps) from output frame number 3000 and later.
  --vbrhq 6000 --dynamic-rc start=3000,vbrhq=12000
```

### --lookahead &lt;int&gt;
Enable lookahead, and specify its target range by the number of frames. (0 - 32)  
This is useful to improve image quality, allowing adaptive insertion of I and B frames.

### --no-i-adapt
Disable adaptive I frame insertion when lookahead is enabled.

### --no-b-adapt
Disable adaptive B frame insertion when lookahead is enabled.

### --strict-gop
Force fixed GOP length.

### --gop-len &lt;int&gt;
Set maximum GOP length. When lookahead is off, this value will always be used. (Not variable, fixed GOP)

### -b, --bframes &lt;int&gt;
Set the number of consecutive B frames.

### --ref &lt;int&gt;
Set the reference distance (max=16).  

### --multiref-l0 &lt;int&gt;  
### --multiref-l1 &lt;int&gt;  
Set max number of reference frames in reference picture list L0/L1 (max=7).

### --weightp
Enable weighted P frames. [H.264 only]

### --nonrefp
enable automatic insertion of non-reference P-frames.

### --aq
Enable adaptive quantization in frame (spatial). (Default: off)

### --aq-temporal
Enable adaptive quantization between frames (temporal). (Default: off)

### --aq-strength &lt;int&gt;
Specify the AQ strength. (1 (weak) - 15 (strong), 0 = auto)

### --bref-mode &lt;string&gt;
Specify B frame reference mode.
- disabled (default)
- each
  use each B frames as references  
- middle
  Only(Number of B-frame)/2 th B-frame will be used for reference  

### --direct &lt;string&gt; [H.264 only]
Specify H.264 B Direct mode.
- auto (default)
- disabled
- spatial
- temporal

### --(no-)adapt-transform [H.264 only]
Enable (or disable) adaptive transform mode of H.264.

### --mv-precision &lt;string&gt;
Motion vector accuracy / default: auto
- auto ... automatic
- Q-pel ... 1/4 pixel accuracy (high precision)
- half-pel ... 1/2 pixel precision
- full-pel ... 1 pixel accuracy (low accuracy)

### --slices &lt;int&gt;
Set number of slices.

### --level &lt;string&gt;
Specify the Level of the codec to be encoded. If not specified, it will be automatically set.
```
h264: auto, 1, 1 b, 1.1, 1.2, 1.3, 2, 2.1, 2.2, 3, 3.1, 3.2, 4, 4.1, 4.2, 5, 5.1, 5.2
hevc: auto, 1, 2, 2.1, 3, 3.1, 4, 4.1, 5, 5.1, 5.2, 6, 6.1, 6.2
```

### --profile &lt;string&gt;
Specify the profile of the codec to be encoded. If not specified, it will be automatically set.
```
h264:  auto, baseline, main, high, high444
hevc:  auto, main, main10, main444
```

### --tier &lt;string&gt;  [HEVC only]
Specify the tier of the codec.
```
hevc:  main, high
```

### --sar &lt;int&gt;:&lt;int&gt;
Set SAR ratio (pixel aspect ratio).

### --dar &lt;int&gt;:&lt;int&gt;
Set DAR ratio (screen aspect ratio).

### --fullrange
Encode as full range YUV.

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
Set chroma location flag of the output bitstream from values 0 ... 5.  
default: 0 = unspecified

### --max-cll &lt;int&gt;,&lt;int&gt; [HEVC only]
Set MaxCLL and MaxFall in nits. 
```
--max-cll 1000,300
```

### --master-display &lt;string&gt; [HEVC only]
Set Mastering display data.
```
Example: --master-display G(13250,34500)B(7500,3000)R(34000,16000)WP(15635,16450)L(10000000,1)
```

### --dhdr10-info &lt;string&gt; [HEVC only]
Apply HDR10+ dynamic metadata from specified json file. Requires [hdr10plus_gen.exe](https://github.com/rigaya/hdr10plus_gen) module  additionally.

### --aud
Insert Access Unit Delimiter NAL.

### --pic-struct
Insert picture timing SEI.

### --cabac [H.264 only]
Use CABAC. (Default: on)

### --cavlc [H.264 only]
Use CAVLC. (Default: off)

### --bluray [H.264 only]
Perform output for Bluray. (Default: off)

### --(no-)deblock [H.264 only]
Enable deblock filter. (Default: on)

### --cu-max &lt;int&gt; [HEVC only]
### --cu-min &lt;int&gt; [HEVC only]
Specify the maximum and minimum size of CU respectively. 8, 16, 32 can be specified.
**Since it is known that image quality may be degraded when this option is used, it is recommended not to use these options.**

## IO / Audio / Subtitle Options

### --input-analyze &lt;int&gt;
Specify the length in seconds that libav parses for file analysis. The default is 5 (sec).
If audio / subtitle tracks etc. are not detected properly, try increasing this value (eg 60).

### --trim &lt;int&gt;:&lt;int&gt;[,&lt;int&gt;:&lt;int&gt;][,&lt;int&gt;:&lt;int&gt;]...
Encode only frames in the specified range.

```
Example 1: --trim 0:1000,2000:3000    (encode from frame #0 to #1000 and from frame #2000 to #3000)
Example 2: --trim 2000:0              (encode from frame #2000 to the end)
```

### --seek [&lt;int&gt;:][&lt;int&gt;:]&lt;int&gt;[.&lt;int&gt;]
The format is hh:mm:ss.ms. "hh" or "mm" could be omitted. The transcode will start from the time specified.

Seeking by this option is not exact but fast, compared to [--trim](#--trim-intintintintintint). If you require exact seek, use [--trim](#--trim-intintintintintint).
```
Example 1: --seek 0:01:15.400
Example 2: --seek 1:15.4
Example 3: --seek 75.4
```

### --input-format &lt;string&gt;
Specify input format for avhw / avsw reader.

### -f, --output-format &lt;string&gt;
Specify output format for muxer.

Since the output format is automatically determined by the output extension, it is usually not necessary to specify it, but you can force the output format with this option.

Available formats can be checked with [--check-formats](#--check-formats). To output H.264 / HEVC as an Elementary Stream, specify "raw".

### --video-track &lt;int&gt;
Set video track to encode by resolution. Will be active when used with avhw/avsw reader.
 - 1 (default)  highest resolution video track
 - 2            next high resolution video track
    ...
 - -1           lowest resolution video track
 - -2           next low resolution video track
    ...
    
### --video-streamid &lt;int&gt;
Set video track to encode in stream id.

### --video-tag &lt;string&gt;
Specify video tag.
```
 -o test.mp4 -c hevc --video-tag hvc1
```

### --audio-copy [&lt;int&gt;[,&lt;int&gt;]...]
Copy audio track into output file. Available only when avhw / avsw reader is used.

If it does not work well, try encoding with [--audio-codec](#--audio-codec-intstring), which is more stable.

You can also specify the audio track (1, 2, ...) to extract.

```
Example: Copy all audio tracks
--audio-copy

Example: Extract track numbers #1 and #2
--audio-copy 1,2
```

### --audio-codec [[&lt;int&gt;?]&lt;string&gt;[:&lt;string&gt;=&lt;string&gt;][,&lt;string&gt;=&lt;string&gt;],...]
Encode audio track with the codec specified. If codec is not set, most suitable codec will be selected automatically. Codecs available could be checked with [--check-encoders](#--check-codecs---check-decoders---check-encoders).

You can also specify the audio track (1, 2, ...) to extract.

Also, you can specify params for audio encoder.
```
Example 1: encode all audio tracks to mp3
--audio-codec libmp3lame

Example 2: encode the 2nd track of audio to aac
--audio-codec 2?aac

Example 3: set param "aac_coder" to "twoloop" which will improve quality at low bitrate for aac encoder
--audio-codec aac:aac_coder=twoloop
```

### --audio-bitrate [&lt;int&gt;?]&lt;int&gt;
Specify the bitrate in kbps when encoding audio.

You can also specify the audio track (1, 2, ...) to extract.
```
Example 1: --audio-bitrate 192 (set bitrate of audio track to 192 kbps)
Example 2: --audio-bitrate 2?256 (set bitrate of 2nd audio track to to 256 kbps)
```

### --audio-profile [&lt;int&gt;?]&lt;string&gt;
Specify audio codec profile when encoding audio.

### --audio-stream [&lt;int&gt;?][&lt;string1&gt;][:&lt;string2&gt;]
Separate or merge audio channels.
Audio tracks specified with this option will always be encoded. (no copying available)

By comma(",") separation, you can generate multiple tracks from the same input track.

**format**

Specify the track to be processed by &lt;int&gt;.

Specify the channel to be used as input by &lt;string1&gt;. If omitted, input will be all the input channels.

Specify the output channel format by &lt;string2&gt;. If omitted, all the channels of &lt;string1&gt; will be used.

```
Example 1: --audio-stream FR,FL
Separate left and right channels of "dual mono" audio track, into two mono audio tracks.

Example 2: --audio-stream :stereo
Convert any audio track to stereo.

Example 3: --audio-stream 2?5.1,5.1:stereo
While encoding the 2nd 5.1 ch audio track of the input file as 5.1 ch,
another stereo downmixed audio track will be generated
from the same source audio track.
```

**Available symbols**
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
Specify the sampling frequency of the sound in Hz.
You can also specify the audio track (1, 2, ...) to extract.
```
Example 1: --audio-bitrate 44100 (converting sound to 44100 Hz)
Example 2: --audio-bitrate 2?22050 (Convert the second track of voice to 22050 Hz)
```

### --audio-resampler &lt;string&gt;
Specify the engine used for mixing audio channels and sampling frequency conversion.
- swr ... swresampler (default)
- soxr ... sox resampler (libsoxr)

### --audio-file [&lt;int&gt;?][&lt;string&gt;]&lt;string&gt;
Extract audio track to the specified path. The output format is determined automatically from the output extension. Available only when avhw / avsw reader is used.

You can also specify the audio track (1, 2, ...) to extract.
```
Example: extract audio track number #2 to test_out2.aac
--audio-file 2?"test_out2.aac"
```

[&lt;string&gt;] allows you to specify the output format.
```
Example: Output in adts format without extension
--audio-file 2?adts:"test_out2"  
```

### --audio-filter [&lt;int&gt;?]&lt;string&gt;
Apply filters to audio track. Filters could be slected from [link](https://ffmpeg.org/ffmpeg-filters.html#Audio-Filters).

You can also specify the audio track (1, 2, ...) to filter.

```
Example 1: --audio-filter volume=0.2  (lowering the volume)
Example 2: --audio-filter 2?volume=-4db (lowering the volume of the 2nd track)
```

### --audio-ignore-decode-error &lt;int&gt;
Ignore the consecutive audio decode error, and continue transcoding within the threshold specified. The portion of audio which could not be decoded properly will be replaced with silence.

The default is 10.

```
Example1: Quit transcoding for a 5 consecutive audio decode error.
--audio-ignore-decode-error 5

Example2: Quit transcoding for a single audio decode error.
--audio-ignore-decode-error 0
```

### --audio-source &lt;string&gt;[:[&lt;int&gt;?][;&lt;param1&gt;=&lt;value1&gt;][;&lt;param2&gt;=&lt;value2&gt;]...][:...]
Mux an external audio file specified.

**params** 
- copy  
  Copy audio track.

- codec=&lt;string&gt;  
  Encode audio to specified audio codec.

- profile=&lt;string&gt;  
  Specify audio codec profile when encoding audio.

- bitrate=&lt;int&gt;  
  Specify audio bitrate in kbps.
  
- samplerate=&lt;int&gt;  
  Specify audio sampling rate.

- enc_prm=&lt;string&gt;  
  Specify params for audio encoder.

- filter=&lt;string&gt;  
  Specify filters for audio.

```
Example1: --audio-source "<audio_file>":copy
Example2: --audio-source "<audio_file>":1?codec=aac;bitrate=256:2?codec=aac;bitrate=192
```

### --chapter &lt;string&gt;
Set chapter in the (separate) chapter file.
The chapter file could be in nero format, apple format or matroska format. Cannot be used with --chapter-copy.

nero format  
```
CHAPTER01=00:00:39.706
CHAPTER01NAME=chapter-1
CHAPTER02=00:01:09.703
CHAPTER02NAME=chapter-2
CHAPTER03=00:01:28.288
CHAPTER03NAME=chapter-3
```

apple format (should be in utf-8)  
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

matroska形式 (hould be in utf-8)  
[Other Samples&gt;&gt;](https://github.com/nmaier/mkvtoolnix/blob/master/examples/example-chapters-1.xml)
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
Copy chapters from input file.

### --key-on-chapter
Set keyframes on chapter position.

### --keyfile &lt;string&gt;
Set keyframes on frames (starting from 0, 1, 2, ...) specified in the file.
There should be one frame ID per line.

### --sub-source &lt;string&gt;
Read subtitle from the specified file and mux into the output file.

### --sub-copy [&lt;int&gt;[,&lt;int&gt;]...]
Copy subtitle tracks from input file. Available only when avhw / avsw reader is used.
It is also possible to specify subtitle tracks (1, 2, ...) to extract with [&lt;int&gt;].

Supported subtitles are PGS / srt / txt / ttxt.

```
Example: Copy subtitle track #1 and #2
--sub-copy 1,2
```

### --caption2ass [&lt;string&gt;]
Enable caption2ass process. This feature requires Caption.dll.  

supported formats ... srt (default), ass

### --data-copy [&lt;int&gt;[,&lt;int&gt;]...]
Copy data stream from input file. Available only when avhw / avsw reader is used.

### -m, --mux-option &lt;string1&gt;:&lt;string2&gt;
Pass optional parameters to muxer. Specify the option name in &lt;string1&gt, and the option value in &lt;string2&gt;.

```
Example: Output for HLS
-i <input> -o test.m3u8 -f hls -m hls_time:5 -m hls_segment_filename:test_%03d.ts --gop-len 30
```

### --avsync &lt;string&gt;
  - cfr (default)
    The input will be assumed as CFR and input pts will not be checked.

  - forcecfr
    Check pts from the input file, and duplicate or remove frames if required to keep CFR, so that synchronization with the audio could be maintained. Please note that this could not be used with --trim.

  - vfr  
    Honor source timestamp and enable vfr output. Only available for avsw/avhw reader.

## Vpp Options

### --vpp-deinterlace &lt;string&gt;
Activate hw deinterlacer. Available only when used with [--avhw](#--avhw-string)(hw decode) and [--interlace](#--interlace-string) tff or [--interlace](#--interlace-string) bff is specified.

- none ... no deinterlace (default)
- normal ... standard 60i → 30p interleave cancellation.
- adaptive ... same as normal
- bob ... 60i → 60p interleaved.

for IT(inverse telecine), use [--vpp-afs](#--vpp-afs-param1value1param2value2).

### --vpp-rff
Reflect the Repeat Field Flag. The avsync error caused by rff could be solved. Available only when [--avhw](#--avhw-string) is used.

rff of 2 or more will not be supported (only  supports rff = 1). Also, it can not be used with [--trim](#--trim-intintintintintint), [--vpp-deinterlace](#--vpp-deinterlace-string).

### --vpp-afs [&lt;param1&gt;=&lt;value1&gt;][,&lt;param2&gt;=&lt;value2&gt;],...
Activate Auto Field Shift (AFS) deinterlacer.

**parameters**
- top=&lt;int&gt;
- bottom=&lt;int&gt;
- left=&lt;int&gt;
- right=&lt;int&gt;
  clip out the range to decide field shift.

- method_switch=&lt;int&gt;  (0 - 256)  
  threshold to swicth field shift algorithm. 

- coeff_shift=&lt;int&gt;  (0 - 256)  
  threshold for field shift, with bigger value, more field shift will be occurred.

- thre_shift=&lt;int&gt;  (0 - 1024)  
  threshold for stripe detection which will be used on shift decision. Lower value will result more stripe detection.

- thre_deint=&lt;int&gt;   (0 - 1024)  
  threshold for stripe detection which will be used on deinterlacing. Lower value will result more stripe detection.

- thre_motion_y=&lt;int&gt;  (0 - 1024)  
- thre_motion_c=&lt;int&gt;  (0 - 1024)  
  threshold for motion detection. Lower value will result more motion detection. 

- level=&lt;int&gt;  (0 - 4)  
  Select how to remove the stripes. 

| level | process | target | decription |
|:---|:---|:---|:---|
| 0 | none  | | Stripe removing process will not be done.<br>New frame generated by field shift will be the output. |
| 1 | triplication | all pixels | Blend previous field into new frame generated by field shift.<br>Stripe caused be motion will all become afterimage. |
| 2 | duplicate | stripe-detected pixels | Blend previous field into new frame generated by field shift, only on stripe detected pixels.<br>Should be used for movies with little motion. |
| 3 (default) | duplicate  | motion-detected pixels | Blend previous field into new frame generated by field shift, only on motion detected pixels. <br>This mode can preserve more edges or small letters compared to level 2. | 
| 4 | interpolate | motion-detected pixels | On motion detected pixels, drop one field, and generate pixel by interpolating from the other field.<br>There will be no afterimage, but the vertical resolution of pixels with motion will halved. |

- shift=&lt;bool&gt;  
  Enable field shift.

- drop=&lt;bool&gt;  
  drop frame which has shorter display time than "1 frame". Note that enabling this option will generate VFR (Variable Frame Rate) output.
  When muxing is done by NVEncC, the timecode will be applied automatically. However, when using raw output,
  you will need output timecode file by adding "timecode=true" to vpp-afs option,
  and mux the timecode file later.

- smooth=&lt;bool&gt;  
  Smoothen picture display timing.

- 24fps=&lt;bool&gt;  
  Force 30fps -> 24fps conversion.

- tune=&lt;bool&gt;  
  When this options is set true, the output will be the result of motion and stripe detection, shown by the color below.

| color | description |
|:---:|:---|
| dark blue | motion was detected |
| grey | stripe was detected|
| light blue | motion & stripe was detected |

- rff=&lt;bool&gt;   
  When this options is set true, rff flag from input will be checked, and when there is progressive frame coded with rff, then deinterlacing will not be applied.

- log=&lt;bool&gt;  
  Generate log of per frame afs status (for debug).

- preset=&lt;string&gt;  
  Parameters will be set as below.

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
example: same as --vpp-afs preset=24fps
--vpp-afs preset=anime,method_switch=92,thre_shift=448,24fps=true
```

### --vpp-nnedi [&lt;param1&gt;=&lt;value1&gt;][,&lt;param2&gt;=&lt;value2&gt;],...  
nnedi deinterlacer.

**parameters**
- field

  - auto (default)  
    Generate latter field from first field.
  - top  
    Generate bottom field using top field.
  - bottom  
    Generate top field using bottom field.

- nns  (default: 32)  
  Neurons of neural net.
  - 16, 32, 64, 128, 256

- nsize  (default: 32x4)  
  Area size which neural net uses to generate a pixel.
  - 8x6, 16x6, 32x6, 48x6, 8x4, 16x4, 32x4

- quality  
  quality settings.

  - fast (default)

  - slow  
    "slow" uses another neural net and blends 2 outputs from different network to enhance quality.

- prescreen
  
  - none  
    No pre-screening is done and all pixels will be generated by neural net.

  - original
  - new  (default)  
    Runs prescreener to determine which pixel to apply neural net, other pixels will be generated from simple interpolation. 

  - original_block
  - new_block  (default)  
    GPU optimized version of "original", "new". Applies screening based on block, and not pixel.

- errortype  
  Select weight parameter for neural net.
  - abs  (default)  
    Use weight trained to minimize absolute error.
  - square  
    Use weight trained to minimize square error.
  
- prec  
  Select precision.
  - auto (default)  
    Use fp16 whenever it is available and will be faster, otherwise use fp32.
  
  - fp16  
    Force to use fp16. x64 only.
  
  - fp32  
    Force to use fp32.
    
  
- weightfile  
  Set path of weight file. By default (not specified), internal weight params will be used.

  
```
example: --vpp-nnedi field=auto,nns=64,nsize=32x6,qual=slow,prescreen=none,prec=fp32
```

### --vpp-yadif [&lt;param1&gt;=&lt;value1&gt;]
Yadif deinterlacer.

**parameters**
- mode

  - auto (default)  
    Generate latter field from first field.
  - tff  
    Generate bottom field using top field.
  - bff  
    Generate top field using bottom field.
  - bob   
    Generate one frame from each field.
  - bob_tff   
    Generate one frame from each field assuming top field first.
  - bob_bff   
    Generate one frame from each field assuming bottom field first.
    
### --vpp-colorspace [&lt;param1&gt;=&lt;value1&gt;][,&lt;param2&gt;=&lt;value2&gt;],...  
Converts colorspace of the video. Available on x64 version.

**parameters**
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

- hdr2sdr=&lt;string&gt;  
  Enables HDR10 to SDR by selected tone-mapping.  

  - none (default)  
    hdr2sdr processing is disabled.
  
  - hable  
    Trys to preserve both bright and dark detailes, but with rather dark result.
    You may specify addtional params (a,b,c,d,e,f,w) for the hable tone-mapping function below.  

    hable(x) = ( (x * (a*x + c*b) + d*e) / (x * (a*x + b) + d*f) ) - e/f  
    output = hable( (source_peak / ldr_nits) * input ) / hable(w)
    
    defaults: a = 0.22, b = 0.3, c = 0.1, d = 0.2, e = 0.01, f = 0.3, w = 11.2

  - mobius  
    Trys to preserve contrast and colors while bright details might be removed.  
    - transition=&lt;float&gt;  (default: 0.3)  
      Threshold to move from linear conversion to mobius tone mapping.  
    - peak=&lt;float&gt;  (default: 1.0)  
      reference peak brightness
  
  - reinhard  
    - contrast=&lt;float&gt;  (default: 0.5)  
      local contrast coefficient  
    - peak=&lt;float&gt;  (default: 1.0)  
      reference peak brightness

- source_peak=&lt;float&gt;  (default: 1000.0)  

- ldr_nits=&lt;float&gt;  (default: 100.0)  
  Target brightness for hdr2sdr function.


```
example1: convert from BT.601 -> BT.709
--vpp-colorspace matrix=smpte170m:bt709

example2: using hdr2sdr (hable tone-mapping)
--vpp-colorspace hdr2sdr=hable,source_peak=1000.0,ldr_nits=100.0

example3: using hdr2sdr (hable tone-mapping) and setting the coefs (this is example for the default settings)
--vpp-colorspace hdr2sdr=hable,source_peak=1000.0,ldr_nits=100.0,a=0.22,b=0.3,c=0.1,d=0.2,e=0.01,f=0.3,w=11.2
```

### --vpp-select-every &lt;int&gt;[,&lt;param1&gt;=&lt;int&gt;]
select one frame per specified frames and create output.

**parameters**
- step=&lt;int&gt;
- offset=&lt;int&gt; (default: 0)

```
example1 (same as "select even"): --vpp-select-every 2
example2 (same as "select odd "): --vpp-select-every 2,offset=1
```

### --vpp-resize &lt;string&gt;
Specify the resizing algorithm.

Those with "○" in nppi64_10.dll use the [NPP library](https://developer.nvidia.com/npp), which supports x64 version only. To use those algorithms, you need to download nppi64_10.dll separately and place it in the same folder as NVEncC64.exe.

| option name | description | require nppi64_10.dll |
|:---|:---|:---:|
| default  | auto select | |
| bilinear | linear interpolation | |
| spline16 | 4x4 spline curve interpolation | |
| spline36 | 6x6 spline curve interpolation | |
| spline64 | 8x8 spline curve interpolation | |
| lanczos2 | 4x4 Lanczos resampling | |
| lanczos3 | 6x6 Lanczos resampling | |
| lanczos4 | 8x8 Lanczos resampling | |
| nn            | nearest neighbor | ○ |
| npp_linear    | linear interpolation by NPP library | ○ |
| cubic         | 4x4 cubic interpolation | ○ |
| super         | So called "super sampling" by NPP library | ○ |
| lanczos       | Lanczos interpolation                    | ○ |


### --vpp-knn [&lt;param1&gt;=&lt;value1&gt;][,&lt;param2&gt;=&lt;value2&gt;],...
Strong noise reduction filter.

**Parameters**
- radius=&lt;int&gt;  (default=3, 1-5)   
  radius of filter.

- strength=&lt;float&gt;  (default=0.08, 0.0 - 1.0)   
  Strength of the filter.

- lerp=&lt;float&gt;   (default=0.2, 0.0 - 1.0)  
  The degree of blending of the original pixel to the noise reduction pixel.

- th_lerp=&lt;float&gt;  (default=0.8, 0.0 - 1.0)  
  Threshold of edge detection. 

```
Example: slightly stronger than default
--vpp-knn radius=3,strength=0.10,lerp=0.1
```

### --vpp-pmd [&lt;param1&gt;=&lt;value1&gt;][,&lt;param2&gt;=&lt;value2&gt;],...
Rather weak noise reduction by modified pmd method, aimed to preserve edge while noise reduction.

**Parameters**
- apply_count=&lt;int&gt;  (default=2, 1- )  
  Number of times to apply the filter. 

- strength=&lt;float&gt;  (default=100, 0-100)  
  Strength of the filter. 

- threshold=&lt;float&gt;  (default=100, 0-255)  
  Threshold for edge detection. The smaller the value is, more will be detected as edge, which will be preserved.

```
Example: Slightly weak than default
--vpp-pmd apply_count=2,strength=90,threshold=120
```

### --vpp-gauss &lt;int&gt;
Specify the size of Gaussian filter, from 3, 5 or 7.  
It is necessary to install nppi64_80.dll, and could be used only in x64 version.

### --vpp-unsharp [&lt;param1&gt;=&lt;value1&gt;][,&lt;param2&gt;=&lt;value2&gt;],...
unsharp filter, for edge and detail enhancement.

**Parameters**
- radius=&lt;int&gt; (default=3, 1-9)  
  radius of edge / detail detection.

- weight=&lt;float&gt; (default=0.5, 0-10)  
  Strength of edge and detail emphasis. Larger value will result stronger effect.

- threshold=&lt;float&gt;  (default=10.0, 0-255)  
  Threshold for edge and detail detection.

```
Example: Somewhat stronger
--vpp-unsharp weight=1.0
```

### --vpp-edgelevel [&lt;param1&gt;=&lt;value1&gt;][,&lt;param2&gt;=&lt;value2&gt;],...
Edge level adjustment filter, for edge sharpening.

**Parameters**
- strength=&lt;float&gt; (default=5.0, -31 - 31)  
  Strength of edge sharpening. Larger value will result stronger edge sharpening.

- threshold=&lt;float&gt;  (default=20.0, 0 - 255)  
  Noise threshold to avoid enhancing noise. Larger value will treat larger luminance change as noise.

- black=&lt;float&gt;  (default=0.0, 0-31)  
  strength to enhance dark part of edges.

- white=&lt;float&gt;  (default=0.0, 0-31)  
  strength to enhance bright part of edges.

```
Example: Somewhat stronger (Aviutl version default)
--vpp-edgelevel strength=10.0,threshold=16.0,black=0,white=0

Example: Strengthening the black part of the outline
--vpp-edgelevel strength=5.0,threshold=24.0,black=6.0
```

### --vpp-deband [&lt;param1&gt;=&lt;value1&gt;][,&lt;param2&gt;=&lt;value2&gt;],...

**Parameters**
- range=&lt;int&gt; (default=15, 0-127)  
  Blur range. Samples to be used for blur are taken from pixels within this range.

- sample=&lt;int&gt; (default=1, 0-2)  
  - sample = 0
    Processing is performed by referring a pixel within "range".

  - sample = 1
    Blur processing is performed by referring total of 2 pixels, a pixel within "range" and its point symmetric pixel.

  - sample = 2
    Blur processing is performed by referring total of 4 pixels including 2 pixels within "range" and their point symmetric pixels.

- thre=&lt;int&gt; (set same threshold for y, cb & cr)
- thre_y=&lt;int&gt; (default=15, 0-31)
- thre_cb=&lt;int&gt; (default=15, 0-31)
- thre_cr=&lt;int&gt; (default=15, 0-31)  
  Threshold for y, cb, cr blur. If this value is high, the filter will be stronger, but thin lines and edges are likely to disappear.

- dither=&lt;int&gt;   (set same dither for y & c)
- dither_y=&lt;int&gt; (default=15, 0-31)
- dither_c=&lt;int&gt; (default=15, 0-31)  
  Dither strength of y & c.

- seed=&lt;int&gt;  
  Change of random number seed. (default = 1234)

- blurfirst (default=off)  
  Stronger effect could be expected, by processing blur first.
  However side effects may also become stronger, which might make thin lines to disappear.

- rand_each_frame (default=off)  
  Change the random number used by the filter every frame.

```
Example:
--vpp-deband range=31,dither=12,rand_each_frame
```

### --vpp-tweak [&lt;param1&gt;=&lt;value1&gt;][,&lt;param2&gt;=&lt;value2&gt;],...

**Parameters**
- brightness=&lt;float&gt; (default=0.0, -1.0 - 1.0)  

- contrast=&lt;float&gt; (default=1.0, -2.0 - 2.0)  

- gamma=&lt;float&gt; (default=1.0, 0.1 - 10.0)  

- saturation=&lt;float&gt; (default=1.0, 0.0 - 3.0)  

- hue=&lt;float&gt; (default=0.0, -180 - 180)  

```
Example:
--vpp-tweak brightness=0.1,contrast=1.5,gamma=0.75
```

### --vpp-pad &lt;int&gt;,&lt;int&gt;,&lt;int&gt;,&lt;int&gt;
add padding to left,top,right,bottom (in pixels)

### --vpp-subburn [&lt;param1&gt;=&lt;value1&gt;][,&lt;param2&gt;=&lt;value2&gt;],...
"Burn in" specified subtitle to the video. Text type subtitles will be rendered by [libass](https://github.com/libass/libass).

**Parameters**
- track=&lt;int&gt;  
  Select subtitle track of the input file to burn in, track count starting from 1. 
  Available when --avhw or --avsw is used.
  
- filename=&lt;string&gt;  
  Select subtitle file path to burn in.

- charcode=&lt;string&gt;  
  Specify subtitle charcter code to burn in, for text type sub.

- shaping=&lt;string&gt;  
  Rendering quality of text, for text type sub.  
  - simple
  - complex (default)

- scale=&lt;float&gt; (default=0.0 (auto))
  scaling multiplizer for bitmap fonts.  

- transparency=&lt;float&gt; (default=0.0, 0.0 - 1.0)
  adds additional transparency for subtitle.  

- brightness=&lt;float&gt; (default=0.0, -1.0 - 1.0)
  modifies brightness of the subtitle.  

- contrast=&lt;float&gt; (default=1.0, -2.0 - 2.0)
  modifies contrast of the subtitle.  

- ts_offset=&lt;float&gt; (default=0.0)
  add offset in seconds to the subtitle timestamps (for debug perpose).  

```
Example1: burn in subtitle from the track of the input file
--vpp-subburn track=1

Example2: burn in PGS subtitle from file
--vpp-subburn filename="subtitle.sup"

Example3: burn in ASS subtitle from file which charcter code is Shift-JIS
--vpp-subburn filename="subtitle.sjis.ass",charcode=sjis,shaping=complex
```

### --vpp-delogo &lt;string&gt;[,&lt;param1&gt;=&lt;value1&gt;][,&lt;param2&gt;=&lt;value2&gt;],...

**Parameters**
- select=&lt;string&gt;  
For logo pack, specify the logo to use with one of the following.
  - Logo name
  - Index (1, 2, ...)
  - Automatic selection ini file

For logo pack, specify the logo to use with one of the following.

- Logo name
- Index (1, 2, ...)
- Automatic selection ini file
```
 [LOGO_AUTO_SELECT]
 logo<num>=<pattern>,<logo name>
```

 Example:
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
Adjustment of logo position with 1/4 pixel accuracy in x:y direction.  

- depth &lt;int&gt;
Adjustment of logo transparency. Default 128.  

- y=&lt;int&gt;  
- cb=&lt;int&gt;  
- cr=&lt;int&gt;  
Adjustment of each color component of the logo.  

- auto_fade=&lt;bool&gt;  
Adjust fade value dynamically. default=false.  
  
- auto_nr=&lt;bool&gt;  
Adjust strength of noise reduction dynamically. default=false.  

- nr_area=&lt;int&gt;  
Area of noise reduction near logo. (default=0 (off), 0 - 3)  

- nr_value=&lt;int&gt;  
Strength of noise reduction near logo. (default=0 (off), 0 - 4)  

- log=&lt;bool&gt;  
auto_fade, auto_nrを使用した場合のfade値の推移をログに出力する。

```
例:
--vpp-delogo logodata.ldp2,select=delogo.auf.ini,auto_fade=true,auto_nr=true,nr_value=3,nr_area=1,log=true
```

### --vpp-perf-monitor
Monitor the performance of each vpp filter, and output the average per frame processing time of the applied filter(s). Note that the overall encoding performance may slightly be harmed.



## Other Options

### --cuda-schedule &lt;string&gt;
  Change the behavior of the CPU when waiting for GPU task completion. The default is auto.

- auto (default)
  Leave the mode decision to the driver of CUDA.

- spin
  Always keep the CPU monitoring the GPU task to finish. The latency of synchronization will be minimun, but will always utilize 100% of one logical CPU core.

- yeild
  Basically it is the same as spin, but switching to another running thread will be allowed.

- sync
  Sleep a thread until the end of the GPU task. Performance might decrease, but will reduce CPU utilization especially when decoding is done by HW.

### --output-buf &lt;int&gt;
Specify the output buffer size in MB. The default is 8 and the maximum value is 128.

The output buffer will store output data until it reaches the buffer size, and then the data will be written at once. Higher performance and reduction of file fragmentation on the disk could be expected.

On the other hand, setting too much buffer size could decrease performance, since writing such a big data to the disk will take some time. Generally, leaving this to default should be fine.

If a protocol other than "file" is used, then this output buffer will not be used.

### --output-thread &lt;int&gt;
Specify whether to use a separate thread for output.
- -1 ... auto (default)
- 0 ... do not use output thread
- 1 ... use output thread  
Using output thread increases memory usage, but sometimes improves encoding speed.

### --log &lt;string&gt;
Output the log to the specified file.

### --log-level &lt;string&gt;
Select the level of log output.

- error ... Display only errors
- warn ... Show errors and warnings
- info ... Display general encoding information (default)
- debug ... Output additional information, mainly for debug
- trace ... Output information for each frame (slow)

### --max-procfps &lt;int&gt;
Set the upper limit of transcode speed. The default is 0 (= unlimited).

This could be used when you want to encode multiple stream and you do not want one stream to use up all the power of CPU or GPU.
```
Example: Limit maximum speed to 90 fps
--max-procfps 90
```

### --perf-monitor [&lt;string&gt;][,&lt;string&gt;]...
Outputs performance information. You can select the information name you want to output as a parameter from the following table. The default is all (all information).

```
 all          ... monitor all info
 cpu_total    ... cpu total usage (%)
 cpu_kernel   ... cpu kernel usage (%)
 cpu_main     ... cpu main thread usage (%)
 cpu_enc      ... cpu encode thread usage (%)
 cpu_in       ... cpu input thread usage (%)
 cpu_out      ... cpu output thread usage (%)
 cpu_aud_proc ... cpu aud proc thread usage (%)
 cpu_aud_enc  ... cpu aud enc thread usage (%)
 cpu          ... monitor all cpu info
 gpu_load    ... gpu usage (%)
 gpu_clock   ... gpu avg clock
 vee_load    ... gpu video encoder usage (%)
 gpu         ... monitor all gpu info
 queue       ... queue usage
 mem_private ... private memory (MB)
 mem_virtual ... virtual memory (MB)
 mem         ... monitor all memory info
 io_read     ... io read  (MB/s)
 io_write    ... io write (MB/s)
 io          ... monitor all io info
 fps         ... encode speed (fps)
 fps_avg     ... encode avg. speed (fps)
 bitrate     ... encode bitrate (kbps)
 bitrate_avg ... encode avg. bitrate (kbps)
 frame_out   ... written_frames
```

### --perf-monitor-interval &lt;int&gt;
Specify the time interval for performance monitoring with [--perf-monitor](#--perf-monitor-stringstring) in ms (should be 50 or more). The default is 500.