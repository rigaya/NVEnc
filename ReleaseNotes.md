# NVEnc Release Notes

## 9.03

- Fix 9.02 did not incude NVEncNVSDKNGX.dll. ( #724 )

## 9.02

- Fix error when using both ngx-vsr and ngx-true-hdr together (issue since 9.00). ( #724 )
- Avoid unintended fps values when front of input file is corrupted.

## 9.01

- Improve handling when input files have negative pts.
- Improve quality of burned in subtitles in --vpp-subburn processing by changing libass initialization method. ( #717 )

## 9.00

- Add NVEnc.auo2 with native support for AviUtl2.
- Add feature to use filters with avcodec encoders.
  - Available with ```-c av_xxx```
    Example: [-c](./QSVEncC_Options.en.md#-c---codec-string) av_libsvtav1 [--avcodec-prms](./QSVEncC_Options.en.md#--avcodec-prms-string) "preset=6,crf=30,svtav1-params=enable-variance-boost=1:variance-boost-strength=2"
    Other usable options include av_libvvenc, av_libvpx-vp9, etc.
- Update ffmpeg libraries. (Windows)
  - ffmpeg 7.1+ (20240822) -> 8.0
  - libpng 1.6.44 -> 1.6.50
  - expat 2.6.2 -> 2.7.1
  - fribidi 1.0.11 -> 1.0.16
  - libogg 1.3.5 -> 1.3.6
  - libxml2 2.12.6 -> 2.14.5
  - libvpl 2.13.0 -> 2.15.0
  - libvpx 1.14.1 -> 1.15.2
  - dav1d 1.4.3 -> 1.5.1
  - libxxhash 0.8.2 -> 0.8.3
  - glslang 15.0.0 -> 15.4.0
  - dovi_tool 2.1.2 -> 2.3.1
  - libjpeg-turbo 2.1.0 -> 3.1.1
  - lcms2 2.16 -> 2.17
  - zimg 3.0.5 -> 3.0.6
  - libplacebo 7.349.0 -> 7.351.0
  - libsvtav1 3.1.0 (new!) x64 only
  - libvvenc 1.13.1 (new!) x64 only
  - libass 0.9.0 -> 0.17.4 (x64), 0.14.0 (x86)
  - harfbuzz 11.4.4 (new)
  - libunibreak 6.1 (new)
  - Remove mmt/tlv patch

## 8.11

- Added options to change CUDA optimization mode ([--cuda-stream](NVEncC_Options.en.md#--cuda-stream-int), [--cuda-mt](NVEncC_Options.en.md#--cuda-mt-int), #710)
  - To address CUDA_ERROR_MAP_FAILED which seems to occur on RTX50xx, multi-threaded calls to the same CUDA context are disabled by default (--cuda-mt 0)
- Fixed --vpp-rff not working properly in NVEnc 8.10.
- Add option for [--bitstream-padding](NVEncC_Options.en.md#--bitstream-padding) for AV1 CBR encoding ( #714 ).

## 8.10

- Fix filtering crushing when using with interlaced encoding.

## 8.09

- Updates for NVEnc.auo (AviUtl/AviUtl2 plugin).

## 8.08

- Fix processing in YUV444 for [--vpp-subburn](./NVEncC_Options.en.md#--vpp-subburn-string). ( #691 )
- Fix handling when end is omitted in [--dynamic-rc](./NVEncC_Options.en.md#--dynamic-rc-param1value1param2value2).

## 8.07

- Fix issues with raw output when using formats like yuv4mpegpipe. ( #699 )
- Fix potential freeze when using raw output.
- Add support for [--option-file](./NVEncC_Options.en.md#--option-file-string) on Linux.

## 8.06

- Fix performance degradation caused in 8.05. ( #696 )

## 8.05

- Add support for combining [--output-format](./NVEncC_Options.en.md#--output-format-string) with ```-c raw```. ( #693 )
  Now supports cases like ```-c raw --output-format nut```.
- Fix black/white processing in 10-bit depth for [--vpp-edgelevel](./NVEncC_Options.en.md#--vpp-edgelevel-param1value1param2value2).
- Improve interlace detection when using [--avsw](./NVEncC_Options.en.md#--avsw-string). ( #688 )

## 8.04

- Add ```inverse_tone_mapping``` option to [--vpp-libplacebo-tonemapping](./NVEncC_Options.en.md#--vpp-libplacebo-tonemapping-param1value1param2value2).
- Fix error when using ```st2094-10``` and ```st2094-40``` for ```tonemapping_function``` in [--vpp-libplacebo-tonemapping](./NVEncC_Options.en.md#--vpp-libplacebo-tonemapping-param1value1param2value2).
- Fix GPU selection defaulting to the first GPU when performance counter information is not available.
- Fix [--vpp-colorspace](./NVEncC_Options.en.md#--vpp-colorspace-param1value1param2value2) creating green line when input is interlaced.
- Add [--task-perf-monitor](./NVEncC_Options.en.md#--task-perf-monitor) to collect per task time comsumption in main thread.
- Adjust log output format.

## 8.03

- Improve audio and video synchronization to achieve more uniform mixing when muxing with subtitles or data tracks.
- Improve invalid input data hadling to avoid freeze when "failed to run h264_mp4toannexb bitstream filter" error occurs.
  Now properly exits with error.
- Add support for uyvy as input color format. ( #678 )
- Fix application freezing when using readers other than avhw.
- Automatically disable --parallel when number of encoders is 1 when using ```--parallel auto```.

## 8.02

- Fix vpp-resize ngx-vsr, libplaceo* not working in 8.01. ( #683 )

## 8.01

- Fix insufficient frame buffer causing error termination when using readers other than avhw (issue since 8.00beta1).
- Fix crash on process termination in Linux environment (issue since 8.00beta1).
- Fix hw decode not working in Linux environment (issue since 8.00beta1).
- Improve stability of Vulkan initialization in Linux environment.
- Avoid unnecessary Dolby Vision RPU conversion.
- Add detailed logging for errors during Dolby Vision RPU conversion.
- Update documentation.

## 8.00beta7

- Fix crush when using ([--parallel](./NVEncC_Options.en.md#--parallel-int-or-param1value1param2value2)).

## 8.00beta6

- Add parallel encoding feature with file splitting. ([--parallel](./NVEncC_Options.en.md#--parallel-int-or-param1value1param2value2))
- Add support for ISO 639-2 T-codes in language code specification. ( #674 )
- Continue processing even when DirectX11/Vulkan initialization fails. ( #675 )
- Fix timestamps occasionally becoming incorrect when using --seek with certain input files.
- Increase priority of GPUs with multiple encoders in auto GPU selection.
- Fix potential freeze when encoder output thread encounters an error.
- Fix potential freeze when encoder terminates with an error.
- Fix incorrect handling of pts for frames before keyframe when decoding from middle of OpenGOP encoded files.

## 8.00beta5

- Fix [--dolby-vision-rpu](https://github.com/rigaya/NVEnc/blob/master/NVEncC_Options.en.md#--dolby-vision-rpu-string) in AV1 encoding. ( #672 )

## 8.00beta4

- Fix some codecs not being able to decode with avsw since 8.00b2.
- Fix interlaced encoding not working when using filters since 8.00b1.
- Add 10.0, 10.1, 10.2, 10.4 options to [--dolby-vision-profile](https://github.com/rigaya/NVEnc/blob/master/NVEncC_Options.en.md#--dolby-vision-profile-string-hevc-av1). ( #672 )

## 8.00beta3

- Fix [--dolby-vision-profile](https://github.com/rigaya/NVEnc/blob/master/NVEncC_Options.en.md#--dolby-vision-profile-string-hevc-av1) not working with readers other than avhw/avsw. ( #663 )
- Fix memory allocation failure when using yuv422 output. ( #670 )

## 8.00beta2

- Improve auto gpu selection for multi (NVIDIA) GPU environments.

## 8.00beta1

- Add support for NVENC SDK 13.0.
  Driver 570.00 or later is required.
  - add support for H.264/AV1 in [--tf-level](https://github.com/rigaya/NVEnc/blob/master/NVEncC_Options.en.md#--tf-level-int).
  - Add forced_4 to [--split-enc](https://github.com/rigaya/NVEnc/blob/master/NVEncC_Options.en.md#--split-enc-string).
  - Support HEVC [--temporal-layers](https://github.com/rigaya/NVEnc/blob/master/NVEncC_Options.en.md#--temporal-layers-int).
  - Add experimental support for yuv422 output. (but untested, as I have no Blackwell GPUs)
- Update CUDA for Windows build to CUDA 11.8.
  - Requires driver 452.39 or later.
  - Requires CC3.5 or later GPUs.
    - NVIDIA GPUs GeForce Maxwell generation or later and some Kepler (GTX Titan, 780(Ti), 730-710)
  - GPUs that do not support CC3.5 (GTX770, 760, 740, 6xx) are no longer supported.
- No longer supports builds with CUDA 10 or earlier.
  - Builds for Ubuntu 18.04 removed.
- Noe NVEnc will be able to handle "frame transfer from CPU to GPU", "filtering", and "frame submission to encoder" in parallel.
- Merge AV1 [--max-temporal-layers](https://github.com/rigaya/NVEnc/blob/master/NVEncC_Options.en.md#--temporal-layers-int) to [--temporal-layers](https://github.com/rigaya/NVEnc/blob/master/NVEncC_Options.en.md#--temporal-layers-int) to be same as other codecs.
- Improve dolby vision rpu handling. ( #663 ) 

## 7.82

- Now AV1 [--level](https://github.com/rigaya/NVEnc/blob/master/NVEncC_Options.en.md#--level-string) can be specified again.
- Added checks for [--max-bitrate](https://github.com/rigaya/NVEnc/blob/master/NVEncC_Options.en.md#--max-bitrate-int) and [--ref](https://github.com/rigaya/NVEnc/blob/master/NVEncC_Options.en.md#--ref-int) to avoid errors when specifying [--level](https://github.com/rigaya/NVEnc/blob/master/NVEncC_Options.en.md#--level-string).
- Avoid --qvbr not working properly when setting [--max-bitrate](https://github.com/rigaya/NVEnc/blob/master/NVEncC_Options.en.md#--max-bitrate-int) too big. ( #486 )
- Fix some case that audio not being able to play when writing to mkv using --audio-copy.
- Now more detailed encoder error information will be shown.


## 7.81

- Avoid width field in mp4 Track Header Box getting 0 when SAR is undefined. ( #680 )

## 7.80

- - Fix some of the paramters of [--vpp-libplacebo-tonemapping](https://github.com/rigaya/NVEnc/blob/master/NVEncC_Options.en.md#--vpp-libplacebo-tonemapping-param1value1param2value2) not working properly.

## 7.79

- Fix documents for --vpp-resize nvvfx-superres ( #658 ).
- Fix --trim being offset for a few frames when input file is a "cut" file (which does not start from key frame) and is coded using OpenGOP.