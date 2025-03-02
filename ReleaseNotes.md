# NVEnc Release Notes


## 8.00beta5

- Improved processing when using [--dolby-vision-rpu](https://github.com/rigaya/NVEnc/blob/master/NVEncC_Options.en.md#--dolby-vision-rpu-string) in AV1 encoding. ( #672 )

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