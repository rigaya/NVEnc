# NVEnc Release Notes

## 9.27

- Support saving/restoring per-project output settings in AviUtl2.
- Speed up [--vpp-kfm](./NVEncC_Options.en.md#--vpp-kfm-param1value1param2value2) / [--vpp-rtgmc](./NVEncC_Options.en.md#--vpp-rtgmc-param1value1) / [--vpp-degrain](./NVEncC_Options.en.md#--vpp-degrain-param1value1) by adding SAD-threshold early termination for motion search.
- Speed up [--vpp-kfm](./NVEncC_Options.en.md#--vpp-kfm-param1value1param2value2) by removing redundant processing.
- Handle mixed RFF sources properly in [--vpp-kfm](./NVEncC_Options.en.md#--vpp-kfm-param1value1param2value2).
- Fix [--vpp-kfm](./NVEncC_Options.en.md#--vpp-kfm-param1value1param2value2) mode=24 stall and timestamps.
- Fix chroma degrain analysis in [--vpp-rtgmc](./NVEncC_Options.en.md#--vpp-rtgmc-param1value1). ( #786 )
- Fix high bit-depth variance calculation in [--vpp-nnedi](./NVEncC_Options.en.md#--vpp-nnedi-param1value1param2value2). ( #779 )

## 9.26

- Add [--vpp-lenscorrection](./NVEncC_Options.en.md#--vpp-lenscorrection-param1value1param2value2) and [--vpp-v360](./NVEncC_Options.en.md#--vpp-v360-param1value1param2value2).
- Add mask input and multi-frame (temporal) support to [--vpp-onnx](./NVEncC_Options.en.md#--vpp-onnx-param1value1param2value2).
- Add custom parameter support to [--vpp-libplacebo-shader](./NVEncC_Options.en.md#--vpp-libplacebo-shader-param1value1param2value2).
- Fix [--vpp-onnx](./NVEncC_Options.en.md#--vpp-onnx-param1value1param2value2) fp16 being treated as fp32 with TensorRT.
- Speed up [--vpp-kfm](./NVEncC_Options.en.md#--vpp-kfm-param1value1param2value2) / [--vpp-degrain](./NVEncC_Options.en.md#--vpp-degrain-param1value1).
- Fix and speed up [--vpp-rtgmc](./NVEncC_Options.en.md#--vpp-rtgmc-param1value1).
- Fix 10-bit overflow in [--vpp-rtgmc](./NVEncC_Options.en.md#--vpp-rtgmc-param1value1).
- Add statistics to [--vship-ssimulacra2](./NVEncC_Options.en.md#--vship-ssimulacra2) log output.
- Separate TensorRT caches by runtime environment and input conditions.

## 9.25

- Add Windows named pipe support. ( #785 )
- Improve accuracy of [--vpp-kfm](./NVEncC_Options.en.md#--vpp-kfm-param1value1param2value2).
- Speed up [--vpp-degrain](./NVEncC_Options.en.md#--vpp-degrain-param1value1) / [--vpp-rtgmc](./NVEncC_Options.en.md#--vpp-rtgmc-param1value1) with zero-copy cache references and parallelized motion search.
- Update libvmaf and enable float. ( #783 )
- Add automatic CPU fallback for libvmaf. ( #783 )
- Fix [--vpp-libplacebo-tonemapping](./NVEncC_Options.en.md#--vpp-libplacebo-tonemapping-param1value1param2value2) rejecting bool strings for `use_dovi`.
- Fix potential issues.

## 9.24

- Improve [--vpp-kfm](./NVEncC_Options.en.md#--vpp-kfm-param1value1param2value2) performance for long processing.
- Extend temporal options in [--vpp-fft3d](./NVEncC_Options.en.md#--vpp-fft3d-param1value1param2value2).
- Improve diagnostics when ONNX Runtime fails to load.
- Avoid nan in CUDA [--vmaf](./NVEncC_Options.en.md#--vmaf-param1value1param2value2). ( #781 )
- Fix Linux VMAF CUDA support. ( #781 )
- Enable libvship build on Linux.
- Add SDR to HDR model support to [--vpp-onnx](./NVEncC_Options.en.md#--vpp-onnx-param1value1param2value2).
- Clarify error for non-existent VMAF json model. ( #783 )
- Add OpenVINO RIFE frame interpolation filter [--vpp-rife-ov](./NVEncC_Options.en.md#--vpp-rife-ov-param1value1param2value2).
- Add coring and hue range to [--vpp-tweak](./NVEncC_Options.en.md#--vpp-tweak-param1value1param2value2).
- Add interpolation mode and fix all in [--vpp-curves](./NVEncC_Options.en.md#--vpp-curves-param1value1param2value2).
- Add detailed parameters to [--vpp-hqdering](./NVEncC_Options.en.md#--vpp-hqdering-param1value1param2value2).
- Add chroma to [--vpp-cas](./NVEncC_Options.en.md#--vpp-cas-param1value1param2value2).
- Add valid source size to [--vpp-descale](./NVEncC_Options.en.md#--vpp-descale-param1value1param2value2).
- Add keep and fix state management in [--vpp-mpdecimate](./NVEncC_Options.en.md#--vpp-mpdecimate-param1value1param2value2).
- Add detection parameters to [--vpp-ivtc](./NVEncC_Options.en.md#--vpp-ivtc-param1value1param2value2).
- Add planes to [--vpp-nnedi](./NVEncC_Options.en.md#--vpp-nnedi-param1value1param2value2).
- Add temporal radius and improve temporal processing in [--vpp-knn](./NVEncC_Options.en.md#--vpp-knn-param1value1param2value2).
- Fix yuv444→y410 conversion.
- Fix mask output in [--vpp-msmooth](./NVEncC_Options.en.md#--vpp-msmooth-param1value1param2value2).
- Fix chroma processing in [--vpp-vinverse](./NVEncC_Options.en.md#--vpp-vinverse-param1value1param2value2).
- Propagate Yadif copyFrameAsync error in [--vpp-yadif](./NVEncC_Options.en.md#--vpp-yadif-param1value1).
- Fix nvvfx output crop height check.
- Fix alpha input buffer width.
- Fix smooth QP table reference coordinate clamp.
- Fix Anime4K spline36 linear term in [--vpp-anime4k-shader](./NVEncC_Options.en.md#--vpp-anime4k-shader-param1value1param2value2).
- Fix AFS mie_spot8 4th element reference in [--vpp-afs](./NVEncC_Options.en.md#--vpp-afs-param1value1param2value2).
- Fix colorspace LUT3D green axis scale in [--vpp-colorspace](./NVEncC_Options.en.md#--vpp-colorspace-param1value1param2value2).
- Fix libplacebo radius warning argument in [--vpp-libplacebo-shader](./NVEncC_Options.en.md#--vpp-libplacebo-shader-param1value1param2value2).
- Fix libplacebo deband grain target in [--vpp-libplacebo-deband](./NVEncC_Options.en.md#--vpp-libplacebo-deband-param1value1param2value2).
- Fix delogo two-point least squares intercept in [--vpp-delogo](./NVEncC_Options.en.md#--vpp-delogo-stringparam1value1param2value2).
- Fix decimate diff grid height in [--vpp-decimate](./NVEncC_Options.en.md#--vpp-decimate-param1value1param2value2).
- Fix decomb stripe grid Y stride in [--vpp-decomb](./NVEncC_Options.en.md#--vpp-decomb-param1value1param2value2).
- Fix chroma reference line in RGB→YUV420 conversion.
- Apply denoise filter fixes.
- Fix threshold in [--vpp-convolution3d](./NVEncC_Options.en.md#--vpp-convolution3d-param1value1param2value2).

## 9.23

- Add libvmaf CUDA support. ( #781 )
- Improve precision of [--vpp-finedehalo](./NVEncC_Options.en.md#--vpp-finedehalo-param1value1param2value2). ( #777 )
- Mark [--vpp-finedehalo](./NVEncC_Options.en.md#--vpp-finedehalo-param1value1param2value2) as interlace unsupported. ( #782 )
- Fix frame corruption in [--vpp-ivtc](./NVEncC_Options.en.md#--vpp-ivtc-param1value1param2value2) expand/mixed path.
- Fix chroma joint-bilateral OOB in [--vpp-anime4k-shader](./NVEncC_Options.en.md#--vpp-anime4k-shader-param1value1param2value2) on YUV444.
- Fix SRCPTR clamp upper limit in [--vpp-afs](./NVEncC_Options.en.md#--vpp-afs-param1value1param2value2) map_filter.
- Fix OOB write in [--vpp-subburn](./NVEncC_Options.en.md#--vpp-subburn-param1value1param2value2) when subtitle position is negative.
- Fix [--vpp-deflicker](./NVEncC_Options.en.md#--vpp-deflicker-param1value1param2value2) disabled after scene change.
- Fix audio track selection fallback loop in `--avs` input.
- Fix LUT boundary clamp and spline coefficient reference in [--vpp-curves](./NVEncC_Options.en.md#--vpp-curves-param1value1param2value2).
- Fix constant-luminance YUV→RGB conversion using wrong inverse matrix in [--vpp-colorspace](./NVEncC_Options.en.md#--vpp-colorspace-param1value1param2value2).
- Fix YUV444/NV24 color conversion bugs.
- Fix degrain fallback analyze MV/SAD snapshot for scene-change delayed output.
- Fix candidate list in [--vpp-denoise-dct](./NVEncC_Options.en.md#--vpp-denoise-dct-param1value1param2value2).

## 9.22

- Improve post-tr2 correction in [--vpp-rtgmc](./NVEncC_Options.en.md#--vpp-rtgmc-param1value1). ( #777 )

## 9.21

- Add ONNX Runtime based filter [--vpp-onnx](./NVEncC_Options.en.md#--vpp-onnx-param1value1param2value2).
  - Model files can be downloaded from the link below.
    https://github.com/rigaya/HWEnc-onnx-models/releases
- Add Anime4K shader filter [--vpp-anime4k-shader](./NVEncC_Options.en.md#--vpp-anime4k-shader-param1value1param2value2).
- Fix chroma degrain processing in [--vpp-rtgmc](./NVEncC_Options.en.md#--vpp-rtgmc-param1value1).
- Fix source frame reference after RFF expansion in RTGMC.
- Add option to force `--parallel` with filters which use large memory. ([--parallel-force-large-memory-filters](./NVEncC_Options.en.md#--parallel-force-large-memory-filters), #780 )

## 9.20

- Add [--vpp-chromashift](./NVEncC_Options.en.md#--vpp-chromashift-param1value1param2value2).
- Add [--vpp-deblock](./NVEncC_Options.en.md#--vpp-deblock-param1value1param2value2).
- Add [--vpp-deflicker](./NVEncC_Options.en.md#--vpp-deflicker-param1value1param2value2).
- Add [--vpp-colorfix](./NVEncC_Options.en.md#--vpp-colorfix-param1value1param2value2).
- Add [--vpp-dehalo](./NVEncC_Options.en.md#--vpp-dehalo-param1value1param2value2), [--vpp-finedehalo](./NVEncC_Options.en.md#--vpp-finedehalo-param1value1param2value2), and [--vpp-hqdering](./NVEncC_Options.en.md#--vpp-hqdering-param1value1param2value2).
- Add [--vpp-maa](./NVEncC_Options.en.md#--vpp-maa-param1value1param2value2) filter.
- Add [--vpp-stab](./NVEncC_Options.en.md#--vpp-stab-param1value1param2value2).
- Add [--vpp-vinverse](./NVEncC_Options.en.md#--vpp-vinverse-param1value1param2value2).
- Add [--vpp-hqdn3d](./NVEncC_Options.en.md#--vpp-hqdn3d-param1value1param2value2), [--vpp-cas](./NVEncC_Options.en.md#--vpp-cas-param1value1param2value2), and [--vpp-descale](./NVEncC_Options.en.md#--vpp-descale-param1value1param2value2).
- Fix [--vpp-nlmeans](./NVEncC_Options.en.md#--vpp-nlmeans-param1value1param2value2) causing artifacts in some settings.
- Extend [--vpp-msmooth](./NVEncC_Options.en.md#--vpp-msmooth-param1value1param2value2), [--vpp-msharpen](./NVEncC_Options.en.md#--vpp-msharpen-param1value1param2value2), and [--vpp-warpsharp](./NVEncC_Options.en.md#--vpp-warpsharp-param1value1param2value2).
- Update [--vpp-ivtc](./NVEncC_Options.en.md#--vpp-ivtc-param1value1param2value2).
- Add fsr1 to [--vpp-resize](./NVEncC_Options.en.md#--vpp-resize-string-or-param1value1param2value2).
- Apply [--vpp-rtgmc](./NVEncC_Options.en.md#--vpp-rtgmc-param1value1) order to actual deinterlacing.
- Fix [--vpp-rtgmc](./NVEncC_Options.en.md#--vpp-rtgmc-param1value1) slower source-match chroma correction.
- Fix [--vpp-deflicker](./NVEncC_Options.en.md#--vpp-deflicker-param1value1param2value2) scene change detection.
- Fix [--vpp-chromashift](./NVEncC_Options.en.md#--vpp-chromashift-param1value1param2value2) auto detection budget.
- Adjust [--vpp-colorfix](./NVEncC_Options.en.md#--vpp-colorfix-param1value1param2value2).
- Fix [--vpp-kfm](./NVEncC_Options.en.md#--vpp-kfm-param1value1param2value2) realtime failure due to insufficient source cache.
- Fix [--vpp-kfm](./NVEncC_Options.en.md#--vpp-kfm-param1value1param2value2) frame pool exhaustion when retaining search luma.
- Fix insufficient reference buffer allocation in [--vpp-nnedi](./NVEncC_Options.en.md#--vpp-nnedi-param1value1param2value2).
- Correct mux timestamps when audio PTS is unset.
- Unify hqdn3d VPP option names.
- Fix pipeline control when many frames are output.

## 9.19

- Bundle nnedi3_weight.bin in the package.
- Fix vpp-kfm RTGMC preset medium error (vpp-nnedi nnsize=5) by reducing NNEDI register usage. ( #776 )
- Fix [--vpp-rtgmc](./NVEncC_Options.en.md#--vpp-rtgmc-param1value1) chroma_motion=false processing. ( #777 )
- Fix failed to copy edi side-data frame with chroma_motion=true, source_match=3.
- Fix Degrain analysis result reuse in [--vpp-rtgmc](./NVEncC_Options.en.md#--vpp-rtgmc-param1value1).
- Fix frame pool allocation in [--vpp-rtgmc-search-prefilter](./NVEncC_Options.en.md#--vpp-rtgmc-search-prefilter-param1value1).

## 9.18

- Add new high quality deinterlace filter [--vpp-kfm](./NVEncC_Options.en.md#--vpp-kfm-param1value1param2value2) which supports 24/30/60 mixed VFR. ( #677 )
- Add new high quality deinterlace filter [--vpp-rtgmc](./NVEncC_Options.en.md#--vpp-rtgmc-param1value1). ( #161, #218, #267, #677)
- Add [--vpp-deint-csp](./NVEncC_Options.en.md#--vpp-deint-csp-string) to specify CSP for deinterlace filters.
- Add new filter [--vpp-degrain](./NVEncC_Options.en.md#--vpp-degrain-param1value1).　( #161, #218, #267, #677)
- Update [--vpp-nnedi](./NVEncC_Options.en.md#--vpp-nnedi-param1value1param2value2) to new specification.
- Support odd crop values when output resolution must be even. ( #772 )
- Fix CUDA texture border handling in [--vpp-yadif](./NVEncC_Options.en.md#--vpp-yadif-param1value1).
- Fix audio desync when libavformat returns negative pts.

## 9.17

- Add [--vpp-bwdif](./NVEncC_Options.en.md#--vpp-bwdif-param1value1) and [--vpp-ivtc](./NVEncC_Options.en.md#--vpp-ivtc-param1value1param2value2).
- Add [--vpp-detailsharpen](./NVEncC_Options.en.md#--vpp-detailsharpen-param1value1param2value2). ( #762 )
- Add [--vpp-degrain](./NVEncC_Options.en.md#--vpp-degrain-param1value1) motion-compensated degrain filter.
- Fix crop processing for yuv444 input where edge pixels were not written correctly. ( #763 )
- Fix neroaacenc 2pass output.
- Fix [--lowlatency](./NVEncC_Options.en.md#--lowlatency) corrupted on Linux from 9.15.
- Extend AV1 [--qvbr](./NVEncC_Options.en.md#--qvbr--float) upper limit to 63.

## 9.16

- Improve subtitle burn-in for Blu-ray and MPEG-TS inputs. ( #756 )
- Fix libopus encoding for 5.1 / 7.1 channel layouts.

## 9.15

- Add [--vpp-msmooth](./NVEncC_Options.en.md#--vpp-msmooth-param1value1param2value2) and [--vpp-msharpen](./NVEncC_Options.en.md#--vpp-msharpen-param1value1param2value2).
- Add quality metric evaluation using vship. ([--vship-ssimulacra2]((./NVEncC_Options.en.md#--vship-ssimulacra2)), [--vship-butteraugli](./NVEncC_Options.en.md#--vship-butteraugli-param1value1param2value2), [--vship-cvvdp](./NVEncC_Options.en.md#--vship-cvvdp-param1value1param2value2))
- Minimize latency with [--lowlatency](./NVEncC_Options.en.md#--lowlatency) by automatically disabling output thread.
- Add sigmoid-related options and input colorspace specification to [--vpp-libplacebo-shader](./NVEncC_Options.en.md#--vpp-libplacebo-shader-param1value1param2value2).
- Display warning using [--vpp-libplacebo-shader](./NVEncC_Options.en.md#--vpp-libplacebo-shader-param1value1param2value2) in when conditions requiring res specification are detected.
- Reduce latency when using pipes.
- Fix subtitles not being passed to vpp-subburn when no audio processing is performed. ( #756 )

## 9.14

- Add [--vmaf](./NVEncC_Options.en.md#--vmaf-param1value1param2value2) support for Linux systems. ( #755 )
- Improve vapoursynth error messages.

## 9.13

- Improve multi-channel audio channel layout handling when encoding with [--audio-codec](./NVEncC_Options.en.md#--audio-codec-intstringstringstringstringstringstring). ( #671 )
- Update documents for Dolby Vision options ([--dolby-vision-profile](./NVEncC_Options.en.md#--dolby-vision-profile-string-hevc-av1), [--dolby-vision-rpu](./NVEncC_Options.en.md#--dolby-vision-rpu-string-hevc-av1), etc.). ( #738 )

## 9.12

- Fix for DTS:X not being copyable.

## 9.11

- Add option to append input command line parameters to `encoding_tool` in muxer metadata. ([--muxer-add-cmd](./NVEncC_Options.en.md#--muxer-add-cmd))
- Fix potential SIGPIPE(141) error termination in Linux multi-GPU environment.
- Fix error on finalization when encoding E-AC3. ( #706 )
- Avoid mixing GPUs with different B-frame availability when using [--parallel](./NVEncC_Options.en.md#--parallel-int-or-param1value1param2value2).
- Migrate Linux build to meson.

## 9.10

- Add feature to set --audio-bitrate to different value depending on audio channels. ( #743 )

## 9.09

- Add support for Vapoursynth API V4. ( #747 )

## 9.08

- Add option to encode only when input audio codec differs from codec specified by [--audio-codec](./NVEncC_Options.en.md#--audio-codec-intstringstringstringstringstringstring). ([--audio-encode-other-codec-only](./NVEncC_Options.en.md#--audio-encode-other-codec-only), #743)
- Remove restriction on dolby vision output. ( #738 )

## 9.07

- Fix error when encoding H.264 for RTMP/FLV output.
- Fix mkv output failure when encoding with [-c](./NVEncC_Options.en.md#-c---codec-string) av_libsvtav1. ( #733 )
- Add option to show preset/tune parameters ([--check-preset-params](./NVEncC_Options.en.md#--check-preset-params)).

## 9.06

- Fix --vpp-resize bilinear,spline*,lanczos* creating artifacts depending on resize ratio. ( #698, #737 )

## 9.05

- Add option to enable unidirect B frame for lossless encoding ([--unidirectb](./NVEncC_Options.en.md#--unidirectb)).
- Add tune option. ([--tune](./NVEncC_Options.en.md#--tune-string))
- Change [--ref](./NVEncC_Options.en.md#--ref-int) default to 4(H.264)/5(HEVC,AV1).
- Now defaults for options below should differ by [--preset](./NVEncC_Options.en.md#-u---preset) and [--tune](./NVEncC_Options.en.md#--tune-string).
  - [--weightp](./NVEncC_Options.en.md#--weightp)
  - [-b, --bframes](./NVEncC_Options.en.md#-b---bframes-int)
  - [--strict-gop](./NVEncC_Options.en.md#--strict-gop)
  - [--no-i-adapt](./NVEncC_Options.en.md#--no-i-adapt)
  - [--no-b-adapt](./NVEncC_Options.en.md#--no-b-adapt)
  - [--aq](./NVEncC_Options.en.md#--aq)
  - [--aq-temporal](./NVEncC_Options.en.md#--aq-temporal)
  - [--aq-strength](./NVEncC_Options.en.md#--aq-strength-int)
  - [--nonrefP](./NVEncC_Options.en.md#--nonrefp)
  - [--lookahead](./NVEncC_Options.en.md#--lookahead-int)
  - [--lookahead-level](./NVEncC_Options.en.md#--lookahead-level-int)
  - [--tf-level](./NVEncC_Options.en.md#--tf-level-int)
  - [--temporal-layers](./NVEncC_Options.en.md#--temporal-layers-int)
- Add option to fallback to 8bit encoding when 10bit encoding is not supported by the hardware.([--fallback-bitdepth](./NVEncC_Options.en.md#--fallback-bitdepth))

## 9.04

- Improve DX11 device initialization to not detect virtual/remote adaptors. ( #725 )
- Improve progress indicator when using [--parallel](./NVEncC_Options.en.md#--parallel-int-or-param1value1param2value2).
- Add support for using [--parallel](./NVEncC_Options.en.md#--parallel-int-or-param1value1param2value2) with multiple pipes.

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
    Example: [-c](./NVEncC_Options.en.md#-c---codec-string) av_libsvtav1 [--avcodec-prms](./NVEncC_Options.en.md#--avcodec-prms-string) "preset=6,crf=30,svtav1-params=enable-variance-boost=1:variance-boost-strength=2"
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
