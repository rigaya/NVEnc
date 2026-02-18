project('nvencc', ['cpp', 'cuda'],
  version: '8.00',
  license: 'MIT',
  default_options: [
    'cpp_std=c++17',
    'buildtype=release',
    'warning_level=1',
  ]
)

# ==============================================================================
# コンパイラ設定
# ==============================================================================
cpp = meson.get_compiler('cpp')
cuda = meson.get_compiler('cuda')
fs = import('fs')

# プラットフォームチェック
is_linux = host_machine.system() == 'linux'
is_x86_64 = host_machine.cpu_family() == 'x86_64'

if not is_linux
  error('This meson.build is for Linux only')
endif

# ==============================================================================
# CUDA パス自動検出
# ==============================================================================
# CUDA_PATH環境変数、または /usr/local/cuda を使用
cuda_path = get_option('nvenc_cuda_dir')
if cuda_path == ''
  cuda_path_env = run_command('sh', '-c', 'echo $CUDA_PATH', check: false).stdout().strip()
  if cuda_path_env != ''
    cuda_path = cuda_path_env
  elif fs.is_dir('/usr/local/cuda')
    cuda_path = '/usr/local/cuda'
  else
    error('CUDA not found. Set CUDA_PATH or cuda_path option.')
  endif
endif
message('Using CUDA path: ' + cuda_path)

cuda_lib_dir = cuda_path / 'lib64'
cuda_bin_dir = cuda_path / 'bin'

# ==============================================================================
# CUDAバージョン検出 & compute_xx 自動設定
# ==============================================================================
nvcc_path = cuda_bin_dir / 'nvcc'
cuda_version_result = run_command(nvcc_path, '--version', check: true)
cuda_version_output = cuda_version_result.stdout()

# "release X.Y" からバージョン抽出
cuda_ver_cmd = run_command('sh', '-c',
  'echo "' + cuda_version_output + '" | grep -oP "release \\K[0-9]+\\.[0-9]+"',
  check: false)
cuda_ver_str = cuda_ver_cmd.stdout().strip()
if cuda_ver_str == ''
  error('Could not parse CUDA version from nvcc output')
endif

cuda_ver_parts = cuda_ver_str.split('.')
cuda_ver_major = cuda_ver_parts[0].to_int()
cuda_ver_minor = cuda_ver_parts[1].to_int()
message('Detected CUDA version: @0@.@1@'.format(cuda_ver_major, cuda_ver_minor))

if cuda_ver_major < 10
  error('CUDA >= 10.0 required, but CUDA @0@.@1@ found.'.format(cuda_ver_major, cuda_ver_minor))
endif

# CUDAバージョンに応じたGPU code生成オプション
gpu_code_gen = []

# CUDA 10以下: compute_30も追加
if cuda_ver_major <= 10
  gpu_code_gen += ['--generate-code', 'arch=compute_30,code=[compute_30,sm_30]']
endif

# CUDA 12以下: compute_50, compute_61追加
if cuda_ver_major <= 12
  gpu_code_gen += ['--generate-code', 'arch=compute_50,code=[compute_50,sm_50]']
  gpu_code_gen += ['--generate-code', 'arch=compute_61,code=[compute_61,sm_61]']
endif

# 常に: compute_75
gpu_code_gen += ['--generate-code', 'arch=compute_75,code=[compute_75,sm_75]']

# CUDA 11.1以上: compute_86追加
if (cuda_ver_major == 11 and cuda_ver_minor >= 1) or cuda_ver_major >= 12
  gpu_code_gen += ['--generate-code', 'arch=compute_86,code=[compute_86,sm_86]']
endif

# CUDA 12.8以上: compute_120追加
if (cuda_ver_major == 12 and cuda_ver_minor >= 8) or cuda_ver_major >= 13
  gpu_code_gen += ['--generate-code', 'arch=compute_120,code=[compute_120,sm_120]']
endif

message('GPU code generation targets: ' + ' '.join(gpu_code_gen))

# ==============================================================================
# ビルドオプション
# ==============================================================================
enable_vapoursynth = get_option('enable_vapoursynth')
enable_avisynth = get_option('enable_avisynth')
enable_libass = get_option('enable_libass')
enable_libplacebo = get_option('enable_libplacebo')
enable_vulkan = get_option('enable_vulkan')

# ==============================================================================
# 依存関係検出
# ==============================================================================
# 必須依存
threads_dep = dependency('threads')
dl_dep = cpp.find_library('dl', required: true)

# CUDA
cuda_dep = dependency('cuda', version: '>=10.0', required: true)

# CUDA追加ライブラリ (Driver API, NPP)
# WSL2環境では /usr/lib/wsl/lib にlibcuda.soがある
cuda_driver_dep = cpp.find_library('cuda', dirs: [cuda_lib_dir, '/usr/lib/wsl/lib'], required: true)
nppif_dep = cpp.find_library('nppif_static', dirs: cuda_lib_dir, required: true)
nppig_dep = cpp.find_library('nppig_static', dirs: cuda_lib_dir, required: true)
nppc_dep = cpp.find_library('nppc_static', dirs: cuda_lib_dir, required: true)
culibos_dep = cpp.find_library('culibos', dirs: cuda_lib_dir, required: true)
rt_dep = cpp.find_library('rt', required: true)

# FFmpeg (必須)
libavutil_dep = dependency('libavutil', required: true)
libavcodec_dep = dependency('libavcodec', required: true)
libavformat_dep = dependency('libavformat', required: true)
libavfilter_dep = dependency('libavfilter', required: true)
libavdevice_dep = dependency('libavdevice', required: true)
libswresample_dep = dependency('libswresample', required: true)

ffmpeg_deps = [
  libavutil_dep,
  libavcodec_dep,
  libavformat_dep,
  libavfilter_dep,
  libavdevice_dep,
  libswresample_dep,
]

# オプション依存
vapoursynth_dep = dependency('vapoursynth', required: enable_vapoursynth)
avisynth_dep = dependency('avisynth', required: enable_avisynth)
libass_dep = dependency('libass', required: enable_libass)
libplacebo_dep = dependency('libplacebo', required: enable_libplacebo)
vulkan_dep = dependency('vulkan', required: enable_vulkan)

# ==============================================================================
# libdovi / libhdr10plus (システムにあれば使う、なければビルドディレクトリ内でビルド)
# ==============================================================================
source_root = meson.project_source_root()
build_root = meson.project_build_root()

# cargoパス検出 (ユーザーの~/.cargo/binを含む)
cargo_path_check = run_command('which', 'cargo', check: false)
have_cargo = cargo_path_check.returncode() == 0
if have_cargo
  message('Found cargo at: ' + cargo_path_check.stdout().strip())
else
  message('cargo not found, Rust libraries will be disabled if not installed system-wide')
endif

# libdovi検出
libdovi_dep = dependency('dovi', required: false)
if not libdovi_dep.found() and have_cargo
  # ビルドディレクトリ内でビルド
  libdovi_install_dir = build_root / 'rust_libs' / 'libdovi'

  message('libdovi not found, building from source...')

  # ビルドスクリプト実行
  libdovi_build_result = run_command(
    'bash', source_root / 'build_libdovi.sh', libdovi_install_dir,
    check: false)

  if libdovi_build_result.returncode() == 0
    # ビルド成功 - declare_dependencyで直接指定
    libdovi_lib_dir = libdovi_install_dir / 'lib' / 'x86_64-linux-gnu'
    libdovi_inc_dir = libdovi_install_dir / 'include'

    if fs.is_file(libdovi_lib_dir / 'libdovi.a')
      libdovi_lib = cpp.find_library('dovi',
        dirs: libdovi_lib_dir,
        static: true,
        required: false)
      if libdovi_lib.found()
        libdovi_dep = declare_dependency(
          dependencies: libdovi_lib,
          include_directories: include_directories(libdovi_inc_dir, is_system: true))
        message('libdovi built successfully')
      endif
    endif
  endif

  if not libdovi_dep.found()
    message('libdovi build failed, continuing without it')
  endif
endif

# libhdr10plus検出
libhdr10plus_dep = dependency('hdr10plus-rs', required: false)
if not libhdr10plus_dep.found() and have_cargo
  # ビルドディレクトリ内でビルド
  libhdr10plus_install_dir = build_root / 'rust_libs' / 'libhdr10plus'

  message('libhdr10plus not found, building from source...')

  # ビルドスクリプト実行
  libhdr10plus_build_result = run_command(
    'bash', source_root / 'build_libhdr10plus.sh', libhdr10plus_install_dir,
    check: false)

  if libhdr10plus_build_result.returncode() == 0
    # ビルド成功 - declare_dependencyで直接指定
    libhdr10plus_lib_dir = libhdr10plus_install_dir / 'lib' / 'x86_64-linux-gnu'
    libhdr10plus_inc_dir = libhdr10plus_install_dir / 'include'

    if fs.is_file(libhdr10plus_lib_dir / 'libhdr10plus-rs.a')
      libhdr10plus_lib = cpp.find_library('hdr10plus-rs',
        dirs: libhdr10plus_lib_dir,
        static: true,
        required: false)
      if libhdr10plus_lib.found()
        libhdr10plus_dep = declare_dependency(
          dependencies: libhdr10plus_lib,
          include_directories: include_directories(libhdr10plus_inc_dir, is_system: true))
        message('libhdr10plus built successfully')
      endif
    endif
  endif

  if not libhdr10plus_dep.found()
    message('libhdr10plus build failed, continuing without it')
  endif
endif

# ==============================================================================
# FFmpeg API バージョンチェック
# ==============================================================================
conf_data = configuration_data()

# AVChannelLayout 構造体の存在確認
av_channel_layout_code = '''
#include <libavutil/channel_layout.h>
int main(void) { AVChannelLayout ch_layout; return 0; }
'''
conf_data.set10('AV_CHANNEL_LAYOUT_STRUCT_AVAIL',
  cpp.compiles(av_channel_layout_code, dependencies: libavutil_dep))

# AVFrame::duration メンバーの存在確認
av_frame_duration_code = '''
#include <libavutil/frame.h>
int main(void) { AVFrame *frame; frame->duration = 0; return 0; }
'''
conf_data.set10('AV_FRAME_DURATION_AVAIL',
  cpp.compiles(av_frame_duration_code, dependencies: libavutil_dep))

# AVCodecParameters::coded_side_data メンバーの存在確認
avcodec_par_side_data_code = '''
#include <libavcodec/codec_par.h>
int main(void) { AVCodecParameters *codecpar; codecpar->coded_side_data = 0; return 0; }
'''
conf_data.set10('AVCODEC_PAR_CODED_SIDE_DATA_AVAIL',
  cpp.compiles(avcodec_par_side_data_code, dependencies: libavcodec_dep))

# 機能有効/無効フラグ
conf_data.set10('ENABLE_AVI_READER', false)
conf_data.set10('ENABLE_AVISYNTH_READER', avisynth_dep.found())
conf_data.set10('ENABLE_VAPOURSYNTH_READER', vapoursynth_dep.found())
conf_data.set10('ENABLE_AVSW_READER', true)
conf_data.set10('ENABLE_SM_READER', false)
conf_data.set10('ENABLE_LIBASS_SUBBURN', libass_dep.found())
conf_data.set10('ENABLE_VMAF', false)
conf_data.set10('ENABLE_AVCODEC_OUT_THREAD', true)
conf_data.set10('ENABLE_CPP_REGEX', true)
conf_data.set10('ENABLE_DTL', true)
conf_data.set10('ENABLE_LIBDOVI', libdovi_dep.found())
conf_data.set10('ENABLE_LIBHDR10PLUS', libhdr10plus_dep.found())
conf_data.set10('ENABLE_VULKAN', vulkan_dep.found())
conf_data.set10('ENABLE_LIBPLACEBO', libplacebo_dep.found())
conf_data.set10('ENABLE_PERF_COUNTER', false)

# AVS_INTERF_VER (AviSynth)
if avisynth_dep.found()
  conf_data.set('AVS_INTERF_VER', 12)
else
  conf_data.set('AVS_INTERF_VER', 0)
endif

# rgy_config.h 生成
configure_file(
  output: 'rgy_config.h',
  configuration: conf_data,
)

# ビルドディレクトリをインクルードパスの先頭に追加
# (ソースディレクトリの古いrgy_config.hより優先させるため)
# mesonの include_directories('.') はソースディレクトリを指すため、
# 明示的にビルドディレクトリを cpp_args で追加する
build_dir_include = '-I' + meson.current_build_dir()

# ==============================================================================
# 共通コンパイルフラグ
# ==============================================================================
common_cpp_args = [
  build_dir_include,  # ビルドディレクトリを先にインクルード
  '-DLINUX',
  '-DUNIX',
  '-D_FILE_OFFSET_BITS=64',
  '-D__USE_LARGEFILE64',
  '-D__STDC_FORMAT_MACROS',
  '-D__STDC_CONSTANT_MACROS',
  '-DLINUX64',
  '-Wno-unknown-pragmas',
  '-Wno-unused',
  '-Wno-missing-braces',
]

common_include_dirs = include_directories(
  '.',
  'NVEncSDK/Common/inc',
  'NVEncCore',
  'NVEncNVOFFRUC',
  'NVEncNVSDKNGX',
  'jitify',
  'tinyxml2',
  'cppcodec',
  'ttmath',
  'dtl',
)

# ==============================================================================
# ソースファイル
# ==============================================================================
nvencsdk_sources = files(
  'NVEncSDK/Common/src/dynlink_nvcuvid.cpp',
)

nvenccore_sources = files(
  'NVEncCore/CuvidDecode.cpp',
  'NVEncCore/FrameQueue.cpp',
  'NVEncCore/NVEncCmd.cpp',
  'NVEncCore/NVEncCore.cpp',
  'NVEncCore/NVEncDevice.cpp',
  'NVEncCore/NVEncFilter.cpp',
  'NVEncCore/NVEncFilterAfs.cpp',
  'NVEncCore/NVEncFilterColorspace.cpp',
  'NVEncCore/NVEncFilterCurves.cpp',
  'NVEncCore/NVEncFilterCustom.cpp',
  'NVEncCore/NVEncFilterDelogo.cpp',
  'NVEncCore/NVEncFilterDenoiseFFT3D.cpp',
  'NVEncCore/NVEncFilterDenoiseGauss.cpp',
  'NVEncCore/NVEncFilterLibplacebo.cpp',
  'NVEncCore/NVEncFilterNGX.cpp',
  'NVEncCore/NVEncFilterNVOFFRUC.cpp',
  'NVEncCore/NVEncFilterNvvfx.cpp',
  'NVEncCore/NVEncFilterOverlay.cpp',
  'NVEncCore/NVEncFilterPad.cpp',
  'NVEncCore/NVEncFilterParam.cpp',
  'NVEncCore/NVEncFilterRff.cpp',
  'NVEncCore/NVEncFilterSelectEvery.cpp',
  'NVEncCore/NVEncFilterSsim.cpp',
  'NVEncCore/NVEncFilterSubburn.cpp',
  'NVEncCore/NVEncParam.cpp',
  'NVEncCore/NVEncUtil.cpp',
  'NVEncCore/NVEncFilterVulkan.cpp',
  'NVEncCore/cl_func.cpp',
  'NVEncCore/convert_csp.cpp',
  'NVEncCore/cpu_info.cpp',
  'NVEncCore/gpu_info.cpp',
  'NVEncCore/gpuz_info.cpp',
  'NVEncCore/logo.cpp',
  'NVEncCore/rgy_aspect_ratio.cpp',
  'NVEncCore/rgy_avlog.cpp',
  'NVEncCore/rgy_avutil.cpp',
  'NVEncCore/rgy_bitstream.cpp',
  'NVEncCore/rgy_bitstream_aac.cpp',
  'NVEncCore/rgy_chapter.cpp',
  'NVEncCore/rgy_cmd.cpp',
  'NVEncCore/rgy_codepage.cpp',
  'NVEncCore/rgy_def.cpp',
  'NVEncCore/rgy_device.cpp',
  'NVEncCore/rgy_device_info_cache.cpp',
  'NVEncCore/rgy_device_info_wmi.cpp',
  'NVEncCore/rgy_device_usage.cpp',
  'NVEncCore/rgy_device_vulkan.cpp',
  'NVEncCore/rgy_env.cpp',
  'NVEncCore/rgy_err.cpp',
  'NVEncCore/rgy_event.cpp',
  'NVEncCore/rgy_faw.cpp',
  'NVEncCore/rgy_filesystem.cpp',
  'NVEncCore/rgy_filter.cpp',
  'NVEncCore/rgy_frame.cpp',
  'NVEncCore/rgy_frame_info.cpp',
  'NVEncCore/rgy_hdr10plus.cpp',
  'NVEncCore/rgy_ini.cpp',
  'NVEncCore/rgy_input.cpp',
  'NVEncCore/rgy_input_avcodec.cpp',
  'NVEncCore/rgy_input_avi.cpp',
  'NVEncCore/rgy_input_avs.cpp',
  'NVEncCore/rgy_input_raw.cpp',
  'NVEncCore/rgy_input_sm.cpp',
  'NVEncCore/rgy_input_vpy.cpp',
  'NVEncCore/rgy_language.cpp',
  'NVEncCore/rgy_level.cpp',
  'NVEncCore/rgy_level_av1.cpp',
  'NVEncCore/rgy_level_h264.cpp',
  'NVEncCore/rgy_level_hevc.cpp',
  'NVEncCore/rgy_libplacebo.cpp',
  'NVEncCore/rgy_log.cpp',
  'NVEncCore/rgy_memmem.cpp',
  'NVEncCore/rgy_nvrtc.cpp',
  'NVEncCore/rgy_output.cpp',
  'NVEncCore/rgy_output_avcodec.cpp',
  'NVEncCore/rgy_perf_counter.cpp',
  'NVEncCore/rgy_parallel_enc.cpp',
  'NVEncCore/rgy_perf_monitor.cpp',
  'NVEncCore/rgy_pipe.cpp',
  'NVEncCore/rgy_pipe_linux.cpp',
  'NVEncCore/rgy_prm.cpp',
  'NVEncCore/rgy_resource.cpp',
  'NVEncCore/rgy_simd.cpp',
  'NVEncCore/rgy_status.cpp',
  'NVEncCore/rgy_thread_affinity.cpp',
  'NVEncCore/rgy_timecode.cpp',
  'NVEncCore/rgy_util.cpp',
  'NVEncCore/rgy_vapoursynth_wrapper.cpp',
  'NVEncCore/rgy_vapoursynth_wrapper_v3.cpp',
  'NVEncCore/rgy_vapoursynth_wrapper_v4.cpp',
  'NVEncCore/rgy_version.cpp',
  'NVEncCore/rgy_vulkan.cpp',
  'NVEncCore/rgy_wav_parser.cpp',
)

# SIMD最適化ソース (個別フラグが必要)
# これらは static_library で個別にビルドしてリンク

tinyxml2_sources = files(
  'tinyxml2/tinyxml2.cpp',
)

nvencc_sources = files(
  'NVEncC/NVEncC.cpp',
)

# CUDAソース
cuda_sources = files(
  'NVEncCore/NVEncFilterAfsAnalyze.cu',
  'NVEncCore/NVEncFilterAfsFilter.cu',
  'NVEncCore/NVEncFilterAfsMerge.cu',
  'NVEncCore/NVEncFilterAfsSynthesize.cu',
  'NVEncCore/NVEncFilterConvolution3d.cu',
  'NVEncCore/NVEncFilterCrop.cu',
  'NVEncCore/NVEncFilterCurves.cu',
  'NVEncCore/NVEncFilterDeband.cu',
  'NVEncCore/NVEncFilterDecimate.cu',
  'NVEncCore/NVEncFilterDecomb.cu',
  'NVEncCore/NVEncFilterDelogo.cu',
  'NVEncCore/NVEncFilterDenoiseDct.cu',
  'NVEncCore/NVEncFilterDenoiseFFT3D8FP16.cu',
  'NVEncCore/NVEncFilterDenoiseFFT3D8FP32.cu',
  'NVEncCore/NVEncFilterDenoiseFFT3D16FP16.cu',
  'NVEncCore/NVEncFilterDenoiseFFT3D16FP32.cu',
  'NVEncCore/NVEncFilterDenoiseKnn.cu',
  'NVEncCore/NVEncFilterDenoiseNLMeans.cu',
  'NVEncCore/NVEncFilterDenoisePmd.cu',
  'NVEncCore/NVEncFilterEdgelevel.cu',
  'NVEncCore/NVEncFilterMpdecimate.cu',
  'NVEncCore/NVEncFilterNnedi.cu',
  'NVEncCore/NVEncFilterOverlay.cu',
  'NVEncCore/NVEncFilterResize.cu',
  'NVEncCore/NVEncFilterSmooth.cu',
  'NVEncCore/NVEncFilterSsim.cu',
  'NVEncCore/NVEncFilterSubburn.cu',
  'NVEncCore/NVEncFilterTransform.cu',
  'NVEncCore/NVEncFilterTweak.cu',
  'NVEncCore/NVEncFilterUnsharp.cu',
  'NVEncCore/NVEncFilterWarpsharp.cu',
  'NVEncCore/NVEncFilterYadif.cu',
)

# ==============================================================================
# SIMD専用ライブラリ (個別コンパイルフラグ)
# ==============================================================================
simd_sse2_lib = static_library('simd_sse2',
  'NVEncCore/convert_csp_sse2.cpp',
  cpp_args: common_cpp_args + ['-msse2'],
  include_directories: common_include_dirs,
)

simd_ssse3_lib = static_library('simd_ssse3',
  'NVEncCore/convert_csp_ssse3.cpp',
  cpp_args: common_cpp_args + ['-mssse3'],
  include_directories: common_include_dirs,
)

simd_sse41_lib = static_library('simd_sse41',
  'NVEncCore/convert_csp_sse41.cpp',
  cpp_args: common_cpp_args + ['-msse4.1'],
  include_directories: common_include_dirs,
)

simd_avx_lib = static_library('simd_avx',
  'NVEncCore/convert_csp_avx.cpp',
  cpp_args: common_cpp_args + ['-mavx', '-mpopcnt'],
  include_directories: common_include_dirs,
)

simd_avx2_lib = static_library('simd_avx2',
  [
    'NVEncCore/convert_csp_avx2.cpp',
    'NVEncCore/rgy_bitstream_avx2.cpp',
    'NVEncCore/rgy_faw_avx2.cpp',
    'NVEncCore/rgy_memmem_avx2.cpp',
  ],
  cpp_args: common_cpp_args + ['-mavx2', '-mfma', '-mpopcnt', '-mbmi', '-mbmi2'],
  include_directories: common_include_dirs,
)

simd_avx512_lib = static_library('simd_avx512',
  [
    'NVEncCore/rgy_bitstream_avx512bw.cpp',
    'NVEncCore/rgy_faw_avx512bw.cpp',
    'NVEncCore/rgy_memmem_avx512bw.cpp',
  ],
  cpp_args: common_cpp_args + ['-mavx512f', '-mavx512bw', '-mpopcnt', '-mbmi', '-mbmi2'],
  include_directories: common_include_dirs,
)

simd_libs = [
  simd_sse2_lib,
  simd_ssse3_lib,
  simd_sse41_lib,
  simd_avx_lib,
  simd_avx2_lib,
  simd_avx512_lib,
]

# ==============================================================================
# CUDA コンパイルフラグ (バージョンに応じて自動設定)
# ==============================================================================
cuda_args = gpu_code_gen + [
  '-Wno-deprecated-gpu-targets',
  '--cudart=static',
  '-std=c++17',
]

# ==============================================================================
# 依存関係まとめ
# ==============================================================================
all_deps = [
  threads_dep,
  dl_dep,
  cuda_dep,
  cuda_driver_dep,
  nppif_dep,
  nppig_dep,
  nppc_dep,
  culibos_dep,
  rt_dep,
] + ffmpeg_deps

if vapoursynth_dep.found()
  all_deps += vapoursynth_dep
endif
if avisynth_dep.found()
  all_deps += avisynth_dep
endif
if libass_dep.found()
  all_deps += libass_dep
endif
if libplacebo_dep.found()
  all_deps += libplacebo_dep
endif
if vulkan_dep.found()
  all_deps += vulkan_dep
endif
if libdovi_dep.found()
  all_deps += libdovi_dep
endif
if libhdr10plus_dep.found()
  all_deps += libhdr10plus_dep
endif

# ==============================================================================
# バイナリリソース埋め込み (objcopy + シンボル名変換)
# ==============================================================================
objcopy = find_program('objcopy')
cp = find_program('cp')

# リソース埋め込みヘルパー関数
# objcopyはパスからシンボル名を生成するため、シンボル名を揃えるためリネームして処理
subdir('meson_embed')  # embed_resource.sh を配置

# ==============================================================================
# 実行ファイル
# ==============================================================================
executable('nvencc',
  nvencsdk_sources + nvenccore_sources + tinyxml2_sources + nvencc_sources + cuda_sources +
  [resource_nnedi3, resource_colorspace, resource_perfmon],
  cpp_args: common_cpp_args,
  cuda_args: cuda_args,
  include_directories: common_include_dirs,
  dependencies: all_deps,
  link_with: simd_libs,
  install: true,
)

# ==============================================================================
# サマリ
# ==============================================================================
summary({
  'CUDA version': '@0@.@1@'.format(cuda_ver_major, cuda_ver_minor),
  'CUDA path': cuda_path,
  'GPU targets': ' '.join(gpu_code_gen),
}, section: 'CUDA')

summary({
  'VapourSynth': vapoursynth_dep.found(),
  'AviSynth': avisynth_dep.found(),
  'libass': libass_dep.found(),
  'libplacebo': libplacebo_dep.found(),
  'Vulkan': vulkan_dep.found(),
  'libdovi': libdovi_dep.found(),
  'libhdr10plus': libhdr10plus_dep.found(),
}, section: 'Optional Features')
