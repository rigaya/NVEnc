# NVEnc

NVIDIAのGPU/APUに搭載されているHWエンコーダ(NVENC)をNVENC SDKを介して呼び出す。
単体で動作するコマンドライン版とAviUtl/AviUtl2出力プラグイン版がある。

QSVEnc / NVEnc / VCEEnc / rkmppenc と共通化できる部分は共通ファイルで同じ実装を用い、そのうえに固有の実装を重ねている。

## ディレクトリ/ファイル構成
  
- `NVEncCore`
  NVEncのコア実装。Windows/Linux両対応。

  - 共通ファイル
    - 以下は QSVEnc / NVEnc / VCEEnc / rkmppenc で共通使用する。
      - afs*[.h/.cpp]
      - cl_func[.h/.cpp]
      - convert_*[.h/.cpp]
      - cpu_info[.h/.cpp]
      - gpu_info[.h/.cpp]
      - logo[.h/.cpp]
      - rgy_*[.h/.cpp]
    - 注意点
      - QSVEnc / NVEnc / VCEEnc / rkmppenc での共通性を維持する
      - 共通化が難しい場合は、部分的なら`rgy_version.h`のマクロ(`ENCODER_QSV`, `ENCODER_NVENC`, `ENCODER_VCEENC`, `ENCODER_MPP`)で切り替える。部分的ですまない場合は、固有実装のほうで実装する。

  - 二層構造により部分的な共通化を行う。以下に2つの例を挙げる。
    - パラメータ類
      - NVEncParam[.h/.cpp]
        NVEnc固有のパラメータ (固有実装)
      - rgy_prm[.h/.cpp]
        エンコーダ共有パラメータ
    - コマンドライン
      - NVEncCmd[.h/.cpp]
        NVEnc固有のパラメータ (固有実装)
      - rgy_cmd[.h/.cpp]
        エンコーダ共有パラメータ

- `NVEncC`
  `NVEncCore`の実装を使用したCLI。Windows/Linux両対応。

- `NVEnc`
  AviUtl/AviUtl2用プラグイン。`NVEncC`を呼び出してエンコードする。
  Win32ビルドはAviUtl向け(.auo)、x64ビルドはAviUtl2向け(.auo2)。

- `NVEncNVOFFRUC`
- `NVEncNVSDKNGX`
- `NVVfxLinker`
  NVEncCoreの拡張モジュール類。

- `NVEncSDK`
  NVIDIA提供のSDKを入れる場所。

- `BuildParallel`
  Windowsビルド用バッチファイル。

- `build_pkg`
  Linuxパッケージ作成用。

- `data`
  ドキュメント用のデータ。

- `docker`
  ビルド用のベースdockerfile。

- `GPUFeatures`
  `NVEncC --check-features`の結果集。適宜追加。Readme.mdから参照
  
- `resource`
  ビルド用のデータ。

- 以下は依存ライブラリ。基本触らない。
  - `ffmpeg_lgpl` (Windowsでのみ使用)
  - `PerfMonitor`
  - `cppcodec`
  - `dtl`
  - `jitify`
  - `json`
  - `tinyxml2`
  - `ttmath`

## ビルド構成

ビルド方法は `Build.ja.md` を参照。

- Windows

  MSBuildで`NVEnc.sln`をビルドする。

  |              |Debug用構成|Release用構成|
  |:---------------------|:------|:--------|
  |NVEncC(64).exe | DebugStatic | RelStatic |
  |NVEnc.[auo,auo2]  | Debug | Release |

  CUDAのバージョンごとの構成は `cudaver.props` に適宜追記していく。

- Linux

  AviUtlプラグイン、NVEncCoreの拡張モジュール類は非対象。

  `./configure` → `make` する。

## ドキュメント

- NVEncC_Options[.md/.ja.md/.cn.md]

  コマンドラインオプションについての記載。`rgy_cmd.cpp`のヘルプともに、オプションを追加したら更新すること。

- Readme[.md/.ja.md/.cn.md]
- Build[.md/.ja.md/.cn.md]
- Install[.md/.ja.md/.cn.md]