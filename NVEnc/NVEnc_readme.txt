---------------------------------------------------


    NVEnc.auo / NVEncC
     by rigaya

---------------------------------------------------

NVEnc.auo は、
NVIDIAのNVEncを使用してエンコードを行うAviutlの出力プラグインです。
NVEncによるハードウェア高速エンコードを目指します。

NVEncC は、上記のコマンドライン版です。
コマンドラインオプションについては、下記urlを参照ください。
https://github.com/rigaya/NVEnc/blob/master/NVEncC_Options.ja.md

【基本動作環境】
Windows 10 (x86/x64)
Aviutl 1.00 以降
NVEncが載ったハードウェア
  NVIDIA製 GPU GeForce Kepler世代以降 (GT/GTX 6xx 以降)
  ※GT 63x, 62x等はFermi世代のリネームであるため非対応なものがあります。
NVIDIA グラフィックドライバ 418.81以降 (x64版)
NVIDIA グラフィックドライバ 456.81以降 (x86版)

【NVEnc 使用にあたっての注意事項】
無保証です。自己責任で使用してください。
NVEncを使用したことによる、いかなる損害・トラブルについても責任を負いません。

【NVEnc 再配布(二次配布)について】
このファイル(NVEnc_readme.txt)と一緒に配布してください。念のため。
まあできればアーカイブまるごとで。

【導入方法】
※ 下記リンク先では図も使用して説明していますので、よりわかりやすいかもしれません。
   https://github.com/rigaya/NVEnc/blob/master/NVEnc_auo_readme.md#NVEnc-の-aviutl-への導入更新

1.
ダウンロードしたAviutl_NVEnc_6.xx.zipを開きます。

2.
zipファイル内のフォルダすべてをAviutlフォルダにコピーします。

3.
Aviutlを起動します。

4.
環境によっては、ウィンドウが表示され必要なモジュールのインストールが行われます。
その際、この不明な発行元のアプリがデバイスに変更を加えることを許可しますか? と出ることがありますが、
「はい」を選択してください。

5.
「その他」>「出力プラグイン情報」にNVEnc 6.xxがあるか確かめます。
ここでNVEncの表示がない場合、
- zipファイル内のフォルダすべてをコピーできていない
- 必要なモジュールのインストールに失敗した
  - この不明な発行元のアプリがデバイスに変更を加えることを許可しますか? で 「はい」を選択しなかった
  - (まれなケース) ウイルス対策ソフトにより、必要な実行ファイルが削除された
などの原因が考えられます。


【削除方法】
※ 下記リンク先では図も使用して説明していますので、よりわかりやすいかもしれません。
   https://github.com/rigaya/NVEnc/blob/master/NVEnc_auo_readme.md#NVEnc-の-aviutl-からの削除

・Aviutlのpulginsフォルダ内から下記フォルダとファイルを削除してください。
  - [フォルダ] NVEnc_stg
  - [ファイル] NVEnc.auo
  - [ファイル] NVEnc.conf (存在する場合)
  - [ファイル] NVEnc(.ini)
  - [ファイル] auo_setup.auf

【iniファイルによる拡張】
NVEnc.iniを書き換えることにより、
音声エンコーダやmuxerのコマンドラインを変更できます。
また音声エンコーダを追加することもできます。

デフォルトの設定では不十分だと思った場合は、
iniファイルの音声やmuxerのコマンドラインを調整してみてください。



コーディングが汚いとか言わないで。

【コンパイル環境】
VC++ 2019 Community

【NVIDIA CORPORATION CUDA SAMPLES EULA のライセンス規定の準拠表記】
本プログラムは、NVIDA CUDA Samplesをベースに作成されています。
すなわちサンプルコードをベースとする派生プログラムであり、サンプルコードを含みます。
“This software contains source code provided by NVIDIA Corporation.”

【dtlの使用表記】
本プログラムは、dtlライブラリを内部で使用しています。
https://github.com/cubicdaiya/dtl

【tinyxml2の使用表記】
本プログラムは、tinyxml2を内部で使用しています。
http://www.grinninglizard.com/tinyxml2/index.html


【検証環境 2014.03～】
Win7 x64
Intel Core i7 4770K + Asrock Z87 Extreme4
GeForce GTX 660
16GB 
NVIDIA グラフィックドライバ 335.23
NVIDIA グラフィックドライバ 347.09

【検証環境 2015.01～】
Win8.1 x64
Intel Core i7 5960X + ASUS X99 Deluxe
Geforce GTX 970
32GB RAM
NVIDIA グラフィックドライバ 347.25

【検証環境 2015.08～】
Win10 x64
Intel Core i7 5960X + ASUS X99 Deluxe
Geforce GTX 970
32GB RAM
NVIDIA グラフィックドライバ 353.62

【検証環境 2015.12～】
Win8.1 x64
Intel Core i3 4170 + Gigabyte Z97M-DS3H
Geforce GTX 970
8GB RAM
NVIDIA グラフィックドライバ 359.00

【検証環境 2015.12～】
Win10 x64
Intel Core i7 5960X + ASUS X99 Deluxe
Geforce GTX 960
32GB RAM
NVIDIA グラフィックドライバ 364.51


【検証環境 2016.07～】
Win10 x64
Intel Core i7 5960X + ASUS X99 Deluxe
Geforce GTX 1060
32GB 
NVIDIA グラフィックドライバ 368.81
NVIDIA グラフィックドライバ 372.70
NVIDIA グラフィックドライバ 375.95
NVIDIA グラフィックドライバ 382.33
NVIDIA グラフィックドライバ 385.41
NVIDIA グラフィックドライバ 385.69

【検証環境 2017.11～】
Win10 x64
Intel Core i9 7980XE + Asrock X299 OC Formula
Geforce GTX 1060
NVIDIA グラフィックドライバ 388.31
NVIDIA グラフィックドライバ 390.77

【検証環境 2018.11～】
Win10 x64
Intel Core i9 7980XE + Asrock X299 OC Formula
Geforce RTX 2070
Geforce GTX 1060
NVIDIA グラフィックドライバ 416.16
NVIDIA グラフィックドライバ 416.81
NVIDIA グラフィックドライバ 417.22
NVIDIA グラフィックドライバ 418.81
NVIDIA グラフィックドライバ 419.35
NVIDIA グラフィックドライバ 419.67
NVIDIA グラフィックドライバ 431.60
NVIDIA グラフィックドライバ 436.02
NVIDIA グラフィックドライバ 436.30
NVIDIA グラフィックドライバ 442.19
NVIDIA グラフィックドライバ 445.75
NVIDIA グラフィックドライバ 446.14
NVIDIA グラフィックドライバ 451.67
NVIDIA グラフィックドライバ 456.71
NVIDIA グラフィックドライバ 457.09
NVIDIA グラフィックドライバ 461.09
NVIDIA グラフィックドライバ 461.40
NVIDIA グラフィックドライバ 471.11
NVIDIA グラフィックドライバ 471.96

【検証環境 2021.9～】
Win11 x64
Intel Core i9 7980XE + Asrock X299 OC Formula
32GB RAM
Geforce RTX 2070
Geforce GTX 1060
NVIDIA グラフィックドライバ 510.06 (WSL2対応版)
NVIDIA グラフィックドライバ 511.79

【検証環境 2021.11～】
Win11 x64
Intel Core i9 12900K + MSI MAG Z690 TOMAHAWK WIFI DDR4
32GB RAM
Geforce RTX 2070
Geforce GTX 1060
NVIDIA グラフィックドライバ 512.15
NVIDIA グラフィックドライバ 522.25

【検証環境 2022.11～】
Win11 x64
Intel Core i9 12900K + MSI MAG Z690 TOMAHAWK WIFI DDR4
32GB RAM
Geforce RTX 4080
Geforce GTX 1060
NVIDIA グラフィックドライバ 527.56
NVIDIA グラフィックドライバ 528.02
NVIDIA グラフィックドライバ 531.68
NVIDIA グラフィックドライバ 536.23
NVIDIA グラフィックドライバ 545.92

【お断り】
今後の更新で設定ファイルの互換性がなくなるかもしれません。

【メモ】
2023.11.08 (7.36)
[NVEncC]
- --vpp-rffをavswにも対応。
- --vpp-afsでrffの考慮をデフォルトで有効に。

[NVEnc.auo]
- 7.35で外部音声エンコーダを使用すると、音声がmuxされない問題を修正。

2023.10.28 (7.35)
[NVEncC]
- qvbrをデフォルトに。

[NVEnc.auo]
- 外部音声エンコーダを使用すると、muxerの制限でAV1が出力できない問題を修正。
  外部muxerの使用を廃止し、内部muxerを使用するよう変更した。
- AV1時のCQPの上限を255に。
  ただ、基本的には固定品質(QVBR)の使用がおすすめ。

2023.10.18 (7.34)
[NVEncC]
- 音声フィルターの切り替えがエンコード中に発生する時に--thread-audio > 1で異常終了する問題を修正。
- --log-levelにquietを追加。
- 新しいAVChannelLayout APIに対応(Windows版)。

2023.10.06 (7.33)
[NVEncC]
- NVEnc 7.32で10bit深度のy4m読みが異常終了する問題を修正。

2023.10.05 (7.32)
[NVEncC]
- エンコードせずにy4m出力するオプションを追加。 (-c raw)
- 出力色空間を設定するオプションを追加。 (--output-csp)
- GPUの自動選択時にnvvfx使用の有無を考慮。
- --vpp-select-everyが最終フレームの処理でエラー終了する場合があったのを修正。

[NVEnc.auo]
- ログ出力に映像と音声の長さを表示するように。
- 一時ファイルの削除・リネーム失敗に関するエラーメッセージを拡充。

2023.08.28 (7.31)
- --audio-streamがavs読み込み時に反映されないのを修正。
- --video-tagの指定がない場合、HEVCでは"hvc1"をデフォルトとする。 (libavformatのデフォルトは"hev1")
- --vpp-padのエラーメッセージを拡充。
- --vpp-decimateが異常終了することがあったのを修正。
- --vpp-afs, --vpp-nnedi, --vpp-yadifのエラーメッセージを拡充。
- --vpp-colorspace lut3dの補間処理を修正。

2023.06.24 (7.30)
[NVEnc.auo]
- faw処理時に音声がブツブツ切れる場合があったのを修正。

2023.06.20 (7.29)
[NVEnc.auo]
- NVEnc 7.27のfaw処理に問題があり、異常終了してしまうケースがあったのを修正。

2023.06.07 (7.28)
[NVEncC]
- tsファイルで--audio-copyを使用すると、"AAC bitstream not in ADTS format and extradata missing" というエラーが出るのを回避。

2023.06.04 (7.27)
[NVEncC]
- ログ出力の調整機能を拡張。
- SAR指定時の出力を改善。
- ヘッダ出力の調整。

[NVEnc.auo]
- faw2aac.auo/fawcl.exeがなくても内蔵機能でfawを処理できるように。

2023.05.26 (7.26)
- NVENC SDK 12.1に対応。
- フレーム分割モードを制御する--split-encオプションを追加。

2023.05.14 (7.25)
[NVEncC]
- 音声処理をトラックごとに並列化。
- dshowのカメラ入力等に対応。
- --audio-source, --sub-sourceのコマンドラインの区切りを変更。
- --vpp-colorspaceの調整。

2023.04.25 (7.24)
[NVEncC]
- NVEnc 7.22で--sub-sourceが正常に動作しなくなっていたのを修正。

2023.04.20 (7.23)
[NVEncC]
- H.264のraw出力で、bsfを適用した場合に必要なメモリを確保できないケースがあったのを修正。

2023.04.19 (7.22)
[NVEncC]
- hwデコードとhwインタレ解除にhwリサイズを組み合わせた際の、530.xxドライバのバグを回避。
  hwデコードとhwインタレ解除にhwリサイズを組み合わせると、縮小リサイズ時に異常な出力がなされる。
  hwデコードとhwインタレ解除使用時には、hwリサイズを使用せず、CUDAによるresizeを使用するよう変更した。
- libavdeviceに対応。
- timestampが0で始まらない音声を--audio-sourceで読み込むと、映像と正しく同期が取れない問題を修正。
- --audio-source/--sub-sourceでファイルのフォーマット等を指定できるように。
- 色空間情報の記載のないy4mファイルの色がおかしくなるのを回避。
- NVENCで並列5ストリームエンコードできるようになったのでメッセージを変更。
- --vpp-resizeにbicubicを追加。

2023.03.13 (7.21)
[NVEncC]
- 音声・字幕のtimestampに負の値が入ることがあったのを回避。
- Linux環境で、PGS字幕をコピーしようとするとpgs_frame_mergeが見つからないというエラーが発生するのを回避。

[NVEnc.auo]
- 出力する動画の長さが短い場合の警告を追加。
- NVENCが利用可能かチェックする際、--log-level debugを付与。

2023.03.04 (7.20)
[NVEncC]
- --vpp-overlayを最後のフィルタに変更。
- 複数のGPUがある場合の集計方法を合計から最大に変更。

[NVEnc.auo]
- オブジェクトエクスプローラからドラッグドロップしたファイルがある場合の出力に対応。

2023.02.21 (7.19)
[NVEncC]
- avs読み込み時に--sub-sourceを指定するとエラー終了してしまうのを回避。

[NVEnc.auo]
- bref-modeの設定をすると、うまく設定が保存されないのを修正。

2023.02.13 (7.18)
[NVEncC]
- フレーム時刻をタイムコードファイルから設定するオプションを追加。(--tcfile-in)
- 時間精度を指定するオプションを追加。(--timebase)
- 色調を指定したカーブに従って変更するオプションを追加。(--vpp-curves)
- --audio-profileが変更できなかった時に警告を表示するように。
- Bフレーム数が3以下ではbref-modeを強制的に無効にするよう修正。

[NVEnc.auo]
- bref-modeの設定欄が効かなくなっていた問題を修正。

2023.02.09 (7.17)
[NVEncC]
- nvvfx系のフィルタで128で割り切れない解像度にも対応。
- 7.15からPGS字幕のコピーがうまく動作しなくなっていた問題を修正。

2023.02.07 (7.16)
[NVEncC]
- 7.15から --vpp-resize spline16, spline36, spline64を使用すると、意図しない線やノイズが入る問題を修正。

2023.02.05 (7.15)
[NVEncC]
- NVIDIA MAXINE VideoEffects SDKによるフィルタを追加。 (Windows x64版のみ)
  - --vpp-nvvfx-denoise
  - --vpp-nvvfx-artifact-reduction
  - --vpp-resize nvvfx-superres
- ffmpegのライブラリを更新 (Windows版)
  ffmpeg     5.0    -> 5.1
  libpng     1.3.8  -> 1.3.9
  expat      2.4.4  -> 2.5.0
  libsndfile 1.0.31 -> 1.2.0
  libxml2    2.9.12 -> 2.10.3
  libbluray  1.3.0  -> 1.3.4
  dav1d      0.9.2  -> 1.0.0
- --sub-sourceでPGS字幕を読み込むと正常にmuxできない問題を回避。
- --check-hwや--check-featuresのログ出力を--log-levelで制御できるように。

2023.01.30 (7.14)
[NVEncC]
- defaultより遅いpresetの場合、可能ならbref-modeを使用するように。
- lowlatency向けにmuxの挙動を調整。
- 動画ファイルに添付ファイルをつけるオプションを追加。 (--attachement-source)
- --perf-monitorでビットレート情報等が出力されないのを修正。
- 音声エンコードスレッド (--thread-audio 1) が動作しなくなっていた問題を修正。

2023.01.22 (7.13)
[NVEncC]
- AV1のmaster-displayの取り扱いを改善。
- maxcllあるいはmastering displayの片方がないときに、AV1エンコードで適切でないデータが発行されていた問題の修正。
- 言語による--audio-copyの指定が適切に動作していなかった問題を修正。
- dolby-vision-profileを使用した場合でも、指定したchromaloc等が優先されるよう動作を変更。

2023.01.21 (7.12)
[NVEncC]
- --vpp-overlayのlumakey使用時にもalphaを指定できるように。

2023.01.20 (7.11)
[NVEncC]
- 画像を焼き込むフィルタ(--vpp-overlay)を複数回使用可能なよう拡張。

2023.01.19 (7.10)
[NVEncC]
- 画像を焼き込むフィルタに輝度値に応じて透明度を決定するオプションを追加。

2023.01.16 (7.09)
[NVEncC]
- 画像を焼き込むフィルタを追加。 ( --vpp-overlay )
- ssim/psnr/vmafでhwデコードがサポートされない場合のエラーメッセージを改善。
- AV1 hwデコードが正常に動作しない問題を修正。
  cuvid: cuvidParseVideoData errorが出てしまっていた。

2022.12.29 (7.08)
[NVEncC]
- vpp-decimateに複数のフレームをdropさせるオプションを追加。(--vpp-decimate drop)
- -c rawを廃止。

2022.12.28 (7.07)
[NVEncC]
- エンコードせずにy4m出力するオプションを追加。 (-c raw)
- 時刻でエンコード終了時刻を指定するオプションを追加。(--seekto)
- 入力ファイルが10bitのとき、--output-depth 10を指定しないとHEVCのlossless出力が動作しなかった問題を修正。

2022.12.10 (7.06)
[NVEncC]
- 入力が10-16bit深度の時、HEVC 10bitのlossless出力ができなくなっていたのを修正。

[NVEnc.auo]
- エラーメッセージの文字化けを修正。

2022.11.21 (7.05)
- AV1エンコードでmp4/mkv等にmuxしながら出力する際、シークができない場合があったのを改善。

2022.11.12 (7.04)
- --vpp-subburnと--dhdr10-info copyを併用すると、dynamic HDR10のmetadataが適切にコピーされない問題を修正。
- Hierarchial P/Bフレームの階層数を指定するオプションを追加。(--temporal-layers)
- エンコードするフレーム数を指定するオプション(--frames)を追加。
- --fpsで入力フレームレートを強制できるように。

2022.11.06 (7.03)
[NVEncC]
- --vpp-subburnにfoced flag付きの字幕のみ焼きこむオプションを追加。
- H.264のHierarchial P/Bフレームを有効にするオプションを追加。(--hierarchial-p, --hierarchial-b)

2022.10.19 (7.02)
[NVEncC]
- AV1エンコード時のいくつかの不具合を修正。
  デバッグ情報のご提供にご協力いただいたSpeed様に感謝いたします。

2022.10.16 (7.01)
[NVEncC]
- AV1エンコード選択時の異常終了を修正。

2022.10.14 (7.00)
[NVEncC]
- NVIDIA Video Codec SDK 12.0に対応。
- AV1エンコードに「仮」対応。
  全くテストできていないので注意！
- --dhdr10-infoの実装方法を変え、Lookahead使用時も対応するように。

2022.09.19 (6.12)
[NVEnc.auo]
- Aviutl中国語対応をされているNsyw様に提供いただいた中国語対応を追加。
  翻訳の対応、ありがとうございました！

2022.09.18 (6.11)
[NVEncC]
- GPU使用率等の情報収集を改善。

[NVEnc.auo]
- 設定画面上にツールチップを追加。
- 英語表示に対応。

2022.08.25 (6.10)
[NVEncC]
- --audio-streamの処理で、途中で音声のチャンネルが変化した場合にも対応。
- GPUでの p210 → yuv444 の変換の不具合を修正。

2022.08.21 (6.09)
[NVEncC]
- --vpp-deinterlace使用時にインタレ解除フラグが適切に設定されない問題を修正。

[NVEnc.auo]
- AVX2使用時にFAWの1/2モードが正常に処理できなかったのを修正。

2022.08.14 (6.08)
[NVEncC]
- Linuxで標準入力から読み込ませたときに、意図せず処理が中断してしまう問題を修正。
  なお、これに伴いLinuxではコンソールからの'q'あるいは'Q'でのプログラム終了はサポートしないよう変更した。(Ctrl+Cで代用のこと)

[NVEnc.auo]
- qaac/fdkaacのコマンドラインに --gapless-mode 2 を追加。

2022.08.10 (6.07)
[NVEncC]
- downmix時に音量が小さくなってしまうのを回避。

[NVEnc.auo]
- NVENCが利用できない場合でも「NVENCが利用可能か確認 [ダブルクリック].bat」だと利用できることになっていたのを修正。

2022.07.29 (6.06)
[NVEncC]
- hdr2sdrでcolormatrix,transfer,colorprimの変換を指定した際には、指定に基づいた変換を行うように。
- 音声デコーダの初期化時にsample fmtが設定されない場合にエラーが発生する問題を回避。
- AVPacket関連の非推奨関数の使用を削減。

[NVEnc.auo]
- デフォルトの音声ビットレートを変更。
- プリセットの音声ビットレートを変更。
- exe_filesから実行ファイルを検出できない場合、plugins\exe_filesを検索するように。
- エンコード終了ログの文字化け対策。

2022.06.16 (6.05)
[NVEncC]
- yuv422読み込み時にcropを使用すると横に黒い線画出てしまう問題を修正。
- arm64ビルドに仮対応。

2022.06.13 (6.04)
[NVEncC]
- Ubuntu 22.04向けパッケージを追加。

[NVEnc.auo]
- 黒窓プラグイン使用時に設定画面の描画を調整。

2022.06.04 (6.03)
[NVEncC]
- YUV420でvpp-afs使用時に、二重化するフレームで縞模様が発生してしまう問題を修正。
- ldr_nits, source_peakに関するエラーチェックを追加。

[NVEnc.auo]
- ログウィンドウでScrollToCaret()を使用しないように。

2022.05.17 (6.02)
[NVEncC]
- アスペクト比を維持しつつ、指定の解像度にリサイズするオプションを追加。(--output-res <w>x<h>,preserve_aspect_ratio=<string>)
- コンソールからの'q'あるいは'Q'でプログラムを終了するように。
- 一部のHEVCの入力ファイルで、--avhw使用時にデコードが正常に行われず、破綻した映像となってしまうのを修正。

[NVEnc.auo]
- 音声の一時出力先が反映されなくなっていたのを修正。
- なるべくremuxerで処理するよう変更。
- ffmpeg (AAC)で -aac_coder twoloop を使用するように。

2022.04.29 (6.01)
[NVEncC]
- HEVC 10bitの入力ファイルをavhwで読み込む際にエラーが発生することがあったのを回避。
- vpp-afsとvpp-convolution3d併用時にtimestamp計算のエラーが出る問題を修正。
- vpp-colorspace の lut3d_interp に pyramid, prism を追加。
- timestampのチェックを追加。

[NVEnc.auo]
- 簡易インストーラを直接実行した場合に、エラーメッセージを表示するように変更。

2022.04.16 (6.00)
[NVEncC]
- 3次元ノイズ除去フィルタを追加。(--vpp-convolution3d)
- 音声の開始時刻が0でなく、かつ映像と音声のtimebaseが異なる場合の音ズレを修正。

[NVEnc.auo]
- .NET Framework 4.8に移行。
- パッケージのフォルダ構成を変更。
- 簡易インストーラによるインストールを廃止。
- パスが指定されていない場合、exe_files内の実行ファイルを検索して使用するように。
- ログに使用した実行ファイルのパスを出力するように。
- 相対パスでのパスの保存をデフォルトに。
- 拡張編集使用時の映像と音声の長さが異なる場合の動作の改善。
  拡張編集で音声を読み込ませたあと、異なるサンプリングレートの音声をAviutl本体に読み込ませると、
  音声のサンプル数はそのままに、サンプリングレートだけが変わってしまい、音声の時間が変わってしまうことがある。
  拡張編集使用時に、映像と音声の長さにずれがある場合、これを疑ってサンプリングレートのずれの可能性がある場合は
  音声のサンプル数を修正する。
- エンコードするフレーム数が0の場合のエラーメッセージを追加。
- ログの保存に失敗すると、例外が発生していたのを修正。
- ログの保存に失敗した場合にその原因を表示するように。
- muxエラーの一部原因を詳しく表示するように。
  mp4出力で対応していない音声エンコーダを選択した場合のエラーメッセージを追加。
- エラーメッセージ
  「NVEncCが予期せず途中終了しました。NVEncCに不正なパラメータ（オプション）が渡された可能性があります。」
    の一部原因を詳しく表示するように。
  1. ディスク容量不足でエンコードに失敗した場合のエラーメッセージを追加。
  2. 環境依存文字を含むファイル名- フォルダ名で出力しようとした場合のエラーメッセージを追加。
  3. Windowsに保護されたフォルダ等、アクセス権のないフォルダに出力しようとした場合のエラーメッセージを追加。

2022.03.05 (5.46)
[NVEncC]
・avcuvid/avsw: avcodec: failed to load dlls.というエラーで異常終了する問題を修正。

2022.03.05 (5.45)
[NVEncC]
・ffmpeg関連のdllを更新。(Windows版)
  ffmpeg     4.x    -> 5.0
  expat      2.2.5  -> 2.4.4
  fribidi    1.0.1  -> 1.0.11
  libogg     1.3.4  -> 1.3.5
  libvorbis  1.3.6  -> 1.3.7
  libsndfile 1.0.28 -> 1.0.31
  libxml2    2.9.10 -> 2.9.12
  libbluray  1.1.2  -> 1.3.0
  dav1d      0.6.0  -> 0.9.2
・SetThreadInformationの使用できない環境でのエラー回避。
[NVEnc.auo]
・5.44で、出力開始時にフリーズしてしまう場合があったのを修正。

2022.02.22 (5.44)
[NVEncC]
・--vpp-delogoでauto_nr/auto_fade使用時に、"Invalid input frame ID -1 sent to encoder." というエラーが発生していたのを修正。
・--qvbr <float>オプションを追加。(--vbr 0 --vbr-quality <float> に同じ)
[NVEnc.auo]
・Aviutlの子プロセスが開いているファイルについても出力ファイルで上書きしないようチェックを追加。
・設定が行われていない場合に、前回出力した設定を読み込むように。

2022.02.14 (5.43)
[NVEncC]
・vpp-colorspaceにlut3dフィルタを追加。(--vpp-colorspace lut3d)
・Dolby Visionのrpuを読み込み反映させるオプションを追加。(--dolby-vision-rpu)
・Dolby Visionのプロファイルを指定するオプションを追加。(--dolby-vision-profile)
[NVEnc.auo]
・NVEnc.auoでの出力の際、Aviutlが開いているファイルに上書きしないように。
・パラメータが設定されていない場合、デフォルトの設定でエンコードするように。

2021.12.11 (5.42)
・--dar指定時に負の解像度を使用すると、sar扱いで計算され意図しない解像度となるのを修正。
・スレッドの優先度とPower Throttolingモードを指定するオプションを追加。(--thread-priority, --thread-throttling)
・avhw使用時に入力ファイルが壊れていると、フリーズする場合があったのを回避。

2021.10.14 (5.41)
[NVEncC]
・12bit深度を10bit深度に変換するときなどに、画面の左上に緑色の線が入ることがあったのを修正。
・bitstreamのヘッダ探索をAVX2/AVX512を用いて高速化。

[NVEnc.auo]
・ログ出力モードをデフォルト以外に変更すると異常終了していたのを修正。

2021.09.30 (5.40)
・想定動作環境にWindows11を追加。
・Windows11の検出を追加。
・Windows11のWSL2での実行に対応。
・スレッドアフィニティを指定するオプションを追加。(--thread-affinity)
・ログの各行に時刻を表示するオプションを追加(デバッグ用)。(--log-opt addtime)

2021.09.23 (5.39)
・5.38で--check-hw, --check-featuresが動作しなかったのを修正。

2021.09.21 (5.38)
・--vpp-padの左右が反対になっていたのを修正。
・--vpp-smoothのfp16版の高速化。
  RTX2070で10%高速。
・音声トラックにbitstream filterを適用するオプションを追加。(--audio-bsf)
・マルチGPU環境で同時session数の上限に達していないのにエラーが出てしまうのを回避。
・5.19から--caption2assが使用できなかったのを修正。

2021.08.10 (5.37)
・vpp-subburnで使用できるフォントのタイプを更新。
・audio-delayが効いていなかったのを修正。

2021.07.31 (5.36)
・NVENC SDK 11.1に更新。
・色差のQPオフセットを指定可能に。(--chroma-qp-offset)
・Linux環境でCUDA11.4からコンパイルできなくなっていた問題を修正。
・NVEnc.auoの設定画面で--lookahead 16が指定できなかった問題を修正。

2021.07.24 (5.35)
・Linux環境でvpp-colorspaceを使用すると、NVEncFilterColorspaceFunc.hが存在しないというエラーが発生するのを修正。 
・字幕や音声の順序がおかしくなる場合があったのを修正。
・5.31からyuv444→p010の変換で色成分がずれてしまっていたのを修正。
・libassのログレベルを変更。

2021.06.15 (5.34)
・AvisynthNeo環境などで生じるエラー終了を修正。
・入力ファイルと出力ファイルが同じである場合にエラー終了するように。

2021.05.23 (5.33)
・raw出力やログ出力の際にカレントディレクトリに出力しようとすると異常終了が発生する問題を修正。

2021.05.18 (5.32)
・5.31で選択したリサイザ名がログに適切に表示されないのを修正。
・--blurayオプションを使用した際にシークしづらかったのを修正。
・--vpp-resizeの引数として、--helpにcubic_xxxxxが表示されていたが、
  動作しないので表示されないよう変更した。

2021.05.08 (5.31)
・avsw/avhwでのファイル読み込み時にファイル解析サイズの上限を設定するオプションを追加。(--input-probesize)
・--input-analyzeを小数点で指定可能なよう拡張。
・読み込んだパケットの情報を出力するオプションを追加。( --log-packets )
・data streamに限り、タイムスタンプの得られていないパケットをそのまま転送するようにする。
・オプションを記載したファイルを読み込む機能を追加。( --option-file )
・冗長な処理化処理の省略。
・動画情報を取得できない場合のエラーメッセージを追加。
・コピーするtrackをコーデック名で選択可能に。
・字幕の変換が必要な場合の処理が有効化されていなかったのを修正。

2021.04.11 (5.30)
・--audio-source/--sub-sourceを複数指定した場合の挙動を改善。
・--avsync forcecfrの連続フレーム挿入の制限を1024まで緩和。
・--sub-metadata, --audio-metadataを指定した場合にも入力ファイルからのmetadataをコピーするように。
・字幕のmetadataが二重に出力されてしまっていた問題を修正。
・--slicesを指定した場合にはログにその情報を出力するように。
・--vpp-solorspaceが4で割り切れない横解像度の時の問題を修正。

2021.02.14 (5.29)
・AvisynthのUnicode対応に伴い、プロセスの文字コードがUTF-8になっているのを
  OSのデフォルトの文字コード(基本的にShiftJIS)に戻すオプションを追加。(--process-codepage os)
  これにより、従来のShiftJISのavsファイルも読み込めるようになる。
  NVEncC実行ファイルと同じ場所に実行ファイルのコピーを作るので、
  Program Files等書き込みに管理者権限の必要な場所に置かないよう注意。

2021.02.12 (5.28)
・5.26から--vpp-edgelevelを使用すると画面が暗くなってしまうのを修正。
・AvisynthのUnicode対応を追加。

2021.02.11 (5.27)
・--vpp-subburnで埋め込みフォントを使用可能なように。
・--vpp-subburnでフォントの存在するフォルダを指定可能なように。
・Windows 10のlong path supportの追加。
・--audio-source / --sub-source でmetadataを指定可能なよう拡張。
・--vpp-warpsharpのhelpを修正。

2021.01.31 (5.26)
・細線化フィルタの追加。(--vpp-warpsharp)
・--vpp-subburnで短いassファイルが正常に処理できなかったのを改善。

2021.01.10 (5.25)
・--vpp-subburnで一部のassファイルを焼きこもうとすると異常終了していたのを修正。
・--vpp-smoothでPascal GPUで実行しようとすると正常に動作せず、緑色の絵になってしまっていたのを修正。
・--videoformatからautoを削除。正常に動作していなかった。
・5.24で --vpp-colorspace hdr2sdr=mobius/reinhardが異常終了するのを修正。

2020.12.30 (5.24)
・x86版をCUDA11ベースに更新し、VS2019に移行。
  NVIDIA グラフィックドライバ 456.81 以降が必要。
・timecodeの出力を追加。(--timecode)
・--check-featureでLevelの値を表示するように。
・--vpp-colorspace hdr2sdr=bt2390の実装見直し。
  処理時の規格化の有無について誤解していた。
・--vpp-colorspace hdr2sdrにdesaturation curveの実装。
  desat_base, desat_strength, desat_expの追加。
・libvmaf 2.0.0+ に更新。従来(v1.3.15)と比べ高速化。
  それでもCPU処理なのでまだエンコードには追いつかない模様。
・YUV444でnppを使用したリサイズを行うとエラーで落ちてしまうのを修正。

2020.12.20 (5.23)
・--tier highで720p等でmax-bitrateが0になってしまうのを修正。
・bit深度を下げるときの丸め方法を変更。
・vpp-colorspaceのhdr2sdrにbt2390によるtone mappingを追加。
・vpp-colorspaceのAmpereへの対応。
・言語による音声や字幕の選択に対応。

2020.12.01 (5.22)
・bit深度を下げるときの丸め方法を変更。
・chapterを読み込む際に、msの値を正しく取得できない場合があったのを修正。

2020.11.19 (5.21)
・重複フレームを削除したVFR動画を作成することで実効的なエンコード速度を向上させるフィルタを追加。(--vpp-mpdecimate )
・VMAFスコアを計算するオプションを追加。(Win x64版のみ、libvmaf v1.3.15を使用した実装)
  CPUでの処理なので非常に重く、VMAF計算のほうで律速してしまうので注意。あまり実用的ではないかも。

2020.11.15 (5.20)
・HLG用のAlternative Transfer Characteristicsを設定する場合は、コンテナ側にはVUI情報をもたせないようにする。
  コンテナ側にはatcの情報をもたせられないので、かちあってしまう。
・--vpp-tweakのswapuvの修正。

2020.11.01 (5.19)
・HLG用のAlternative Transfer Characteristicsを指定するオプションを追加。( --atc-sei )
・--sub-copy指定時に常にすべての字幕がコピーされるようになっていた問題を修正。
・HEVCエンコードで常にrepeat-headersが有効になっていたのを修正。
・avsw/avhw読み込み以外でもvpp-subburnに対応。
・timestampの順序が反転した場合への対応。
・エラーメッセージの改善。
・AmpereのCUDAコア数の判定を追加。

2020.10.18 (5.18)
・NVENC SDK 11のサポートを追加。
  AV1, HEVC 12bitのhw decodeが可能…かもしれない。(RTX30xxを未所持のため、未テスト)

2020.10.15 (5.17)
・456.38以降のドライバの問題により、--vpp-knnが動作しなくなっていたので、この問題を回避できるようコードを改変。

2020.10.12 (5.16)
・--vpp-subburnと--sub-copyを同時に指定可能に。
・--vpp-tweakにU,V成分を反転させるオプションを追加。
・--check-hw, --check-featureが常に戻り値0を返していたのを修正。

2020.09.12 (5.15)
・raw読み込み時に色空間を指定するオプションを追加。( --input-csp )
  raw読み込みのyuv420/422/444の8-16bitの読み込みに対応。
・p210→yv12変換を追加。
・--maxcll/--masterdisplayが指定されている場合は、IDRフレームごとにヘッダ(SPS/PPS/VPS)を出力するとともに、
  そのうしろに--maxcll/--masterdisplayを付加するようにした。
・--maxcll/--masterdisplayをそれぞれ異なるnalユニットに出力するように。
・proresがデコードできないのを修正。
・vpp-nnedi、vpp-padのコード見直し(簡略化)。

2020.08.04 (5.14)
・ロードするAvisynth.dllを指定するオプションを追加。(--avsdll)

2020.07.29 (5.13)
・ffmpeg関連のdllを更新。
  これにより、ts/m2tsへのPGSのmuxを可能とする。

2020.07.26 (5.12)
・--audio-stream stereoが動作しないのを修正。
・mkv出力時にdefault-durationが設定されるように。
・bref-modeが使用可能かについて、each/only middleを区別して判定するように。

2020.07.15 (5.11)
・--multipassのオプションの誤字を修正。
  2pass-quater -> 2pass-quarter
・--presetで7段階のプリセットを指定可能に。(API v10.0以降のみ対応)

2020.07.14 (5.10)
[NVEncC]
・NVENC SDK 10.0への対応を追加。SDK 10.0の機能を使用するには、ドライバ445.87以降が必要。
・マルチパス解析モードを詳細に指定するオプションを追加。(--multipass)
・NVENC SDK 9.0/9.1 との互換性を維持。
  必要なドライババージョンを満たさない場合、旧APIバージョンでの互換動作を行う。
  下記の互換動作とする。
     | SDK API 9.1  | SDK API 10.0                 |
     | --vbrhq      | --vbr --multipass 2pass-full |
     | --cbrhq      | --cbr --multipass 2pass-full |

2020.07.05 (5.09)
[NVEncC]
・raw出力でSAR比を指定したときに発生するメモリリークを修正。
・--vpp-decimate の blockx, blocky オプションで4,8にも対応。

2020.07.01 (5.08)
[NVEncC]
・5.07でnppc64_10.dllが動作に必須になってしまっていたのを修正。

2020.06.30 (5.07)
[NVEncC]
・キーフレームごとに VPS,SPS,PPS を出力するオプションを追加。(--repeat-headers)

[NVEnc.auo]
・簡易インストーラ更新。
  VC runtimeのダウンロード先のリンク切れを修正。

2020.06.21 (5.06v2)
[NVEnc.auo]
・設定画面に追加コマンド指定欄を追加。

2020.06.16 (5.06)
[NVEncC]
・5.01からvpy読み込みがシングルスレッド動作になっていたのを
  マルチスレッド動作に戻した。

2020.06.14 (5.05)
[NVEncC]
・5.04でHEVCのhwデコードができなくなっていたのを修正。
・--audio-sourceでもdelayを指定できるように。

2020.06.11 (5.04)
[NVEncC]
・一部のHEVCファイルで、正常にデコードできないことがあるのに対し、可能であればswデコーダでデコードできるようにした。
・--dhdr10-infoとlookaheadの相性が悪く、nvEncEncodePicture内でクラッシュしてしまうため、--dhdr10-info使用時にはlookaheadを無効にするようにした。
・入力のエラーを捕捉してもエラーコードが0を返してしまう場合があったのを修正。
・avs読み込みで、より詳細なAvisynthのバージョンを取得するように。
・GPU自動選択の際、ssim/psnr計算に必要なhwデコーダのチェックもするように。

[NVEnc.auo]
・NVEnc.auoの設定画面でも、--output-resに負の値を指定できるように。

2020.05.31 (5.03)
[NVEncC]
・遅延を伴う一部の--audio-filterで音声の最後がエンコードされなくなってしまう問題を修正。
・lowlatencyが使用できないのを修正。
・--video-tagを指定すると異常終了してしまうのを修正。 

2020.05.23 (5.02)
[NVEncC]
・出力するmetadata制御を行うオプション群を追加。
  --metadata
  --video-metadata
  --audio-metadata
  --sub-metadata
・streamのdispositionを指定するオプションを追加。 (--audio-disposition, --sub-disposition)
・--audio-source/--sub-sourceでうまくファイル名を取得できないことがあるのを修正。
・--helpに記載のなかった下記オプションを追記。
  --video-tag
  --keyfile
  --vpp-smooth
・--vpp-delogoでデバッグ用のメッセージが標準出力にでるようになってしまっていたのを修正。
・読み込み時に発生したエラーを補足するように。
・オプションリストを表示するオプションを追加。 (--option-list)
・動画の最初のtimestampをデバッグログに表示するように。

2020.05.07 (5.01)
[NVEncC]
・Linuxに対応。
・デコーダのエラー検出時に異常終了するようになっていなかったのを修正。
・--check-featuresにhwデコードの情報を追加。
・yuv444→yv12/p010変換のマルチスレッド時のメモリアクセスエラーを修正。
・重複フレームを削除するフィルタを追加。 ( --vpp-decimate )
・attachmentをコピーするオプションを追加。 ( --attachment-copy )

[NVEnc.auo]
・外部エンコーダ使用時に、音声エンコードを「同時」に行うと異常終了するのを修正。

2020.04.18 (5.00)
[NVEncC]
・動画を回転/転置するフィルタを追加。( --vpp-rotate, --vpp-transform)
・エンコード遅延を抑制するオプションを追加。( --lowlatency )
  エンコードのパフォーマンス(スループット)が落ちるので、あまり意味はない。
・hdr10plusのメタ情報をそのままコピーするオプションを追加。(--dhdr10-info copy)
  現状制限事項が多いので注意。
  - avsw/avhwのみ。
  - avhwはraw streamなどtimestampが取得できない場合、正常に動作しない。
・音声デコーダやエンコーダへのオプション指定が誤っていた場合に、
  エラーで異常終了するのではなく、警告を出して継続するよう変更。
・--chapterがavsw/avhw利用時にしか効かなかったのを修正。

[NVEnc.auo]
・NVEnc.auoで内部エンコーダを使用するモードを追加。
  こちらの動作をデフォルトにし、外部エンコーダを使うほうはオプションに。

2020.03.25 (4.69)
[NVEncC]
・ドライバ445.75で、不必要なエラーメッセージが表示される問題に対策。
・HEVCのYUV444出力時のプロファイルを修正。
・ffmpeg関連のdllを更新。
  libopusのビルドを修正。
  dav1d 0.5.2 -> 0.6.0
  bzip2 1.0.6 -> 1.0.8

[NVEnc.auo]
・ffmpeg_audenc.exeを更新。
・L-SMASHを下記のブランチのコードをいただいて更新。
  https://github.com/nekopanda/l-smash/tree/fast

2020.03.07 (4.68)
[NVEncC]
・avsw/avhw読み込み時の入力オプションを指定するオプションを追加。( --input-option )
・trueHDなどの一部音声がうまくmuxできないのを改善。
・4.66から、vpp-yadifの出力が異常となってしまう問題への対策。

[NVEnc.auo]
・4.67での変更が特にAviutlで自動フィールドシフトを使用した場合に正常に動作しなかったのを修正。

[NVEnc.auo]
・NVEnc.auoから出力するときに、Aviutlのウィンドウを最小化したり元に戻すなどするとフレームが化ける問題を修正。

2020.03.01 (4.67)
[NVEncC]
・NVEnc.auoの修正に対応する変更を実施。

[NVEnc.auo]
・NVEnc.auoから出力するときに、Aviutlのウィンドウを最小化したり元に戻すなどするとフレームが化ける問題を修正。

2020.02.29 (4.66)
[NVEncC]
・新たなノイズ除去フィルタを追加する。(--vpp-smooth)
・HEVCでも自動での最大ビットレート上限の設定にrefを参照するように。
・vpp-subburnに動画ファイルのタイムスタンプに対する補正を行うかを指定するパラメータを追加する。(vid_ts_offset)
・vpp-subburnで、動画のtimestampが0始まりでなかった場合の時刻調整の誤りを修正。
・vpp-colorspaceでcolorrangeのみの変換を可能に。

[NVEnc.auo]
・簡易インストーラの安定動作を目指した改修。
  必要な実行ファイルをダウンロードしてインストールする形式から、
  あらかじめ同梱した実行ファイルを展開してインストールする方式に変更する。
・デフォルトの音声エンコーダをffmpegによるAACに変更。
・NVEnc.auoの設定画面のタブによる遷移順を調整。

2020.02.20 (4.65)
[NVEncC]
・コマンドラインの指定ミスの際のエラーメッセージを改善。
・caption2assが正常に動作しないケースがあったのを修正。
・必要ドライバのバージョンを更新。

[NVEnc.auo]
・ビットレート上限の解放。

2020.02.11 (4.64)
[NVEncC]
・lookaheadを使用した場合に不要なエラーメッセージが表示されていたのを修正。

2020.02.10 (4.63)
[NVEncC]
・mux時の動作の安定性を向上し、シーク時に不安定になる症状を改善。
・起動時の初期化動作の安定化。起動時に異常終了することがあるのを改善。
・--interlace autoが使えない状態だったのを修正。
・--chromalocの設定結果がおかしかったのを修正。
・エンコードを中断した際に、まれにフリーズしてしまうのを修正。
・デバッグ用ログ出力の拡充。
・ログに文字化けしている箇所があったのを修正。

2020.02.01 (4.62)
[NVEncC]
・colormatrix等の情報を入力ファイルからコピーする機能を追加。
  --colormtarix auto
  --colorprim auto
  --transfer auto
  --chromaloc auto
  --colorrange auto
  また、vpp-colorspaceでも使用可能。
・avsw/avhw読み込み時に、フレームごとにインタレかどうかを判定してインタレ解除を行うオプションを追加。(--interlace auto)
・インタレ対応のyuv422→yuv420変換がないため、yuv444を経由するように。
・HEVCエンコ時に、high tierの存在しないlevelが設定された場合、main tierに修正するように。
・ssim/psnr計算の安定性向上。
・4.60から--vpp-subburnのscaleオプションが動作しなくなっていたのを修正。
・vpp-subburnを使用した場合の頑健性向上。
・ログに常に出力ファイル名を表示するように。
・VUI情報、mastering dsiplay, maxcllの情報をログに表示するように。

[NVEnc.auo]
・NVEncCとの連携のための実装を変更。
  たまに緑のフレームが入ったりする(?)という問題に対処できているとよいが…。

2020.01.16 (4.61)
[NVEncC]
・コピーすべきmaxcll/maxfallの情報がない時に、--master-display copyや--max-cll copyを使用してmkv出力すると
  おかしな値がmaxcll/maxfallに設定されてしまうのを修正。
・--ssim/--psnr使用時に解放されないメモリが残っていたのを修正。

[NVEnc.auo]
・vpp-yadifでbob化を指定できるように。

2020.01.13 (4.60)
[共通]
・ssimの計算を行うオプションを追加。(--ssim)
・psnrの計算を行うオプションを追加。(--psnr)
・プロセスのGPU使用率情報を使用するように。 (Win10のみ)

[NVEncC]
・vpp-subburnで指定したtrackがない場合、エラー終了するのではなく、警告メッセージを出して処理を継続するように。
・HDR関連のmeta情報を入力ファイルからコピーできるように。
  (--master-display copy, --max-cll copy)
・ffmpeg関連のdllを更新。
  libogg-1.3.3 -> 1.3.4
  twolame-0.3.13 -> 0.4.0
  wavpack-5.1.0 -> 5.2.0
  libxml2-2.9.9 -> 2.9.10
  dav1d-0.5.2 !new!

[NVEnc.auo]
・vpp-yadifを追加。

2019.12.24 (4.59)
[NVEncC]
・vpp-subburnで字幕の色成分が欠けて焼きこまれるようになっていたのを修正。

[NVEnc.auo]
・簡易インストーラを更新。

2019.12.16 (4.58)
[NVEncC]
・HEVCエンコ時のlevelがログに正しく表示されないのを修正。
・音声処理でのメモリリークを解消。
・--pref-monitorにエンコード速度やビットレートが表示されなかったのを修正。

2019.12.05 (4.57)
[NVEnc.auo]
・8bitでインタレ保持・インタレ解除を行う際に色がおかしくなるのを修正。

[NVEncC]
・vpp-afsのYUV420処理時のlevel=0の出力がおかしいのを修正。
・誤字修正: arib-srd-b67 →　arib-std-b67
・--vpp-colorspaceにHLG→SDRのサポートを追加。
  --vpp-colorspace transfer=arib-srd-b67:bt709,hdr2sdr=hable
・trueHD in mkvなどで、音声デコードに失敗する場合があるのを修正。
・4.56で字幕のコピーが動かなくなっていたのを修正。
・vpp-colorspace hdr2sdrのパラメータ"w"を廃止。source_peak / ldr_nits で求めるべきだった。
・vpp-colorspaceのsource_peak周りの扱いを見直し。

2019.11.25 (4.56)
[NVEnc.auo]
・NVEnc.auoの出力をmp4/mkv出力に変更し、特に自動フィールドシフト使用時のmux工程数を削減する。
  また、NVEncCのmuxerを使用することで、コンテナを作成したライブラリとしてNVEncCを記載するようにする。

[NVEncC]
・YUV444で出力する際、vpp-afsの判定結果がおかしくなっていたのを修正。
・mkv入りのVC-1をカットした動画のエンコードに失敗する問題を修正。
・output-resに負の値を指定できるように。
・HEVC + weightpが不安定とする警告はPascal/Volta世代のGPUに限定する。
・HEVCのmultirefを自動的に制限するように。
・音声に遅延を加えるオプションを追加。(--audio-delay)

2019.11.05 (4.55)
[NVEncC]
・4.53での下記の変更の修正方法が間違っていたのを修正。
  - master-display, max-cllなどを指定してmp4/mkv等にmuxした際に、
    値が化けてしまうのを修正。

2019.11.02 (4.54)
・なかったことに…。

2019.10.30 (4.53)
[NVEncC]
・avsからの音声処理に対応。
・master-display, max-cllなどを指定してmp4/mkv等にmuxした際に、
  値が化けてしまうのを修正。
・横解像度が16で割り切れない場合の安定化。

[NVEnc.auo]
・横解像度が16で割り切れない場合の動作を改善。
・縦解像度が4で割り切れない場合に異常終了する問題を修正。

2019.10.07 (4.52)
[NVEncC]
・Multiple refsの指定方法を変更。(--multiref-l0/--multiref-l1)
・--multiref-**をhelpに追加。

2019.10.07 (4.51)
[NVEncC]
・NVENC SDKを9.1に更新。
  Multiple refsの指定機能を追加。(--ref)
・可能なら進捗表示に沿うフレーム数も表示するように。

2019.09.21 (4.50)
[NVEncC]
・yuv444(16bit)->nv12/yv12でオーバーフローが発生することがあったのを修正。
・vpp-resize spline16の係数設定に誤りがあったのを修正。

[NVEnc.auo]
・自動フィールドシフトのトラックバーとテキストボックスの連動が解除されていたのを修正。

2019.09.18 (4.49)
[NVEncC]
・vpp-subburnに透過性、輝度、コントラストの調整を行うオプションを追加。

2019.09.17 (4.48)
[NVEncC]
・vpp-colorspace reinhard/mobiusの変換式を修正。高輝度の場合の値が正しく算出されていなかった。
・vpp-colorspace hdr2sdrについて、RGBを同期して値を調整するように。
・vpp-afsのフレームバッファの実装を簡略化。無駄に複雑になっていた。
・コンソールの幅に合わせた進捗表示に。

2019.09.01 (4.47)
[NVEnc.auo]
・NVEnc.auo - NVEncC間のフレーム転送を効率化して高速化。
  2～10%程度高速化。
・エンコードを中断できるように。
・ログウィンドウに何%エンコードできたかと、予想ファイルサイズを表示。
  自動フィールドシフト使用時を除く。

[NVEncC]
・高負荷時の安定化。
・字幕ファイルを読み込むオプションを追加。 (--sub-source)
・環境によっては、GPUを正しく検出できない問題を改善。
・4.44以降、--audio-sourceが正常に動作しない問題を修正。
・4.44以降、--output-formatが使用できなくなっていたのを修正。

2019.08.27 (4.46)
[NVEncC]
・高負荷時にデッドロックが発生しうる問題を修正。
・出力ファイルサイズを推定するように。
・GPUチェック時(--check-hw)のログ出力を追加。

2019.08.19 (4.45)
[NVEncC]
・--vpp-subburnで緑の線が出てしまう問題を修正。
・音声エンコードの安定性を向上。
・wma proのデコードに失敗する問題を修正。

2019.08.10 (4.44)
[NVEncC]
・--audio-sourceを改修し、より柔軟に音声ファイルを取り扱えるように。
・--dhrd10-infoが正常に動作しない問題を修正。
・--vpp-subburnで字幕を拡大すると、緑の線が入ってしまうことがあるのを修正。

2019.07.14 (4.43)
[NVEncC]
・--audio-copyなどがないと、trackを指定した字幕の焼きこみができないのを修正。
・複数の字幕ファイルを焼きこめるよう拡張。
・データストリームをコピーするオプションを追加する。(--data-copy)
・CPUの動作周波数が適切に取得できないことがあったのを修正。
・字幕を拡大/縮小して焼きこめるように。
・pulldownの判定を改善。
・ffmpegと関連dllを追加/更新。
  - [追加] libxml2 2.9.9
  - [追加] libbluray 1.1.2
  - [追加] aribb24 rev85
  - [更新] libpng 1.6.34 -> 1.6.37
  - [更新] libvorbis 1.3.5 -> 1.3.6
  - [更新] opus 1.2.1 -> 1.3.1
  - [更新] soxr 0.1.2 -> 0.1.3

2019.06.02 (4.42)
[NVEnc.auo]
・losslessを設定画面に追加。

[NVEncC]
・--vpp-subburnで入力ファイルによっては、字幕が正常に焼きこまれないのを修正。
・non-reference P-framesを自動的に挿入するオプションを追加。(--nonrefp)
・lookahead未使用時のGPUメモリ使用量を削減。
・4.40の修正が不十分な場合があったのを修正。

2019.05.25 (4.41)
[NVEncC]
・--chapterでmatroska形式に対応する。
・動的にレート制御モードを変更するオプションを追加。(--dynamic-rc)
・dynamic HDR10+ metadataを付与する機能を追加。(--dhdr10-info)
・字幕焼きこみ時に--cropを反映して焼きこむように。
・nppライブラリによるリサイズが正常に動作しなかったのを修正。
・nv12->nv12で並列化した場合に映像がずれてしまう問題を修正。

2019.05.21 (4.40)
[NVEncC]
・avhw以外のリーダーを使用した際に、横解像度が64で割り切れない場合に
  並列で色フォーマットの変換をすると左端にノイズが生じる問題を修正。

2019.05.20 (4.39)
[NVEncC]
・字幕焼きこみのフィルタを追加。(--vpp-subburn)
・--sub-copyで字幕の選択番号がひとつずれてしまっているのを修正。
・--vpp-colorspaceのhdr2sdrにhable, mobius, reinhardを追加。
  hdr2sdrの指定方法が変更になるので注意。(hable, mobius, reinhardの中から選択)
・--vpp-colorspaceにsmpte240m用の行列が欠けているのを修正。
・--avhw利用時、cuvidでcropを使用しないようにして、CUDAでcropするようにする。
  cuvidでcropを行うと、縦のcropが4で割り切れない場合に指定通りにcropされないなど、よくわからない現象に悩まされるため。

2019.05.07 (4.38)
[共通]
・vpp-nnediで埋め込みの重みデータを使用する場合に一時バッファを使用しないようにする。
  cufilter.aufなどの32bit環境では最悪メモリ確保に失敗する恐れがあった。

[NVEnc.auo]
・AVX2を搭載しないCPUでもAVX2を使用した関数が使われるようになってしまっていた問題を修正。

[NVEncC]
・--vpp-colorspaceを実行する際に、nvrtc-builtins64_101.dllも必要だったのだがこれが含まれておらず、正常に実行できなかったのを修正。

2019.05.05 (4.37)
[共通]
・リサイズアルゴリズムを追加。(lanczos2,lanczos3,lanczos4,spline16,spline64)
・色空間変換の並列化。

[NVEncC]
・x64版をVC++2019に移行。
・YUV444のhwデコードに対応。(Turing以降)
・インタレ解除フィルタyadifの追加。(--vpp-yadif)
・RGBの対応範囲を拡大。
・色空間変換を行うフィルタを追加。(--vpp-colorspace)
  zimgの実装をベースにmatrix/colorprim/transferの変換とhdr2sdrの変換を行う。
  この際、jitifyを使用したCUDAの実行時コンパイルを行うため、nvrtc_101.dllが必要。この関係でx64版のみのサポートとなる。

2019.04.02 (4.36)
[共通]
・エンコードできない場合のエラーメッセージを改善。

[NVEncC]
・--audio-copyでTrueHDなどが正しくコピーされないのを修正。

2019.03.24 (4.35)
[共通]
・vpp-nnedi利用時に異常終了する可能性があったのを修正。

[NVEncC]
・audio-filter利用時にフィルターによっては異常終了する可能性があったのを修正。

2019.03.20 (4.34)
[共通]
・CUDA 10.1で必要なdll名が間違っていたのを修正。

[NVEncC]
・helpにnnediについての記述を追記。

2019.03.17 (4.33)
[共通]
・3つめのインタレ解除フィルタを追加。(--vpp-nnedi)
・Turing世代のGPUが2倍のコア数として表示されてしまっていたのを修正。
・[x64版のみ] CUDA 10.1に移行。

2019.03.04 (4.32)
[NVEncC]
・--trimを使用すると音声とずれてしまう場合があったのを修正。
・映像のcodec tagを指定するオプションを追加。(--video-tag)

2019.02.12 (4.31)
[共通]
・NVENC SDK 9.0に更新。NVIDIA グラフィックドライバ 418.81以降が必要。
・HEVCエンコ時のB ref modeの設定を追加。(--bref-mode)

[NVEnc.auo]
・設定画面に「品質(--preset)」「Bフレーム参照モード(--bref-mode)」を追加。

[NVEncC]
・--presetをreadmeに追加。


2019.02.07 (4.30)
[共通]
・TuringでHEVCのBフレームが使用可能になったので、HEVCでもデフォルトのBフレーム数を3にする。
・インタレ保持エンコで出力したファイルのシーク時の挙動が不安定だったのを修正。

[NVEnc.auo]
・NVEnc.auoからHEVCのBフレームが使用できなかった問題を修正する。

[NVEncC]
・音声エンコード時のtimestampを取り扱いを改良、VFR時の音ズレを抑制。

2018.12.17 (4.29)
[NVEncC]
・--master-displayが正常に動作しない場合があったのを修正。

2018.12.11 (4.28)
[NVEnc.auo]
・Aviutlからのフレーム取得時間がエンコードを中断した場合に正常に計算されないのを修正。

2018.12.10 (4.27)
[NVEncC]
・--chapterを指定した場合、暗黙のうちに--chapter-copyを有効にする。
・計算時にオーバーフローが発生してしまう場合があったのを修正。
  mkvのchapterをmuxする際などに正常にchapterをmuxできなかった。
  
[NVEnc.auo]
・自動フィールドシフト使用時、widthが32で割り切れない場合に範囲外アクセスの例外で落ちる可能性があったのを修正。

2018.11.24 (4.26)
[NVEncC]
・色空間変換とGPU転送の効率化。
・--audio-fileが正常に動作しないことがあったのを修正。

2018.11.19 (4.25)
[NVEncC]
・読み込みにudp等のプロトコルを使用する場合に、正常に処理できなくなっていたのを修正。
  4.22以降のバグ。

2018.11.18 (4.24)
[NVEncC]
・--vpp-select-everyを使用してもログ表示に反映されないのを改善。
・muxなしで出力すると、caption2assを使用しないときでもメッセージが出ていたのを修正。
・古いAvisynthを使うと正常に動作しなくなっていたのを修正。

[NVEnc.auo]
・簡易インストーラを更新。
  - Apple dllがダウンロードできなくなっていたので対応。
  - システムのプロキシ設定を自動的に使用するように。

2018.11.08 (4.23)
[NVEncC]
・指定stepフレームごとに1フレームを選択してフレームを間引くオプションを追加。(--vpp-select-every)
・VC-1デコードの際にエラー終了することがあったのを改善。
・perf-monitorにPCIe周りの情報を追加。
・マルチGPU環境でのGPU選択改善。
  - インタレ保持エンコを考慮したGPU選択をするように。
  - Bフレームの指定があった場合には、それをサポートするGPUを選択するように。

2018.11.03 (4.22)
[共通]
・yuv420のlossless出力に対応。

[NVEncC]
・Caption.dllによる字幕抽出処理を実装。(--caption2ass)
・--check-featuresでGPU名が正しく表示されていなかったのを修正。
・--check-featuresにバージョン情報も出力するように。
・--check-environmentの出力先をstderrからstdoutに。

2018.10.27 (4.21)
[NVEnc.auo]
・NVEnc.auoのリサイズアルゴリズム選択を修正。
・NVEnc.iniにffmpegによる音声エンコードと、デュアルモノの分離処理を追加。
・NVEnc.auoの設定画面からwav出力できなかったのを修正。
  指定された動画エンコーダは存在しません。[ ]とエラーが出てしまっていた。
・faw2aac処理後も音声エンコ後バッチ処理を行うように。
  なお、音声エンコ前バッチ処理は実施されない。

2018.10.13 (4.20)
[NVEnc.auo]
・--vbr-qualityが小数で指定できるように。

[NVEncC]
・VC-1 hwデコードを有効に。

2018.10.08 (4.19)
[共通]
・GPUバイナリを含めないようにして、配布バイナリを軽量化。

[NVEnc.auo]
・品質指定のプロファイルを追加。
・プロファイル「HEVC ビットレート指定 高画質」から重み付きPフレームを外した。
・一時フォルダの相対パス指定に対応した。
・多重音声を扱う際、muxer.exeがエラー終了してしまうのを修正。

[NVEncC]
・--aud/--pic-struct/--slicesを追加。
・--check-hwの出力を改善。
・NPPライブラリのリサイザアルゴリズムのうち、最近NPP_INTERPOLATION_ERRORを返すようになったものをドキュメントから削除。
  cubic_bspline, cubic_catmull, cubic_b05c03が削除。cubicは問題ないので、そちらを使用してください。

2018.09.27 (4.18)
・--key-on-chapterをmuxしない場合にも使用可能に。
・ファイルによるキーフレームの指定に対応。(--keyfile)

2018.09.26 (4.17)
[NVEncC]
・チャプターのあるフレームに、キーフレームを挿入する機能を追加。(--key-on-chapter)
  ただし、--trimとの併用は不可。

2018.09.18 (4.16)
[NVEncC]
・一部のmp4/mkv等のコンテナに入った10bit HEVCの入力ファイルが正常にデコードできない問題について、
  avhwでの問題を回避。

2018.09.12 (4.15)
[NVEncC]
・一部のmp4/mkv等のコンテナに入った10bit HEVCの入力ファイルが正常にデコードできない問題について、
  avswでは問題を解消。(avhwでは未解決)
・Max MB Per secについてはチェックをしていないので、--check-featuresの表示項目から外した。

2018.08.28 (4.14)
[NVEncC]
・vpp-delogoで自動フェードを有効にするとエラー終了する問題を修正。

2018.08.19 (4.13)
[NVEnc.auo]
・NVEnc 4.12で動作しなくなっていた(NVEncCが異常終了する)のを修正。

[NVEncC]
・vpp-delogoの自動フェード・自動NRを大幅に高速化。
・高ビット深度出力時のvpp-padの動作を修正。

2018.08.10 (4.12)
[NVEncC]
・vpp-delogoの自動フェード・自動NR機能を追加。
  ・合わせてvpp-delogoのオプションの指定方法を変更。
  ・一部機能を除いた簡易版です。
  ・まだ遅いです。

2018.08.06 (4.11)
[NVEncC]
・一部の動画ファイルで、音ズレの発生するケースに対処。
・BlurayオプションのGOP長の制限を緩和。

2018.07.27 (4.10)
[NVEncC]
・進捗状況でtrimを考慮するように。
・OpenCLがまともに動作しない環境でのクラッシュを回避。
  まれによくあることらしい。

2018.07.10 (4.09)
[NVEncC]
・音声エンコーダにオプションを引き渡せるように。
  例: --audio-codec aac:aac_coder=twoloop
・音声エンコード時にプロファイルを指定できるように。(--audio-profile)
・高ビットレートでのメモリ使用量を少し削減。
・パディングを付加するフィルタを追加。(--vpp-pad)
・可変フレームレートなどの場合に、中途半端なフレームレートとなってしまうのを改善。
・音声のほうが先に始まる場合の同期を改善。
  
2018.07.05 (4.08)
[NVEncC]
・--audio-fileが正常に動作していなかったのを修正。
・--input-analyzeの効果を改善。
・SAR指定時の安定性を改善。

2018.06.05 (4.07)
[共通]
・--darが4.04以降正常に動作しなかったのを修正。
・4.02以降、主に海外でコマンドラインの浮動小数点がうまく読めない場合があったのを修正。

2018.06.02 (4.06)
[NVEncC]
・--audio-codec / --audio-bitrate / --audio-samplerate / --audio-filter等のコマンドを
  トラックを指定せずに指定した場合、入力ファイルのすべての音声トラックを処理対象に。
・--max-cll / --masterdisplay 使用時の互換性を改善。

2018.05.29 (4.05)
[NVEnc.auo]
・4.04で設定画面を表示しようとするとクラッシュしたのを修正。

[NVEncC]
・--max-cll / --masterdisplay 使用時の出力を改善。
・--sarと--max-cll / --masterdisplay を同時に使用すると、正常に動作しなかったのを修正。

2018.05.28 (4.04)
[共通]
・意図したsar比がセットされないのを回避。

[NVEncC]
・chroma locationのフラグを指定するオプションを追加。
・インタレ保持エンコードでmuxしながら出力する際、フィールド単位でmuxせず、フレーム単位でmuxするように。

2018.05.20 (4.03)
[NVEncC]
・ffmpegと関連ライブラリのdllを更新。

2018.05.14 (4.02)
[NVEncC]
・プロセスのロケールを明示的にシステムのロケールに合わせるように。

2018.05.03 (4.01)
[NVEncC]
・muxしながら出力する際、--max-cllや--masterdisplayを使用するとフリーズしてしまうのを修正。
・ロゴの自動選択が正常に動作しないのを修正。

2018.05.02 (4.00)
[共通]
・NVENC SDKを8.1に更新。

[NVEnc.auo]
・エンコーダを内蔵せず、NVEncCにパイプ渡しするように。
  Aviutl本体プロセスのメモリ使用量を削減。
  またwin7における連続バッチ出力時のリソース開放漏れ問題が発生しなくなる。

[NVEncC]
・Bフレームの参照モードを設定するオプションを追加。(--bref-mode)
  現行のGPUではサポートされない模様。
・HEVCのtierを指定するオプションを追加。(--tier)
・HEVCエンコ時にVUI情報の自動設定が行われないのを修正。
・mux時にHDR関連のmetadataの反映を改善。
・mux時の映像/音声の同期を改善。

2018.03.13 (3.33)
[NVEncC]
3.32でtrimを使うとエラー終了してしまうのを修正。

2018.03.11 (3.32)
[共通]
・やはりHEVC + --weightpは不安定な場合があるようなので、警告メッセージを出すようにした。

[NVEncC]
・--avsync vfrの安定性を改善。
・動画のrotation情報があればコピーするように。
・"%"を含む出力ファイル名で--logを指定すると落ちるのを修正。
・--input-analysisを大きくしすぎると、エラー終了する場合があったのを修正。

2018.03.04 (3.31)
[NVEncC]
・"%"を含む出力ファイル名で出力しようとすると落ちるのを修正。
・avswのデコーダのスレッド数を16までに制限。
・--avsync vfrを追加。avhw/avswモード時にソースのtimestampのままで出力するモード。
  --trimとは併用できない。
・tsファイルなどでtrim使用時に、ずれてしまう場合があったのを修正。

2018.02.20 (3.30)
[共通]
・--max-cll, --master-displayを使用しないときの出力がおかしかったのを修正。
・出力バッファサイズをドライバに決めさせるようにして安定性を改善。

[NVEncC]
・動画のDEFAULT stream情報とlanguage情報もあればコピーするように。

2018.02.19 (3.29)
[共通]
・バッファサイズを動的に変更するようにして、lookaheadが多い時の安定性を改善。
・HEVCエンコ時にsliceを明示的に1にするようにして、安定性を改善。
・HEVCエンコ時にweightpを再度有効に。390.77では問題なさそう?

[NVEncC]
・--audio-copy, --sub-copy等で、streamの情報を適切にコピーするように。

2018.02.14 (3.28)
[NVEncC]
・HDR関連metadataを設定するオプションを追加。(--max-cll, --master-display)

2018.02.03 (3.27v2)
[NVEnc.auo]
・設定画面でOKボタンが押せない場合があったのを修正(たぶん)。
  120dpiベースでGUIが作成されてしまっていたのを96dpiベースに戻した。

2018.01.07 (3.27)
[共通]
・色調補正フィルタを追加。(vpp-tweak)
・--vpp-deinterlace bobを使用時に、ビットレートが半分として表示されてしまうのを修正。
・同時エンコードが2までに制限されていることのエラーメッセージを強化。

[NVEnc]
・リソース開放がうまく行われていなかったのを修正。
・デバッグログ出力を設定画面から有効にできるように。

[NVEncC]
・不適切なデバイスIDを指定したときに、自動的にデバイスを変更するように。
・vpp-delogoが正常に動かなくなっていたのを修正。
・avsからのyuv420/yuv422/yuv444の高ビット深度読み込みに対応。
  ただし、いわゆるhigh bitdepth hackには対応しない。

2017.12.14 (3.26)
[共通]
・VUI情報が正しく反映されないことがあるのを修正。
・ログ表示のミスを修正。

[NVEncC]
・--audio-copy/--audio-codec/--sub-copy指定時に、入力ファイルに音声/字幕トラックがない場合でもエラー終了しないように。

2017.11.26 (3.25)
[共通]
・不安定なため、HEVCエンコード時にはweightpを無効化。

2017.11.14 (3.24)
[共通]
・フィルタ使用時のGPUメモリ使用量を削減。

[NVEncC]
・pulldownを検出するように。
  完全な2:3pulldownの場合には29.97fpsでなく、23.976fpsとして検出する。
  avsync, vpp-rff, vpp-afsを使用していない場合のみ有効。
・nvmlのエラー情報を詳細に表示するように。
・yv12(high)->p010[AVX2]のバグを修正。
・GPUが見つからないと表示される箇所があったのを修正。

2017.09.26 (3.23)
[共通]
・vpp-afsで、最終フレームがdropとなると異常終了することがあったのを修正。

2017.09.23 (3.22)
[共通]
・--cuda-scheduleのデフォルトをautoに戻す。
  syncだと速度がかなり落ちてしまう場合があった。

2017.09.19 (3.21)
[共通]
・Unsharpフィルタを使用した際に、色が赤みがかったり、青みがかったりすることがあったのを修正。

2017.09.18 (3.20)
[共通]
・Unsharpフィルタを追加。(--vpp-unsharp)
・エッジレベル調整フィルタを追加。(--vpp-edgelevel)
・特に指定のない場合、deviceの選択を実行時に自動で決定するように。
  エンコーダ/デコーダ/GPUの使用率/GPUの世代/GPUのコア数等を考慮して、自動的に決定する。
  --deviceで明示的に指定した場合は、従来通り指定に従う。

[NVEncC]
・avhwリーダー使用時に、cropとresizeを使用すると、cropが二重にかかるようになっていたのを修正。
・--transferの引数をx264等で使用されているものに合わせる。
  smpte-st-2048 → smpte2048
  smpte-st-428  → smpte428

2017.09.11 (3.19)
[NVEnc.auo]
・HEVCエンコ時にフレームタイプが設定できるように。

2017.09.10 (3.18)
[共通]
・自動フィールドシフトを追加。(vpp-afs)

  Aviutl版とほぼ同様だが、GPUでの実装の都合上全く同じ結果にはならないのと、一部機能制限がある。

  パラメータ ... 基本的にはAviutl版のパラメータをそのまま使用する。
    top=<int>           (上)
    bottom=<int>        (下)
    left=<int>          (左)
    right=<int>         (右)
    method_switch=<int> (切替点)
    coeff_shift=<int>   (判定比)
    thre_shift=<int>    (縞(シフト))
    thre_deint=<int>    (縞(解除))
    thre_motion_y=<int> (Y動き)
    thre_motion_c=<int> (C動き)
    analyze=<int>       (解除Lv)
    shift=<bool>        (フィールドシフト)
    drop=<bool>         (ドロップ)
    smooth=<bool>       (スムージング)
    24fps=<bool>        (24fps化)
    tune=<bool>         (調整モード)
    rff=<bool>          (rffフラグをチェックして反映)
    log=<bool>          (デバッグ用のログ出力)

  下記には未対応
    解除Lv5
    シーンチェンジ検出(解除Lv1)
    編集モード
    ログ保存
    ログ再生
    YUY2補間
    シフト・解除なし

・各種フィルタがインタレ保持でも適切に処理できるように。
  vpp-resize, vpp-knn, vpp-pmd, vpp-gauss, vpp-unsharpが対象。
  vpp-delogo, vpp-debandはもとから対応済み。
  
[NVEnc.auo]
・簡易インストーラを更新。

[NVEncC]
・rffを適切に反映し、フィールドに展開する機能を追加。(vpp-rff)
  avcuvid読み込み時専用。またtrimとは併用できない。
・高ビット深度のyuv420からp010に変換するときに異常終了することがあるのを修正。

2017.08.01 (3.17)
[NVEncC]
・rawでの読み込みが正常に動作していなかったのを修正。

2017.07.26 (3.16)
[NVEncC]
・x64版が動かなかったのを修正。

2017.07.25 (3.15)
[共通]
・GPUのドライババージョンをチェックするように。
・CUDAの存在しない環境で、クラッシュしてしまうのを修正。
・ヘルプのささいな修正。

[NVEncC]
・高ビット深度のyuv422/yuv444をy4mから読み込むと色成分がおかしくなるのを修正。
・高ビット深度でdelogoが正常に動作しなかったのを修正。
・3.08からy4mからパイプで読み込めなくなっていたのを修正。

2017.06.30 (3.14)
[共通]
・CPU使用率を低減。特に、HWデコーダ使用時のCPU使用率を大きく削減。
・CUDAのスケジューリングモードを指定するオプションを追加。(--cuda-schedule <string>)
  主に、GPUのタスク終了を待機する際のCPUの挙動を決める。デフォルトはsync。
  - auto  ... CUDAのドライバにモード決定を委ねる。
  - spin  ... 常にCPUを稼働させ、GPUタスクの終了を監視する。
              復帰のレイテンシが最小となり、最も高速だが、1コア分のCPU使用率を常に使用する。
  - yield ... 基本的にはspinと同じだが、他のスレッドがあればそちらに譲る。
  - sync  ... GPUタスクの終了まで、スレッドをスリープさせる。
              わずかに性能が落ちるかわりに、特にHWデコード使用時に、CPU使用率を大きく削減する。
・実行時のCUDAのバージョンをログに表示するように。

[NVEncC]
・helpの表示がおかしかった箇所を修正。
・エンコード終了時に進捗表示のゴミが残らないように。
・NVMLを使用してGPU使用率などの情報を取得するように。x64版のみ。

2017.06.24 (3.13)
[共通]
・バンディング低減フィルタを追加。
・パフォーマンス分析ができなくなっていたのを修正。

[NVEncC]
・--avcuvidを使用すると、--cropが正しく反映されない場合があったのを修正。

2017.06.19 (3.12)
[NVEnc.auo]
・NVEnc.auoで10bit深度、yuv444のエンコードができなくなっていたのを修正。

2017.06.18 (3.11)
[共通]
・より柔軟にSAR比の指定に対応。
・NVEncのrevision情報を表示するように。

[NVEncC]
・実行時に取得したデコーダの機能をもとに、デコードの可否を判定するように。
・avswにrgb読み込みを追加。
・avsw/y4m/vpyからのyuv422読み込みに対応(インタレは除く)。
・--audio-streamを使用した際に、条件によっては、再生できないファイルができてしまうのを修正。

2017.06.12 (3.10)
[NVEncC]
・y4m渡しが3.09でも壊れていたので修正。
・正常終了した場合でも、エラーコード上はエラーを返していることがあるのを修正。

2017.06.11 (3.09)
[NVEncC]
・高ビット深度をy4m渡しすると、絵が破綻するのを修正。

2017.06.10 (3.08)
[共通]
・NVENC SDKを8.0に更新。
・重み付きPフレームを有効にするオプションを追加。(--weightp)
・Windowsのビルドバージョンをログに表示するように。
・32で割りきれない高さの動画をインタレ保持エンコードできない場合があったのを修正。
・GPU-Zの"Video Engine Load"を集計できるように。

[NVEnc.auo]
・簡易インストーラを更新。
・QPの上限・下限・初期値の設定を追加。
・VBR品質目標の設定を追加。

[NVEncC]
・10bit HEVCのHWデコードに対応。
・ffmpegと関連ライブラリのdllを更新。
・HWデコード時の安定性を向上。
・--vbr-qualityを小数点指定に対応。
・aviファイルリーダーを追加。
・LTR Trust modeは今後非推奨となる予定のため、--enable-ltrオプションを削除。
・vpyリーダー使用時に、エンコードを中断しようとするとフリーズしてしまう問題を修正。
・字幕のコピーが正常に行われない場合があったのを修正。

2017.03.08 (3.07)
[共通]
・ログにいくつかのパラメータを追加。

[NVEnc.auo]
・プリセットを現状に合わせて調整。
・簡易インストーラを更新。

[NVEncC]
・H.264ではLTRが非対応であるとのことをhelpに明記。
・いくつかのオプションを追加。(--direct, --adapt-transform)
・NVENCのpresetを反映するオプションを追加。(--preset)

2017.02.05 (3.06)
※同梱のdll類も更新してください!
[NVEncC]
・HEVC/VP8/VP9のハードウェアデコードを追加。
・メモリリークを修正。
・使用しているCUDAのバージョン情報を表示。

2017.01.11 (3.05)
[共通]
・3.04のミスを修正。

2017.01.10 (3.04)
[共通]
・HEVCでもロスレス出力可能なように。

2017.01.09 (3.03)
[共通]
・NVENC SDKを7.1に更新。
・NVENC SDKを7.1に合わせてレート制御モードを整理。
  - CQP
  - VBR
  - VBRHQ (旧VBR2)
  - CBR
  - CBRHQ
・低解像度で--level 3などを指定してもエラー終了してしまう問題を解消。

[NVEncC]
・mkvを入力としたHEVCエンコードで、エンコード開始直後にフリーズしてしまうのを解消。

[NVEnc.auo]
・自動フィールドシフト使用時に、最後のフレームがdropであると1フレーム多く出力されてしまう問題を修正。
  これが原因で、timecodeとフレーム数が合わずmuxに失敗する問題があった。

2016.11.21 (3.02)
[共通]
・2.xxのnvcuvid利用時及び3.00以降のすべてのケースでインタレ保持エンコードが正常にできなくなっていたのを修正。

[NVEnc.auo]
・簡易インストーラを更新。

2016.10.11 (3.01)
[共通]
・CUDA 8.0正式版にコンパイラを変更。

[NVEncC]
・muxする際、プログレッシブエンコードなのにBFFのフラグが立っていた問題を修正。
・--audio-sourceが期待どおりに動作しない場合があったのを修正。

2016.09.18 (3.00)
[共通]
・さまざまなリサイズアルゴリズムを追加。(--vpp-resize)
・Knn(K nearest neighbor)ノイズ除去を追加。(--vpp-knn)
・正則化PMD法によるノイズ除去を追加。(--vpp-pmd)

[NVEncC]
・音声処理のエラー耐性を向上。
・avcuvid読み込み以外でもリサイズを可能に。
・avcuvid読み込み以外でもtrimを可能に。
・左cropが動作しないのを解消。
・透過性ロゴフィルタを追加。(--vpp-delogo)
・ガウシアンフィルタを追加(x64のみ)。(--vpp-gauss)

2016.08.07 (2.11)
[NVEncC]
・2.08から、vpp-deinterlaceが使用できなくなっていたのを修正。

2016.07.28 (2.10)
[共通]
・PascalのHEVC YUV444 10bitエンコードに対応。

[NVEncC]
・出力ビット深度を設定するオプションを追加。(--output-depth <int>)
・avsw/y4m/vpyリーダーからのyuv420高ビット深度の入力に対応。
・avsw/y4m/vpyリーダーからのyuv444(8bit,高ビット深度)の入力に対応。

[NVEnc.auo]
・afs使用時にHEVC 10bitエンコードができなかった問題を修正。
・進捗表示がされない問題を修正。

2016.07.22 (2.09)
[共通]
・--helpが壊れている問題を修正。
・fpsが正しく取得できていない場合のエラーを追加。
・HEVC 4:4:4に対応。
  profileでmain444を指定してください。

[NVEncC]
・10bitエンコードを追加 (HEVCのみ)。(--output-depth <int>)

[NVEnc.auo]
・10bitエンコードを追加 (HEVCのみ)。
  プロファイルで「main10」を指定してください。
  YC48から10bitへの直接変換を行います。

2016.07.19 (2.08)
[共通]
・NVENC SDK 7.0に対応
  NVIDIA グラフィックドライバ 368.69以降が必要
・SDK 7.0で追加された機能のオプションを追加。
  --lookahead <int> (0-32)
  --strict-gop (NVEncCのみ)
  --no-i-adapt (NVEncCのみ)
  --no-b-adapt (NVEncCのみ)
  --enable-ltr (NVEncCのみ)
  --aq-temporal
  --aq-strength <int> (0-15)
  --vbr-quality <int> (0-51)
・--avswを追加。
・複数の動画トラックがある際に、これを選択するオプションを追加。(--video-track, --video-streamid)
  --video-trackは最も解像度の高いトラックから1,2,3...、あるいは低い解像度から -1,-2,-3,...と選択する。
  --video-streamidは動画ストリームののstream idで指定する。
・入力ファイルのフォーマットを指定するオプションを追加。(--input-format)

2016.06.18 (2.07v2)
・簡易インストーラを更新。

2016.05.29 (2.07)
・Bluray出力が行えなくなっていたのを修正。
・ログが正常に表示されないものがあったのを修正。
・コマンドラインのオプション名が存在しない場合のエラーメッセージを改善。
・NVEnc.auoで中断できないのを修正。

2016.04.29 (2.06)
・avcuvid使用時にデコーダのモードを指定できるように。
  --avcuvid native (デフォルト)
  --avcuvid cuda
  なにも指定しないときはnative。

2016.04.20 (2.05v2)
・簡易インストーラを更新。

2016.04.15 (2.05)
[NVEncC]
・--audio-copyの際のエラー回避を追加。

2016.04.03 (2.04)
[NVEncC]
・qp-min, qp-max, qp-initなどが指定できなかった問題を修正。

2016.04.01 (2.03)
[NVEncC]
・入力ファイルにudpなどのプロトコルが指定されていたら、自動的にavcuvidリーダーを使用するように。
・音声関連ログの体裁改善とフィルタ情報の追加。
・音声フィルタリングを可能に。 (--audio-filter)
  ffmpegのdllを含めて更新してください。(ソースは QSVEnc_2.42_lgpl_dll_src.7z)
  音量変更の場合は、"--audio-filter volume=0.2"など。
  書式はffmpegの-afと同じ。いわゆるsimple filter (1 stream in 1 stream out) なら使用可能なはず。

2016.03.27 (2.02)
[NVEncC]
・エンコード速度が低い時のCPU使用率を大幅に低減。
・mux時に書き込む情報がQSVEncになっていたのを修正。
・HEVCのmuxが正常に行えないことがあるのを修正。
・avsync forcecfr + trimは併用できないので、エラー終了するように。
・dll更新。(ソースは QSVEnc_2.42_lgpl_dll_src.7z)

2016.03.24 (2.01)
[共通]
・MB per secのチェックを行わないようにした。
[NVEnc]
・簡易インストーラを更新。
[NVEncC]
・QSVEncからmux関連機能を追加する。
  --avcuvid-analyze
  --trim
  --seek
  -f, --format
  --audio-copy
  --audio-codec
  --audio-bitrate
  --audio-stream
  --audio-samplerate
  --audio-resampler
  --audio-file
  --audio-ignore-decode-error
  --audio-ignore-notrack-error
  --audio-source
  --chapter
  --chapter-copy
  --sub-copy
  -m, --mux-options
  --output-buf
  --output-thread
  --max-procfps
  あわせてffmpegのdllを追加。(ソースは QSVEnc_2.29_lgpl_dll_src.7z)
・リサイズが行われるときは、入力からのsar比の自動反映を行わないように。
・--levelの読み取りを柔軟に。
・コマンドライン読み取り時のエラー表示を改善。

2016.01.05 (2.00β4)
[NVEncC]
・DeviceIDを指定してエンコードできるように。(--device <int>)
・利用可能なGPUのDeviceIdを表示できるように。(--check-device)
・--check-hw, --check-featuresがdeviceIDを引数にとれるように。
  指定したdeviceIdをチェックする。省略した場合は"DeviceId:0"をチェック。

2015.12.31 (2.00β3v2)
[NVEncC]
・ffmpegのdllをSSE2ビルドに変更。
  ソースはQSVEnc_2.26_lgpl_dll_src.7zのものを流用。

2015.12.26 (2.00β3)
[NVEncC]
・--qp-init, --qp-max, --qp-minを追加。
・デバッグ用のメッセージを大量に追加。

2015.12.06 (2.00β2)
[NVEncC]
・NVENC SDK 6.0に対応
  NVIDIA グラフィックドライバ 358.xx 以降が必要
・NVIDIA CUVIDによるインターレース解除に対応。
  --vpp-deinterlace bob,adaptive
・help-enにoutput-resがなかったのを修正。
・avcuvidでは、下記がサポートされないので、エラーチェックを追加。
  lossless, high444, crop left
・HEVCエンコード時に色関連の設定が反映可能に。

2015.11.29 (2.00β1)
[NVEncC]
・NVIDIA CUVIDによるデコード・リサイズに対応。
  ソフトウェアデコードより高速。
  H.264 / MPEG1 / MPEG2 のデコードに対応。
  --avcuvid
  --output-res <int>x<int>
・ログファイルの出力に対応。
  --log <string>
  --log-level <string>
・ffmpegのdllはQSVEnc_2.22_lgpl_dll_src.7zのものを流用。
  
2015.11.08 (1.13v2)
[NVEncC]
・x64の実行ファイルが最新版になっていなかったので修正。

2015.11.06 (1.13)
[共通]
・VBR2モードを追加。--vbr2。SDKのいうところの2passVBR。
・AQを追加。

[NVEncC]
・y4mからsar情報を受け取れるように。
  特に指定がない場合に、y4mからの情報を使用する。

2015.11.02 (1.12)
[NVEnc]
・fdk-aac (ffmpeg)にもaudio delay cut用のパラメータをNVEnc.iniに追加。

[NVEncC]
・y4mでのyuv422/yuv444読み込みを追加。

2015.10.24 (1.11)
[共通]
・VC2015に移行。
・OSのバージョン情報取得を改善、Windows10に対応。
・NVEncのH.264/AVC high444 profileとロスレス出力に対応。
  QSVEnc
    YUV444出力…プロファイルをhigh444と指定する
    ロスレス出力…CQPでIフレーム、Bフレーム、PフレームのQP値を0にする。

  QSVEncC
    YUV444出力…--profile high444
    ロスレス出力…--lossless

2015.08.18 (1.10)
[共通]
・ハードウェア上限に達した場合のエラーメッセージを表示しようとすると落ちる問題を修正。

2015.07.13 (1.09)
[NVEnc]
・.NET Framework 4.5に移行。
・音声エンコードでフリーズする場合があったのを修正。

[NVEncC]
・特になし。

2015.04.29 (1.08)
[共通]
・環境によって例外:0xc0000096で落ちることがあるのを回避。
[NVEnc]
・音声エンコ前後にバッチ処理を行う機能を追加。
[NVEncC]
・64bit版にavsリーダーを追加。

2015.04.16 (1.07)
[NVEnc]
※要NVEnc.ini更新
・neroを使用すると途中でフリーズする問題を修正。
・いくつかの音声エンコーダの設定を追加。

2015.04.12 (1.06)
[共通]
・VBVバッファサイズを指定するオプションを追加。
・Bluray用出力を行うオプションを追加。
  Bluray用に出力する際には、必ず指定してください。

2015.04.05 (1.05)
[NVEncC]
・--check-featuresをNVENCのない環境で実行するとフリーズする問題を修正。

2015.03.15 (1.04)
[NVEncC]
・英語化が一部不完全だったのを修正。

2015.03.09 (1.03)
[共通]
・1.00からインタレ保持エンコードができていなかったのを修正。
・インタレ保持でtffかbffかを選択できるように。
[NVEncC]
・コンソールで問題が起こることがあるので、ログ表記等を英語化。

2015.02.27 (1.02)
[NVEncC]
・--inputと--outputが効いていなかったのを修正。

2015.02.14 (1.01)
[共通]
・SAR/DARを指定できるように。
[NVEnc]
・自動フィールドシフト使用時以外には、muxerでmuxを行うように。
  muxを一工程削減できる。
[NVEncC]
・--cropオプションを追加。
  LSMASHSourceでdr=trueを使用して高速化できる。
・読み込みでエラーになった際に、エラー情報を表示するように。
・初期化に失敗した際の処理を改善。
・vpyリーダーを追加。
・x64版を追加(avsリーダーは無効)
  
2015.01.24 (1.00)
[共通]
・エンコードログの表示に動きベクトル精度、CABAC、deblockを追加。
・GOP長のデフォルトを0:自動に。
・HEVCの参照距離が適切に設定されないのを修正。
・デフォルトパラメータを高品質よりに調整。
・プリセットを更新。
[NVEncC]
・colormatrix, colorprim, transferが正しく設定されないのを修正。
・短縮オプションの一部がヘルプにないのを修正。
・AVSリーダーでYUY2読み込みが正常に行われていなかったのを修正。

2015.01.24 (1.00 べ～た)
・NVEnc API v5.0に対応
  - HEVCエンコードに対応
・コマンドライン版 NVEncCを追加。
  - raw, y4m, avs読み込みに対応。
・x264guiEx 2.24までの更新を反映。
  - qaacについて、edtsにより音声ディレイのカットをする機能を追加
  - 音声エンコーダにfdkaacを追加
  - muxerのコマンドに--file-formatを追加。
  - flacの圧縮率を変更できるように
  - 音声やmuxerのログも出力できるように
  - 0秒時点にチャプターがないときは、ダミーのチャプターを追加するように。
    Apple形式のチャプター埋め込み時に最初のチャプターが時間指定を無視して0秒時点に振られてしまうのを回避。
  - ログにmuxer/音声エンコーダのバージョンを表示するように。
  - ログが文字化けすることがあるのを改善。
    また、文字コード判定コードのバグを修正。SJISと判定されやすくなっていた。
  - 音声エンコーダにopusencを追加。
  - nero形式のチャプターをUTF-8に変換すると、秒単位に切り捨てられてしまう問題を修正。
  - CPU使用率を表示。

2014.04.21 (0.03)
・nero形式のチャプターをUTF-8に変換してからmuxする機能を追加
・なおも99.9%で停止することがある問題を修正 

2014.04.13 (0.02)
・99.9%で停止してしまう問題を改善…できているかもしれない。

2014.04.05 (0.01)
・nvcuda.dllの存在しない環境で、「コンピューターにnvcuda.dllがないため、プログラムを開始できません。」と出る問題を解決

2014.03.28 (0.00)
・公開版

2014.03.20
製作開始
