
# NVEncC オプションリスト


## コマンドラインの例


### 基本的なコマンドの表記
```Batchfile
NVEncC.exe [Options] -i <filename> -o <filename>
```

### もっと実用的なコマンド
#### cuvidデコードを使用する例
```Batchfile
NVEncC --avhw -i "<mp4(H.264/AVC) file>" -o "<outfilename.264>"
```

#### cuvidデコードを使用する例 (インタレ保持)
```Batchfile
NVEncC --avhw --interlace tff -i "<mp4(H.264/AVC) file>" -o "<outfilename.264>"
```

#### avs(Avisynth)の例 (avsやvpyはvfw経由でも読み込み可能です)
```Batchfile
NVEncC -i "<avsfile>" -o "<outfilename.264>"
```

#### パイプ利用の例
```Batchfile
avs2pipemod -y4mp "<avsfile>" | NVEncC --y4m -i - -o "<outfilename.264>"
```

#### ffmpegからパイプ渡し

```Batchfile
ffmpeg -y -i "<ソース動画>" -an -pix_fmt yuv420p -f yuv4mpegpipe - | NVEncC --y4m -i - -o "<outfilename.264>"
```

#### ffmpegから映像と音声を両方パイプ渡したい
--> "nut"フォーマットでくるんで受け渡しするとよいでしょう
```Batchfile
ffmpeg -y -i "<input>" <options for ffmpeg> -codec:a copy -codec:v rawvideo -pix_fmt yuv420p -f nut - | NVEncC --avsw -i - --audio-codec aac -o "<outfilename.mp4>"
```

#### raw H.264/ESのmux
H.264/ESで出力し、mp4に格納したり、AAC音声とmuxする場合には、L-SMASHを使って、

```Batchfile
muxer.exe -i "<raw H.264/ES file>" -i "<ADTS-AAC>" -o "<muxed mp4 file>"
```

としてAAC音声と多重化できます。音声がALACの場合には、

```Batchfile
muxer.exe -i "<raw H.264/ES file>" -o "<video mp4file>"
remuxer.exe -i "<video mp4file>" -i "<m4a(ALAC in mp4)file>" -o "<muxed mp4 file>"
```

のように2段階のステップが必要です。

同様にmkvtoolnixに含まれるmkvmergeでmuxし、mkvに格納することもできます。


## オプションの指定方法

```
-<短縮オプション名>、--<オプション名> <引数>  
引数なしの場合は単体で効果を発揮。

引数のタイプは
- なし
- <int>　　 整数で指定
- <float>　小数点で指定
- <string> 文字列で指定

引数の [ ] 内は、省略可能です。

--(no-)xxx
と付いている場合は、--no-xxxとすることで、--xxxと逆の効果を得る。  
例1: --xxx : xxxを有効にする → --no-xxx: xxxを無効にする  
例2: --xxx : xxxを無効にする → --no-xxx: xxxを有効にする
```

## 表示系オプション

### -h,-? --help
ヘルプの表示

### -v, --version
バージョンの表示

### --check-device
NVEncが認識している利用可能なGPUのDeviceIdを表示

### --check-hw [&lt;int&gt;]
ハードウェアエンコの可否の表示。数字でDeviceIDを指定できる。省略した場合は"0"。

### --check-features [&lt;int&gt;]
NVEncの使用可能なエンコード機能を表示する。数字でDeviceIDを指定できる。省略した場合は"0"。

### --check-environment
NVEncCの認識している環境情報を表示

### --check-codecs, --check-decoders, --check-encoders
利用可能な音声コーデック名を表示

### --check-formats
利用可能な出力フォーマットを表示

### --check-protocols
利用可能なプロトコルを表示

### --check-filters
利用可能な音声フィルタを表示

### --check-avversion
dllのバージョンを表示

## エンコードの基本的なオプション

### -d, --device &lt;int&gt;
NVEncで使用するDeviceIdを指定する。使用できるデバイスは、[--check-device](#--check-device)で確認できる。

デフォルトは自動。複数利用可能なGPUがある場合、自動的に使用するGPUを選択する。

判断基準は  
- 指定のエンコードが可能かどうか
- --avhwが指定されていれば入力ファイルのHWデコードが可能かどうか
- Video Engineの使用率が低い方
- GPUの使用率が低い方
- GPUの世代が新しい方 (Compute Capabilityで判定)
- GPUのコア数が多い方

Video Engineの使用率とGPUの使用率の取得には、x64版は[NVMLライブラリ](https://developer.nvidia.com/nvidia-management-library-nvml)、x86版はnvidia-smi.exeを実行して取得している。

nvidia-smi.exeは通常ドライバと一緒に"C:\Program Files\NVIDIA Corporation\NVSMI\nvidia-smi.exe"にインストールされている。


### -c, --codec &lt;string&gt;
エンコードするコーデックの指定
 - h264 (デフォルト)
 - hevc

### -o, --output &lt;string&gt;
出力ファイル名の表示、"-"でパイプ出力

### -i, --input &lt;string&gt;
入力ファイル名の設定、"-"でパイプ入力

NVEncの入力方法は下の表のとおり。入力フォーマットをしてしない場合は、拡張子で自動的に判定される。

| 使用される読み込み |  対象拡張子 |
|:---|:---|          
| Avisynthリーダー    | avs |
| VapourSynthリーダー | vpy |
| aviリーダー         | avi |
| y4mリーダー         | y4m |
| rawリーダー         | yuv |
| avhw/avswリーダー | それ以外 |

| 入力方法の対応色空間 | yuv420 | yuy2 | yuv422 | yuv444 | rgb24 | rgb32 |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| raw | ○ |  |  |  |  |  |
| y4m | ◎ |  | ◎ | ◎ |  |  |
| avi | ○ | ○ |  |  | ○ | ○ |
| avs | ◎ | ○ | ◎ | ◎ |  |  |
| vpy | ◎ |  | ◎ | ◎ |  |  |
| avhw | ◎ |  |  |  |  |  |
| avsw | ◎ |  | ◎ | ◎ | ○ | ○ |

◎ … 8bit / 9bit / 10bit / 12bit / 14bit / 16bitに対応  
○ … 8bitのみ対応

### --raw
入力をraw形式に設定する。
入力解像度、入力fpsの設定が必要。

### --y4m
入力をy4m(YUV4MPEG2)形式として読み込む。

### --avi
入力ファイルをaviファイルとして読み込む。

### --avs
入力ファイルをAvisynthで読み込む。

### --vpy
入力ファイルをVapourSynthで読み込む。

### --avsw
avformat + sw decoderを使用して読み込む。
ffmpegの対応するほとんどのコーデックを読み込み可能。

### --avhw [&lt;string&gt;]
avformat + cuvid decoderを使用して読み込む。
デコードからエンコードまでを一貫してGPUで行うため高速。

オプションでcuvid使用時にデコーダのモードを指定できるが、通常は特に指定せずに"--avhw"のみで問題ない。
省略された場合はnative cuvidモードでデコードする。
- native (デフォルト)
- cuda

| コーデック | 対応状況 |
|:---|:---:|
| MPEG1      | ○ |
| MPEG2      | ○ |
| H.264/AVC  | ○ |
| H.265/HEVC | ○ |
| VP8        | × |
| VP9        | ○ |
| VC-1       | × |
| WMV3/WMV9  | × |

### --interlace &lt;string&gt;
**入力**フレームがインターレースかどうかと、そのフィールドオーダーを設定する。

[--vpp-deinterlace](#--vpp-deinterlace-string) / [--vpp-afs](#--vpp-afs-param1value1param2value2) によりNVEncC内でインタレ解除を行ったり、そのままインタレ保持エンコードを行う。(インタレ保持エンコードはH.264のみ)

- none ... プログレッシブ
- tff ... トップフィールドファースト
- bff ... ボトムフィールドファースト

### --crop &lt;int&gt;,&lt;int&gt;,&lt;int&gt;,&lt;int&gt;
左、上、右、下の切り落とし画素数。

### --fps &lt;int&gt;/&lt;int&gt; or &lt;float&gt;
入力フレームレートの設定。raw形式の場合は必須。

### --input-res &lt;int&gt;x&lt;int&gt;
入力解像度の設定。raw形式の場合は必須。

### --output-res &lt;int&gt;x&lt;int&gt;
出力解像度の設定。入力解像度と異なる場合、自動的にHW/GPUリサイズを行う。

指定がない場合、入力解像度と同じになり、リサイズは行われない。


## エンコードモードのオプション

デフォルトはCQP(固定量子化量)。

### --cqp &lt;int&gt; or &lt;int&gt;:&lt;int&gt;:&lt;int&gt;　(固定量子化量)
CQP(固定量子化量)でエンコードを行う。&lt;Iフレーム&gt;:&lt;Pフレーム&gt;:&lt;Bフレーム&gt;のQP値を設定。

基本的にQP値は I &lt; P &lt; B になるように設定することをおすすめ。

### --cbr &lt;int&gt;   (固定ビットレート)
### --cbrhq &lt;int&gt; (固定ビットレート (高品質))
### --vbr &lt;int&gt;   (可変ビットレート)
### --vbrhq &lt;int&gt; (可変ビットレート (高品質))
ビットレートをkbps単位で指定してエンコードを行う。


## その他のオプション


### --output-depth &lt;int&gt;
出力ビット深度を設定。
- 8 ... 8bit (デフォルト)
- 10 ... 10bit

### --max-bitrate &lt;int&gt;
最大ビットレート(kbps単位)。

### --qp-init &lt;int&gt; or &lt;int&gt;:&lt;int&gt;:&lt;int&gt;
初期QP値を&lt;Iフレーム&gt;:&lt;Pフレーム&gt;:&lt;Bフレーム&gt;で設定する。
ビットレート指定のエンコードモードを使用する際に、
エンコード開始時に適用するQP値として指定される。

動画の先頭部分で必要以上にビットレートが割り振られたり、
逆に圧縮がかかりすぎてひどい画質になる場合など、
動画の先頭部分の画質を調整したい場合に使用する。

### --qp-min &lt;int&gt; or &lt;int&gt;:&lt;int&gt;:&lt;int&gt;
最小QP値を&lt;Iフレーム&gt;:&lt;Pフレーム&gt;:&lt;Bフレーム&gt;で設定する。
ビットレート指定のエンコードモード使用時のみ有効。設定したQP値より低いQP値は使用されなくなる。

ビットレート指定モードなどで、静止画などの部分で過剰にビットレートが割り当てられることがあるのを抑制したりするのに使用する。

### --qp-max &lt;int&gt; or &lt;int&gt;:&lt;int&gt;:&lt;int&gt;
最大QP値を&lt;Iフレーム&gt;:&lt;Pフレーム&gt;:&lt;Bフレーム&gt;設定する。
ビットレート指定のエンコードモード使用時のみ有効。設定したQP値より高いQP値は使用されなくなる。

指定したビットレートを超えてでも、動画のどんな部分でもある程度の品質を維持したい場合に使用する。

### --vbr-quality &lt;float&gt;
VBRモード使用時の目標品質を設定する。(0.0-51.0, 0 = 自動)

### --lookahead &lt;int&gt;
lookaheadを有効にし、その対象範囲をフレーム数で指定する。(0-32)
画質の向上に役立つとともに、適応的なI,Bフレーム挿入が有効になる。

### --no-i-apapt
lookahead有効時の適応的なIフレーム挿入を無効化する。

### --no-b-apapt
lookahead有効時の適応的なBフレーム挿入を無効化する。

### --strict-gop
固定GOP長を強制する。

### --gop-len &lt;int&gt;
最大GOP長。lookaheadオフでは、この値が常に使用される。(可変ではない)

### -b, --bframes &lt;int&gt;
連続Bフレーム数。

### --ref &lt;int&gt;
参照距離を設定する。NVEncではあまり増やしても品質は向上しない。

### --weightp
重み付きPフレームを使用する。 [H.264のみ]

### --aq
フレーム内(空間方向)の適応的量子化を有効にする。(デフォルト:オフ)

### --aq-temporal
フレーム間(時間方向)の適応的量子化を有効にする。(デフォルト:オフ)

### --aq-strength &lt;int&gt;
AQ強度を指定する。(1(弱) ～ 15(強)、0 = 自動)

### --bref-mode &lt;string&gt; [H.264のみ]
Bフレームの参照モードを指定する。
- disabled (default)
- each
  すべてのBフレームを参照フレームとして利用する  
- middle
  偶数番目のBフレームのみが参照フレームとして利用できる  

### --direct &lt;string&gt; [H.264のみ]
H.264のBDirect modeを指定する。
- auto(default)
- none
- spatial
- temporal

### --(no-)adapt-transform [H.264のみ]
H.264のadaptive transform modeを有効(無効)にする。

### --mv-precision &lt;string&gt;
動きベクトル精度 / デフォルト: auto
- auto     ... 自動
- Q-pel    ... 1/4画素精度 (高精度)
- half-pel ... 1/2画素精度
- full-pel ... 1 画素精度 (低精度)

### --level &lt;string&gt;
エンコードするコーデックのLevelを指定する。指定しない場合は自動的に決定される。
```
h264:  auto, 1, 1b, 1.1, 1.2, 1.3, 2, 2.1, 2.2, 3, 3.1, 3.2, 4, 4.1, 4.2, 5, 5.1, 5.2
hevc:  auto, 1, 2, 2.1, 3, 3.1, 4, 4.1, 5, 5.1, 5.2, 6, 6.1, 6.2
```

### --profile &lt;string&gt;
エンコードするコーデックのプロファイルを指定する。指定しない場合は自動的に決定される。
```
h264:  auto, baseline, main, high, high444
hevc:  auto, main, main10, main444
```

### --tier &lt;string&gt;
コーデックのtierを指定する。
```
hevc:  main, high
```

### --sar &lt;int&gt;:&lt;int&gt;
SAR比 (画素アスペクト比) の指定。

### --dar &lt;int&gt;:&lt;int&gt;
DAR比 (画面アスペクト比) の指定。

### --fullrange
フルレンジYUVとしてエンコードする。

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
出力データのchroma location flagを 0 ～ 5 の範囲で指定する。  
デフォルト = 0 (unspecified)

### --max-cll &lt;int&gt;,&lt;int&gt; [HEVCのみ]
MaxCLL and MaxFall を nits で指定する。
```
--max-cll 1000,300
```

### --master-display &lt;string&gt; [HEVCのみ]
Mastering display data の設定。
```
Example: --master-display G(13250,34500)B(7500,3000)R(34000,16000)WP(15635,16450)L(10000000,1)
```

各種フラグの設定。

### --cabac [H.264のみ]
CABACを使用する。 (デフォルト: オン)

### --cavlc [H.264のみ]
CAVLCを使用する。 (デフォルト: オフ)

### --bluray [H.264のみ]
Bluray用出力を行う。(デフォルト: オフ)

### --lossless [H.264のみ]
ロスレス出力を行う。自動的に--profile high444が指定される。(デフォルト: オフ)

### --(no-)deblock [H.264のみ]
デブロックフィルタを有効にする。(デフォルト: オン)

### --cu-max &lt;int&gt; [HEVCのみ]
### --cu-min &lt;int&gt; [HEVCのみ]
それぞれCUの最大、最小サイズを指定する。8, 16, 32 を指定可能。
HEVCの規格では64まで存在するが、現状NVENCでは32までしかサポートされていない。  
**画質が低下する恐れがあることがわかっているので、--cu-min / --cu-max の使用は非推奨。**

## 入出力 / 音声 / 字幕などのオプション

### --input-analyze &lt;int&gt;
libavが読み込み時に解析するファイルの時間を秒で指定。デフォルトは5。
音声トラックなどが正しく抽出されない場合、この値を大きくしてみてください(例:60)。

### --trim &lt;int&gt;:&lt;int&gt;[,&lt;int&gt;:&lt;int&gt;][,&lt;int&gt;:&lt;int&gt;]...
指定した範囲のフレームのみをエンコードする。

### --seek [&lt;int&gt;:][&lt;int&gt;:]&lt;int&gt;[.&lt;int&gt;]
書式は、hh:mm:ss.ms。"hh"や"mm"は省略可。
高速だが不正確なシークをしてからエンコードを開始する。正確な範囲指定を行いたい場合は[--trim](#--trim-intintintintintint)で行う。
```
例1: --seek 0:01:15.400
例2: --seek 1:15.4
例3: --seek 75.4
```

### --input-format &lt;string&gt;
avhw/avswリーダー使用時に、入力のフォーマットを指定する。

### -f, --output-format &lt;string&gt;
muxerに出力フォーマットを指定して出力する。

### --video-track &lt;int&gt;
エンコード対象の映像トラックの選択。avsw/avhwリーダー使用時のみ有効。
 - 1  ... 最も高解像度の映像トラック (デフォルト)
 - 2  ... 2番目に高解像度の映像トラック
    ...
 - -1 ... 最も低解像度の映像トラック
 - -2 ... 2番目に低解像度の映像トラック

### --video-streamid &lt;int&gt;
エンコード対象の映像トラックをstream idで選択。

出力フォーマットは出力拡張子から自動的に決定されるので、通常、特に指定する必要はないが、このオプションで出力フォーマットを強制できる。

使用可能なフォーマットは[--check-formats](#--check-formats)で確認できる。H.264/HEVCをElementary Streamで出力する場合には、"raw"を指定する。

### --audio-copy [&lt;int&gt;[,&lt;int&gt;]...]
音声をそのままコピーしながら映像とともに出力する。avhw/avswリーダー使用時のみ有効。

tsなどでエラーが出るなどしてうまく動作しない場合は、[--audio-codec](#--audio-codec-intstring)で一度エンコードしたほうが安定動作するかもしれない。

```
例: トラック番号#1,#2を抽出
--audio-copy 1,2
```

### --audio-codec [[&lt;int&gt;?]&lt;string&gt;[:&lt;string&gt;=&lt;string&gt;][,&lt;string&gt;=&lt;string&gt;]...]
音声をエンコードして映像とともに出力する。使用可能なコーデックは[--check-encoders](#--check-codecs---check-decoders---check-encoders)で確認できる。

[&lt;int&gt;]で、抽出する音声トラック(1,2,...)を指定することもできる。

さらに、[&lt;string&gt;=&lt;string&gt;]の形式で、音声エンコーダのオプションを指定することもできる。
```
例1: 音声をmp3に変換
--audio-codec libmp3lame

例2: 音声の第2トラックをaacに変換
--audio-codec 2?aac

例3: aacエンコーダのパラメータ"aac_coder"に低ビットレートでより高品質な"twoloop"を指定
--audio-codec aac:aac_coder=twoloop
```

### --audio-bitrate [&lt;int&gt;?]&lt;int&gt;
音声をエンコードする際のビットレートをkbpsで指定する。

[&lt;int&gt;]で、抽出する音声トラック(1,2,...)を指定することもできる。
```
例1: --audio-bitrate 192   (音声を192kbpsで変換)
例2: --audio-bitrate 2?256 (音声の第2トラックを256kbpsで変換)
```

### --audio-profile [[&lt;int&gt;?]&lt;string&gt;
音声をエンコードする際、そのプロファイルを指定する。

### --audio-stream [&lt;int&gt;?][&lt;string1&gt;][:&lt;string2&gt;]
音声チャンネルの分離・統合などを行う。
--audio-streamが指定された音声トラックは常にエンコードされる。(コピー不可)
,(カンマ)で区切ることで、入力の同じトラックから複数のトラックを生成できる。

##### 書式
&lt;int&gt;に処理対象のトラックを指定する。

&lt;string1&gt;に入力として使用するチャンネルを指定する。省略された場合は入力の全チャンネルを使用する。

&lt;string2&gt;に出力チャンネル形式を指定する。省略された場合は、&lt;string1&gt;のチャンネルをすべて使用する。

```
例1: --audio-stream FR,FL
デュアルモノから左右のチャンネルを2つのモノラル音声に分離する。

例2: --audio-stream :stereo
どんな音声もステレオに変換する。

例3: --audio-stream 2?5.1,5.1:stereo
入力ファイルの第２トラックを、5.1chの音声を5.1chとしてエンコードしつつ、ステレオにダウンミックスしたトラックを生成する。
実際に使うことがあるかは微妙だが、書式の紹介例としてはわかりやすいかと。
```

##### 使用できる記号
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
音声のサンプリング周波数をHzで指定する。
[&lt;int&gt;]で、抽出する音声トラック(1,2,...)を指定することもできる。
```
例1: --audio-bitrate 44100   (音声を44100Hzに変換)
例2: --audio-bitrate 2?22050 (音声の第2トラックを22050Hzに変換)
```

### --audio-resampler &lt;string&gt;
音声チャンネルのmixやサンプリング周波数変換に使用されるエンジンの指定。
- swr  ... swresampler (デフォルト)
- soxr ... sox resampler (libsoxr)

### --audio-file [&lt;int&gt;][&lt;string&gt;?]&lt;string&gt;
指定したパスに音声を抽出する。出力フォーマットは出力拡張子から自動的に決定する。avhw/avswリーダー使用時のみ有効。

[&lt;int&gt;]で、抽出する音声トラック(1,2,...)を指定することもできる。
```
例: test_out2.aacにトラック番号#2を抽出
--audio-file 2?"test_out2.aac"
```

[&lt;string&gt;]では、出力フォーマットを指定することができる。
```
例: 拡張子なしでもadtsフォーマットで出力
--audio-file 2?adts:"test_out2"  
```

### --audio-filter [&lt;int&gt;?]&lt;string&gt;
音声に音声フィルタを適用する。適用可能なフィルタは[こちら](https://ffmpeg.org/ffmpeg-filters.html#Audio-Filters)。


[&lt;int&gt;]で、抽出する音声トラック(1,2,...)を指定することもできる。

```
例1: --audio-filter volume=0.2     (音量を下げる例)
例2: --audio-filter 2?volume=-4db  (第2トラックの音量を下げる例)
```

### --audio-ignore-decode-error &lt;int&gt;
指定した連続する音声のデコードエラーの数をカウントし、閾値以内ならエラーを無視して処理を継続し、エラーの箇所は無音に置き換える。

デフォルトは10。 0とすれば、1回でもデコードエラーが起これば処理を中断してエラー終了する。

### --audio-source &lt;string&gt;
--audio-copyと併用することで、外部音声ファイルをmuxする。

### --chapter &lt;string&gt;
指定したチャプターファイルを読み込み反映させる。
nero形式とapple形式に対応する。--chapter-copyとは併用できない。

### --chapter-copy
チャプターをコピーする。

### --sub-copy [&lt;int&gt;[,&lt;int&gt;]...]
字幕をコピーする。avhw/avswリーダー使用時のみ有効。
[&lt;int&gt;[,&lt;int&gt;]...]で、抽出する字幕トラック(1,2,...)を指定することもできる。

対応する字幕は、PGS/srt/txt/ttxtなど。

```
例: 字幕トラック #1と#2をコピー
--sub-copy 1,2
```

### -m, --mux-option &lt;string1&gt;:&lt;string2&gt;
mux時にオプションパラメータを渡す。&lt;string1&gt;にオプション名、&lt;string2&gt;にオプションの値を指定する。

```
例: HLS用の出力
-i <input> -o test.m3u8 -f hls -m hls_time:5 -m hls_segment_filename:test_%03d.ts --gop-len 30
```

### --avsync &lt;string&gt;
  - cfr (default)  
    入力はCFRを仮定し、入力ptsをチェックしない。

  - forcecfr  
    入力ptsを見ながら、CFRに合うようフレームの水増し・間引きを行い、音声との同期が維持できるようにする。主に、入力がvfrやRFFなどのときに音ズレしてしまう問題への対策。

  - vfr  
    入力に従い、フレームのタイムスタンプをそのまま引き渡す。avsw/avhwリーダによる読み込みの時のみ使用可能。また、--trimとは併用できない。

## vppオプション


### --vpp-deinterlace &lt;string&gt;
[--interlace](#--interlace-string) tff か [--interlace](#--interlace-string) bff が指定されていて、[--avhw](#--avhw-string)使用時のみ有効。HWによるインタレ解除を使用する。

- none   ... インタレ解除を行わない
- normal ... 標準的な60i→30pインタレ解除。
- adaptive ... normalと同じ
- bob ... 60i→60pインタレ解除。

avhwを使用していないがインタレ解除を行いたい場合や、24fps化(Inverse Telecine)を行いたい場合は、[--vpp-afs](#--vpp-afs-param1value1param2value2)を使用する。

### --vpp-rff
Repeat Field Flagを反映して、フレームを再構築する。rffによる音ズレ問題が解消できる。[--avhw](#--avhw-string)使用時のみ有効。

rff=1の場合のみの対応。(rff > 1には対応しない) また、[--trim](#--trim-intintintintintint), [--vpp-deinterlace](#--vpp-deinterlace-string)とは併用できない。

### --vpp-afs [&lt;param1&gt;=&lt;value1&gt;][,&lt;param2&gt;=&lt;value2&gt;],...
自動フィールドシフトによるインタレ解除を行う。

**パラメータ** ... 基本的にはAviutl版のパラメータをそのまま使用する。
- top=&lt;int&gt;           (上)
- bottom=&lt;int&gt;        (下)
- left=&lt;int&gt;          (左)
- right=&lt;int&gt;         (右)  
  判定に使用する領域から除外する範囲の指定。NVEncでは、"左"と"右"は4の倍数である必要がある。

- method_switch=&lt;int&gt; (切替点)  (0-256)  
  切替点が大きいほど、新方式の判定になりやすい(0で常に新方式判定off)

- coeff_shift=&lt;int&gt;   (判定比)  (0-256)  
  判定比率が小さいほど、フィールドをシフトしにくい(0で常にシフト判定off)

- thre_shift=&lt;int&gt;    (縞(シフト))  (0-1024)  
  シフトの判定に使用する縞検出の閾値。値が小さいほど、縞と判定されやすくなる。

- thre_deint=&lt;int&gt;    (縞(解除))  (0-1024)  
  縞解除用の縞検出の閾値。値が小さいほど、縞と判定されやすくなる。

- thre_motion_y=&lt;int&gt; (Y動き) (0-1024)  
- thre_motion_c=&lt;int&gt; (C動き) (0-1024)  
  動き検出の閾値。値が小さいほど、動きと判定されやすくなる。

- level=&lt;int&gt;         (解除Lv)  
  縞解除の方法の選択。(0 - 4)

| 解除Lv | | |
|:---|:---|:---|
| Lv0 | 解除なし | 横縞模様の解除を行わない。<br>フィールドシフトで組み合わされた新しいフレームがそのまま出力になる。|
| Lv1 | フィールド三重化 | フィールドシフトで組み合わされた新しいフレームに、さらに１つ前の フィールドを残像として足しこむ。<br>動きによる縞模様は完全に残像に変換される。 |
| Lv2 | 縞検出二重化 | フレーム内で縞模様を検出して、縞の部分を平均化して残像に変える。<br>フィールド単位の動きが少ない映像向け。 |
| Lv3 | 動き検出二重化 | 前のフレームと比較をして、動き(色の変化)があった部分だけ縞の平均化を行う。 <br>解除Lv2だと平均化されてしまう静止した横縞模様を保存できる。<br>静止したテロップの細かい文字や、アニメなどの枠線付きの静止画の 輪郭をつぶしたくない場合に使用する。| 
| Lv4 | 動き検出補間 | 前のフレームと比較をして動きがあった部分は、片方のフィールドを潰して残す方のフィールドの画像で補間する。<br>残像はなくなりますが、この解除がかかった部分は縦の解像度が半分になる。 |
| Lv5 | 斜め線補正補間 | **非対応** |

- shift=&lt;bool&gt;        (フィールドシフト)  
  フィールドシフトを行う。

- drop=&lt;bool&gt;         (間引き)  
  フィールドシフトを行うことで生じた表示時間の1フレームより短いフレームを間引く。これを有効にするとVFR(可変フレームレート)になるので注意。
  NVEncCでmuxしながら出力する場合には、このタイムコードは自動的に反映される。
  一方、raw出力する場合には、タイムコード反映されないので、vpp-afsのオプションにtimecode=trueを追加してタイムコードを別途出力し、あとからtimecodeファイルを含めてmuxする必要がある。

- smooth=&lt;bool&gt;       (スムージング)  
- 24fps=&lt;bool&gt;        (24fps化)   
  24fps化を強制する、映画・アニメ用のオプション。フィールドシフトと間引きをonにする必要がある。

- tune=&lt;bool&gt;         (調整モード)  
  縞模様と動きの判定結果の確認用。

| 色 | 意味 |
|:---:|:---|
| 青 | 動きを検出 |
| 灰 | 縞を検出 |
| 水色 | 動き + 縞を検出 |

- rff=&lt;bool&gt;  
  入力フレームにrffフラグ付きのプログレフレームがある場合、これをチェックしてインタレ解除処理に反映する。rffフラグ付きのプログレフレームについては、フィールドシフトを行わずに、フレームの表示時間の修正のみを行う。

- log=&lt;bool&gt;  
  フレームごとの判定状況等をcsvファイルで出力。(デバッグ用のログ出力)

- timecode=&lt;bool&gt;  
  タイムコードを出力する。
  
**一括設定用オプション**

  たくさんあるパラメータを一括指定するためのオプション。一括設定用オプションは必ず先に読み込まれ、個別オプションの指定があればそちらで上書きされる。

- ini=&lt;string&gt;  
  指定したini設定ファイルから設定を読み込む。この設定ファイルはAviutl版自動フィールドシフト 高速化 7.5a+20以降のafs.aufで出力できるものを使用する。
  
```
[AFS_STG]
up=8
bottom=8
left=16
right=16
method_watershed=91
coeff_shift=191
thre_shift=447
thre_deint=44
thre_Ymotion=111
thre_Cmotion=222
mode=4
field_shift=1
drop=1
smooth=1
force24=1
tune_mode=0
rff=0
log=0
```

- preset=&lt;string&gt;

以下の表のプリセットをロードします。

```
例: アニメプリセットをロード後、"24fps"をonに、"rff"を"on"に
--vpp-afs preset=anime,24fps=true,rff=true
```

|              | default | triple<br>(動き重視) | double<br>(二重化) | anime<br>cinema<br>(アニメ/映画) | min_afterimg<br>(残像最小化) |  24fps<br>(24fps固定)  | 30fps<br>(30fps固定) |
|:---          |:---:| :---:| :---:|:---:|:---:| :---:| :---:|
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

**vpp-afs使用上の注意**  
- Aviutl版とは全く同じ挙動にはなりません。
- Aviutl版の下記機能には非対応です。
  - 解除Lv5
  - シーンチェンジ検出(解除Lv1)
  - 編集モード
  - ログ保存
  - ログ再生
  - YUY2補間
  - シフト・解除なし

### --vpp-resize &lt;string&gt;
リサイズのアルゴリズムを指定する。

要nppi64_80.dllに"○"のあるものは、[NPPライブラリ](https://developer.nvidia.com/npp)を使用しており、x64版のみ対応。また、使用には別途nppi64_80.dllをダウンロードし、NVEncC64.exeと同じフォルダに配置する必要がある。

| オプション名 | 説明 | 要nppi64_80.dll |
|:---|:---|:---:|
| default  | 自動的に適切なものを選択 | |
| bilinear | 線形補間 | |
| spline36 | 6x6 Spline補間 | |
| nn            | 最近傍点選択 | ○ |
| npp_linear    | nppの線形補間 | ○ |
| cubic         | 4x4 3次補間 | ○ |
| cubic_bspline | 4x4 3次補間 (B=1, C=0)       | ○ |
| cubic_catmull | 4x4 3次補間 (B=0, C=1/2)      | ○ |
| cubic_b05c03  | 4x4 3次補間 (B=1/2, C=3/10)   | ○ |
| super         | nppのsuper sampling(詳細不明) | ○ |
| lanczos       | Lanczos法                    | ○ |

  
### --vpp-knn [&lt;param1&gt;=&lt;value1&gt;][,&lt;param2&gt;=&lt;value2&gt;],...

**パラメータ**
- radius=&lt;int&gt;  (default=3, 1-5)  
  適用半径。

- strength=&lt;float&gt;  (default=0.08, 0.0 - 1.0)    
  フィルタの強さ。

- lerp=&lt;float&gt;  (default=0.2, 0.0 - 1.0)  
  ノイズ除去ピクセルへのオリジナルピクセルのブレンド度合い。

- th_lerp=&lt;float&gt;   (default=0.8, 0.0 - 1.0)  
  エッジ検出の閾値。

```
例: すこし強め
--vpp-knn radius=3,strength=0.10,lerp=0.1
```

### --vpp-pmd [&lt;param1&gt;=&lt;value1&gt;][,&lt;param2&gt;=&lt;value2&gt;],...
正則化pmd法によるノイズ除去。弱めのノイズ除去を行いたいときに使用する。

**パラメータ**
- apply_count=&lt;int&gt;  (default=2, 1- )  
  適用回数。デフォルトは2。

- strength=&lt;float&gt;  (default=100, 0-100)  
  フィルタの強さ。

- threshold=&lt;float&gt;  (default=100, 0-255)  
  フィルタの輪郭検出の閾値。小さいほど輪郭を保持するようになるが、フィルタの効果も弱まる。

```
例: すこし弱め
--vpp-pmd apply_count=2,strength=90,threshold=120
```

### --vpp-gauss &lt;int&gt;
適用サイズを指定してガウスフィルタをかける。サイズは3,5,7のどれか。

nppi64_80.dll導入が必要で、x64版のみ使用可。

### --vpp-unsharp [&lt;param1&gt;=&lt;value1&gt;][,&lt;param2&gt;=&lt;value2&gt;],...
unsharpフィルタ。輪郭・ディテール強調用のフィルタ。

**パラメータ**
- radius=&lt;int&gt; (default=3, 1-9)  
  輪郭・ディテール検出の範囲。より大きな値とすることで、より広い範囲のディテールに反応して強調をかけるようになる。

- weight=&lt;float&gt; (default=0.5, 0-10)  
  輪郭・ディテール強調の強さ。より大きな値とすることで、強く強調がかかる。

- threshold=&lt;float&gt;  (default=10.0, 0-255)  
  輪郭・ディテール検出の閾値。閾値以上の差異がある画素に対して、輪郭強調を行う。

```
例: やや強め
--vpp-unsharp weight=1.0
```

### --vpp-edgelevel [&lt;param1&gt;=&lt;value1&gt;][,&lt;param2&gt;=&lt;value2&gt;],...
エッジレベル調整フィルタ。輪郭強調用のフィルタ。

**パラメータ**
- strength=&lt;float&gt; (default=5.0, -31 - 31)  
  輪郭強調の強さ。より大きな値とすることで、輪郭強調が強力になる。

- threshold=&lt;float&gt;  (default=20.0, 0 - 255)  
  輪郭強調を行わないようにするノイズの閾値。より大きな値ほど大きな輝度の変化をノイズとして扱うようになる。

- black=&lt;float&gt;  (default=0.0, 0-31)  
  輪郭の黒い部分について、より黒くシュートさせて輪郭を強調するようにする。

- white=&lt;float&gt;  (default=0.0, 0-31)  
  輪郭の白く部分について、より白くシュートさせて輪郭を強調するようにする。

```
例: やや強め(Aviutl版のデフォルト)
--vpp-edgelevel strength=10.0,threshold=16.0,black=0,white=0

例: 輪郭の黒い部分を気持ち強める
--vpp-edgelevel strength=5.0,threshold=24.0,black=6.0
```

### --vpp-deband [&lt;param1&gt;=&lt;value1&gt;][,&lt;param2&gt;=&lt;value2&gt;],...

**パラメータ**
- range=&lt;int&gt; (default=15, 0-127)  
  ぼかす範囲。この範囲内の近傍画素からサンプルを取り、ブラー処理を行う。

- sample=&lt;int&gt; (default=1, 0-2)  
  - 設定値：0  
    周辺1画素を参照し、元の画素値を維持したまま処理を行う。

  - 設定値：1  
    周辺1画素とその点対称画素の計2画素を参照し、ブラー処理を行う。

  - 設定値：2  
    周辺2画素とその点対称画素の計4画素を参照し、ブラー処理を行う。

- thre=&lt;int&gt; (一括設定)
- thre_y=&lt;int&gt; (default=15, 0-31)
- thre_cb=&lt;int&gt; (default=15, 0-31)
- thre_cr=&lt;int&gt; (default=15, 0-31)  
  y,cb,cr 各成分の閾値。この値が高いと階調飛びを減らす一方で、細かい線などが潰れやすくなる。

- dither=&lt;int&gt; (一括設定)
- dither_y=&lt;int&gt; (default=15, 0-31)
- dither_c=&lt;int&gt; (default=15, 0-31)  
  y成分と cb+cr成分のディザの強さ。

- seed=&lt;int&gt;  
  乱数シードの変更。 (default=1234)

- blurfirst (default=off)  
  ブラー処理を先にすることでディザ強度を減らしつつ、階調飛びが多い素材での効果を上げる。
  全体的に副作用が強くなり細かい線が潰れやすくなる。

- rand_each_frame (default=off)  
  毎フレーム使用する乱数を変更する。

```
例:
--vpp-deband range=31,dither=12,rand_each_frame
```

### --vpp-tweak [&lt;param1&gt;=&lt;value1&gt;][,&lt;param2&gt;=&lt;value2&gt;],...

**パラメータ**
- brightness=&lt;float&gt; (default=0.0, -1.0 - 1.0)  

- contrast=&lt;float&gt; (default=1.0, -2.0 - 2.0)  

- gamma=&lt;float&gt; (default=1.0, 0.1 - 10.0)  

- saturation=&lt;float&gt; (default=1.0, 0.0 - 3.0)  

- hue=&lt;float&gt; (default=0.0, -180 - 180)  

```
例:
--vpp-tweak brightness=0.1,contrast=1.5,gamma=0.75
```

### --vpp-pad &lt;int&gt,&lt;int&gt,&lt;int&gt,&lt;int&gt
指定のピクセル数(偶数)分のパディングを行う。左、上、右、下の順にピクセル数で指定する。

### --vpp-delogo &lt;string&gt;[,&lt;param1&gt;=&lt;value1&gt;][,&lt;param2&gt;=&lt;value2&gt;],...
ロゴファイルとロゴ消しのオプションを指定する。ロゴファイルは、".lgd",".ldp",".ldp2"に対応。

**パラメータ**
- select=&lt;string&gt;  
ロゴパックの場合に、使用するロゴを以下のいずれかで指定する。

  - ロゴ名
  - インデックス (1,2,...)
  - 自動選択用iniファイル
```
 [LOGO_AUTO_SELECT]
 logo<連番数字>=<マッチパターン>,<リストに表示されているロゴ名(完全一致!)>
```

 例:
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

- pos=&lt;int&gt;:&lt;int&gt;  
1/4画素精度のロゴ位置の調整。Aviutlで言うところの &lt;位置 X&gt;:&lt;位置 Y&gt;。

- depth=&lt;int&gt;  
ロゴの透明度の補正。デフォルト128。Aviutlで言うところの &lt;深度&gt;。

- y=&lt;int&gt;  
- cb=&lt;int&gt;  
- cr=&lt;int&gt;  
ロゴの各色成分の補正。Aviutlで言うところの &lt;Y&gt;, &lt;Cb&gt;, &lt;Cr&gt;。  

- auto_fade=&lt;bool&gt;  
ロゴの実際の濃さに合わせて、fade値を自動的に調整する。デフォルト = off。  
  
- auto_nr=&lt;bool&gt;  
ロゴの輪郭周辺のノイズを除去する際、その強さを自動的に変化させる。デフォルト = off。  

- nr_area=&lt;int&gt;  
ロゴの輪郭周辺に対するノイズ除去適用範囲の広さ。(default=0 (オフ), 0 - 3)  

- nr_value=&lt;int&gt;  
ロゴの輪郭周辺に対するノイズ除去の強さ。(default=0 (オフ), 0 - 4)  

- log=&lt;bool&gt;  
auto_fade, auto_nrを使用した場合のfade値の推移をログに出力する。

```
例:
--vpp-delogo logodata.ldp2,select=delogo.auf.ini,auto_fade=true,auto_nr=true,nr_value=3,nr_area=1,log=true
```


### --vpp-perf-monitor
各フィルタのパフォーマンス測定を行い、適用したフィルタの1フレームあたりの平均処理時間を最後に出力する。全体のエンコード速度がやや遅くなることがある点に注意。



## 制御系のオプション

### --cuda-schedule &lt;string&gt;
  主に、GPUのタスク終了を待機する際のCPUの挙動を決める。デフォルトはauto。

- auto (デフォルト)  
  CUDAのドライバにモード決定を委ねる。

- spin  
  常にCPUを稼働させ、GPUタスクの終了を監視する。復帰のレイテンシが最小となり、最も高速だが、1コア分のCPU使用率を常に使用する。

- yeild  
  基本的にはspinと同じだが、他のスレッドがあればそちらに譲る。

- sync  
 GPUタスクの終了まで、スレッドをスリープさせる。性能が落ちる可能性があるかわりに、特にHWデコード使用時に、CPU使用率を大きく削減する。

### --output-buf &lt;int&gt;
出力バッファサイズをMB単位で指定する。デフォルトは8、最大値は128。0で使用しない。

出力バッファはディスクへの書き込みをアプリケーション側で制御し、バッファサイズ分たまるまで書き出さないようにする。
これにより、ディスクへの書き込み単位が大きくなり、エンコード中のディスクへの読み書きの混在を防ぎ、高速化が期待できる。
またディスク上でのファイルの断片化の抑止も期待できる。

一方、あまり大きく設定しすぎると、逆に遅くなることがあるので注意。基本的にはデフォルトのままで良いと思われる。

file以外のプロトコルを使用する場合には、この出力バッファは使用されず、この設定は反映されない。
また、出力バッファ用のメモリは縮退確保するので、必ず指定した分確保されるとは限らない。

### --output-thread &lt;int&gt;
出力スレッドを使用するかどうかを指定する。
- -1 ... 自動(デフォルト)
-  0 ... 使用しない
-  1 ... 使用する  
出力スレッドを使用すると、メモリ使用量が増加するが、エンコード速度が向上する場合がある。

### --log &lt;string&gt;
ログを指定したファイルに出力する。

### --log-level &lt;string&gt;
ログ出力の段階を選択する。不具合などあった場合には、--log-level debug --log log.txtのようにしてデバッグ用情報を出力したものをコメントなどで教えていただけると、不具合の原因が明確になる場合があります。
- error ... エラーのみ表示
- warn ... エラーと警告を表示
- info ... 一般的なエンコード情報を表示、デフォルト
- debug ... デバッグ情報を追加で出力
- trace ... フレームごとに情報を出力

### --max-procfps &lt;int&gt;
エンコード速度の上限を設定。デフォルトは0 ( = 無制限)。
複数本NVENCでエンコードをしていて、ひとつのストリームにCPU/GPUの全力を奪われたくないというときのためのオプション。
```
例: 最大速度を90fpsに制限
--max-procfps 90
```

### --perf-monitor [&lt;string&gt;][,&lt;string&gt;]...
エンコーダのパフォーマンス情報を出力する。パラメータとして出力したい情報名を下記から選択できる。デフォルトはall (すべての情報)。

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
[--perf-monitor](#--perf-monitor-stringstring)でパフォーマンス測定を行う時間間隔をms単位で指定する(50以上)。デフォルトは 500。