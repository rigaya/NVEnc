#include <Windows.h>
#include <tchar.h>
#include <locale.h>
#include <algorithm>
#include <vector>
#include <cstdio>
#include "NVEncVersion.h"
#include "NVEncCore.h"
#include "NVEncParam.h"
#include "NVEncUtil.h"

static void show_version() {
	static const TCHAR *const ENABLED_INFO[] = { _T("disabled"), _T("enabled") };
#ifdef _M_IX86
	_ftprintf(stdout, _T("NVEncC (x86) %s by rigaya, build %s %s\n"), VER_STR_FILEVERSION_TCHAR, _T(__DATE__), _T(__TIME__));
#else
	_ftprintf(stdout, _T("NVEncC (x64) %s by rigaya, build %s %s\n"), VER_STR_FILEVERSION_TCHAR, _T(__DATE__), _T(__TIME__));
#endif
	_ftprintf(stdout, _T("\n"));
}


//適当に改行しながら表示する
static void print_list_options(FILE *fp, TCHAR *option_name, const CX_DESC *list, int default_index) {
	const TCHAR *indent_space = _T("                                  ");
	const int indent_len = (int)_tcslen(indent_space);
	const int max_len = 77;
	int print_len = _ftprintf(fp, _T("   %s "), option_name);
	while (print_len < indent_len)
		 print_len += _ftprintf(stdout, _T(" "));
	for (int i = 0; list[i].desc; i++) {
		if (print_len + _tcslen(list[i].desc) + _tcslen(_T(", ")) >= max_len) {
			_ftprintf(fp, _T("\n%s"), indent_space);
			print_len = indent_len;
		} else {
			if (i)
				print_len += _ftprintf(fp, _T(", "));
		}
		print_len += _ftprintf(fp, _T("%s"), list[i].desc);
	}
	_ftprintf(fp, _T("\n%s default: %s\n"), indent_space, list[default_index].desc);
}

static void show_help() {
	show_version();

	_ftprintf(stdout, _T("使用方法: NVEncC.exe [オプション] -i <入力ファイル名> -o <出力ファイル名>\n"));
	_ftprintf(stdout, _T("\n")
		_T("入力は %s%sraw YUV, YUV4MPEG2(y4m)です。\n")
		_T("Raw入力時には, fps, input-res の指定も必要です。\n")
		_T("\n")
		_T("出力フォーマットは raw H.264/AVC または H.265/HEVC ESです。\n")
		_T("\n")
		_T("実行例:\n")
		_T("  NVEncC -i \"<avsfilename>\" -o \"<outfilename>\"\n")
		_T("  avs2pipemod -y4mp \"<avsfile>\" | NVEncC --y4m -i - -o \"<outfilename>\"\n")
		_T("\n")
		_T("オプション: \n")
		_T("-h,-? --help                      ヘルプの表示\n")
		_T("-v,--version                      バージョン情報の表示\n")
		_T("   --check-hw                     NVEncが使用可能かチェック\n")
		_T("   --check-features               NVEncの使用可能な機能を表示\n")
		_T("   --check-environment            NVEncの認識している環境情報を表示\n")
		_T("\n")
		_T("-i,--input <filename>             入力ファイル名の指定\n")
		_T("-o,--output <filename>            出力ファイル名の指定\n")
		_T("\n")
		_T(" 入力フォーマット (指定のない場合、拡張子から自動判定)\n")
		_T("   --raw                          rawとしてファイルを読み込み\n")
		_T("   --y4m                          y4mとしてファイルを読み込み\n")
#if AVI_READER
		_T("   --avi                          aviとしてファイルを読み込み\n")
#endif
#if AVS_READER
		_T("   --avs                          avsとしてファイルを読み込み\n")
#endif
		//_T("   --vpy                          vpyとしてファイルを読み込み\n")
		//_T("   --vpy-mt                       vpy(mt)としてファイルを読み込み\n")
		_T("\n")
		_T("   --input-res <int>x<int>        入力解像度\n")
		_T("-f,--fps <int>/<int> or <float>   フレームレートの指定\n")
		_T("\n")
		_T("-c,--codec <string>               出力コーデックの指定\n")
		_T("                                    h264 (or avc), h265 (or hevc)\n")
		_T("   --profile <string>             プロファイルの指定\n")
		_T("                                    H.264: baseline, main, high(デフォルト)\n")
		_T("   --level <string>               Levelの指定\n")
		_T("\n")
		_T("   --cqp <int> or                 固定量子化量でエンコード\n")
		_T("         <int>:<int>:<int>          デフォルト: <I>:<P>:<B>=<%d>:<%d>:<%d>\n")
		_T("   --vbr <int>                    可変ビットレートのビットレート (kbps)\n")
		_T("   --cbr <int>                    固定ビットレートのビットレート (kbps)\n")
		_T("                                    デフォルト %d kbps\n")
		_T("\n")
		_T("   --max-bitrate <int>            最大ビットレート(kbps) / デフォルト: %d kbps\n")
		_T("   --gop-len <int>                GOPのフレーム数 / デフォルト: %d frames%s")
		_T("-b,--bframes <int>                連続Bフレーム数 / デフォルト %d フレーム\n")
		_T("   --ref <int>                    参照距離 / デフォルト %d フレーム\n")
		_T("   --mv-precision <string>        動きベクトル精度 / デフォルト: Q-pel\n")
		_T("                                    Q-pel    … 1/4画素精度 (高精度)\n")
		_T("                                    half-pel … 1/2画素精度\n")
		_T("                                    full-pel … 1  画素精度 (低精度)\n")
		_T("\n")
		_T("H.264/AVC\n")
		_T("   --interlaced                   インタレ保持エンコ\n")
		_T("   --cabac                        CABACを使用する\n")
		_T("   --cavlc                        CAVLCを使用する\n")
		_T("   --deblock                      デブロックフィルタを有効にする\n")
		_T("   --no-deblock                   デブロックフィルタを無効にする\n")
		_T("   --fullrange                    fullrangeの指定\n"),
		(AVI_READER) ? _T("avi, ") : _T(""),
		(AVS_READER) ? _T("avs, ") : _T(""),
		DEFAUTL_QP_I, DEFAULT_QP_P, DEFAULT_QP_B,
		DEFAULT_AVG_BITRATE / 1000, DEFAULT_MAX_BITRATE / 1000,
		DEFAULT_GOP_LENGTH, (DEFAULT_GOP_LENGTH == 0) ? _T(" (自動)") : _T(""),
		DEFAULT_B_FRAMES, DEFAULT_REF_FRAMES);

		print_list_options(stdout, _T("--videoformat <string>"), list_videoformat, 0);
		print_list_options(stdout, _T("--colormatrix <string>"), list_colormatrix, 0);
		print_list_options(stdout, _T("--colorprim <string>"),   list_colorprim,   0);
		print_list_options(stdout, _T("--transfer <string>"),    list_transfer,    0);

		_ftprintf(stdout, _T("\n")
			_T("H.265/HEVC\n")
			_T("   --cu-max <int>                 CUの最大サイズを指定する\n")
			_T("   --cu-min  <int>                CUの最小サイズを指定する\n")
			_T("                                    8, 16, 32 を指定可能"));
}

static void show_hw() {
	show_version();
	
	NVEncParam nvParam;
	nvParam.createCacheAsync(0);
	auto nvEncCaps = nvParam.GetCachedNVEncCapability();
	if (nvEncCaps.size()) {
		_ftprintf(stdout, _T("使用可能なコーデック\n"));
		for (auto codecNVEncCaps : nvEncCaps) {
			_ftprintf(stdout, _T("%s\n"), get_name_from_guid(codecNVEncCaps.codec, list_nvenc_codecs));
		}
	} else {
		_ftprintf(stdout, _T("NVEncは使用できません。\n"));
	}
}

static void show_environment_info() {
	TCHAR buf[1024];
	getEnviromentInfo(buf, _countof(buf));
	_ftprintf(stderr, _T("%s\n"), buf);
}

static void show_nvenc_features() {
	TCHAR buf[1024];
	getEnviromentInfo(buf, _countof(buf));
	
	NVEncParam nvParam;
	nvParam.createCacheAsync(0);
	auto nvEncCaps = nvParam.GetCachedNVEncCapability();


	_ftprintf(stdout, _T("%s\n"), buf);
	
	_ftprintf(stdout, _T("使用可能な各機能の情報を表示します。\n"));
	for (auto codecNVEncCaps : nvEncCaps) {
		_ftprintf(stdout, _T("コーデック: %s\n"), get_name_from_guid(codecNVEncCaps.codec, list_nvenc_codecs));
		size_t max_length = 0;
		std::for_each(codecNVEncCaps.caps.begin(), codecNVEncCaps.caps.end(), [&max_length](const NVEncCap& x) { max_length = (std::max)(max_length, _tcslen(x.name)); });
		for (auto cap : codecNVEncCaps.caps) {
			_ftprintf(stdout, _T("%s"), cap.name);
			for (size_t i = _tcslen(cap.name); i <= max_length; i++)
				_ftprintf(stdout, _T(" "));
			_ftprintf(stdout, _T("%d\n"), cap.value);
		}
		_ftprintf(stdout, _T("\n"));
	}
}

static inline BOOL check_range(int value, int min, int max) {
	return (min <= value && value <= max);
}

int parse_cmd(InEncodeVideoParam *conf_set, NV_ENC_CODEC_CONFIG *codecPrm, int argc, TCHAR **argv) {

	for (int i_arg = 1; i_arg < argc; i_arg++) {
		TCHAR *option_name = nullptr;
		if (argv[i_arg][0] == _T('-')) {
			switch (argv[i_arg][1]) {
				case _T('-'):
					option_name = &argv[i_arg][2];
					break;
				case _T('b'):
					option_name = _T("bframes");
					break;
				case _T('c'):
					option_name = _T("codec");
					break;
				case _T('u'):
					option_name = _T("quality");
					break;
				case _T('f'):
					option_name = _T("fps");
					break;
				case _T('i'):
					option_name = _T("input-file");
					break;
				case _T('o'):
					option_name = _T("output-file");
					break;
				case _T('v'):
					option_name = _T("version");
					break;
				case _T('h'):
				case _T('?'):
					option_name = _T("help");
					break;
				default:
					_ftprintf(stderr, _T("不明なオプションです。 : %s"), argv[i_arg]);
					return -1;
			}
		}

		if (nullptr == option_name) {
			_ftprintf(stderr, _T("不明なオプションです。 : %s"), argv[i_arg]);
			return -1;
		}

		auto get_list_value = [](const CX_DESC * list, const TCHAR *chr, int *value) {
			for (int i = 0; list[i].desc; i++) {
				if (0 == _tcsicmp(list[i].desc, chr)) {
					*value = list[i].value;
					return true;
				}
			}
			return false;
		};
		auto get_list_guid_value = [](const guid_desc * list, const TCHAR *chr, int *value) {
			for (int i = 0; list[i].desc; i++) {
				if (0 == _tcsicmp(list[i].desc, chr)) {
					*value = list[i].value;
					return true;
				}
			}
			return false;
		};

		
#define IS_OPTION(x) (0 == _tcscmp(option_name, _T(x)))
		if (IS_OPTION("help")) {
			show_help();
			return 1;
		} else if (IS_OPTION("version")) {
			show_version();
			return 1;
		} else if (IS_OPTION("check-hw")) {
			show_hw();
			return 1;
		} else if (IS_OPTION("check-environment")) {
			show_environment_info();
			return 1;
		} else if (IS_OPTION("check-features")) {
			show_nvenc_features();
			return 1;
		} else if (IS_OPTION("input-file")) {
			i_arg++;
			conf_set->input.filename = argv[i_arg];
		} else if (IS_OPTION("output-file")) {
			i_arg++;
			conf_set->outputFilename = argv[i_arg];
		} else if (IS_OPTION("fps")) {
			i_arg++;
			int a[2] = { 0 };
			if (   2 == _stscanf_s(argv[i_arg], _T("%d/%d"), &a[0], &a[1])
				|| 2 == _stscanf_s(argv[i_arg], _T("%d:%d"), &a[0], &a[1])
				|| 2 == _stscanf_s(argv[i_arg], _T("%d,%d"), &a[0], &a[1])) {
				conf_set->input.rate  = a[0];
				conf_set->input.scale = a[1];
			} else {
				double d;
				if (1 == _stscanf_s(argv[i_arg], _T("%lf"), &d)) {
					int rate = (int)(d * 1001.0 + 0.5);
					if (rate % 1000 == 0) {
						conf_set->input.rate = rate;
						conf_set->input.scale = 1001;
					} else {
						conf_set->input.scale = 100000;
						conf_set->input.rate = (int)(d * conf_set->input.scale + 0.5);
						int gcd = nv_get_gcd(conf_set->input.rate, conf_set->input.scale);
						conf_set->input.scale /= gcd;
						conf_set->input.rate  /= gcd;
					}
				} else  {
					_ftprintf(stderr, _T("不正な値が指定されています。 %s : %s\n"), option_name, argv[i_arg]);
					return -1;
				}
			}
		} else if (IS_OPTION("input-res")) {
			i_arg++;
			int a[2] = { 0 };
			if (   2 == _stscanf_s(argv[i_arg], _T("%dx%d"), &a[0], &a[1])
				|| 2 == _stscanf_s(argv[i_arg], _T("%d:%d"), &a[0], &a[1])
				|| 2 == _stscanf_s(argv[i_arg], _T("%d,%d"), &a[0], &a[1])) {
				conf_set->input.width  = a[0];
				conf_set->input.height = a[1];
			} else {
				_ftprintf(stderr, _T("不正な値が指定されています。 %s : %s\n"), option_name, argv[i_arg]);
				return -1;
			}
		} else if (IS_OPTION("codec")) {
			i_arg++;
			int value = 0;
			if (get_list_value(list_nvenc_codecs_for_opt, argv[i_arg], &value)) {
				conf_set->codec = value;
			} else {
				_ftprintf(stderr, _T("不正な値が指定されています。 %s : %s\n"), option_name, argv[i_arg]);
				return -1;
			}
		} else if (IS_OPTION("raw")) {
			conf_set->input.type = NV_ENC_INPUT_RAW;
		} else if (IS_OPTION("y4m")) {
			conf_set->input.type = NV_ENC_INPUT_Y4M;
#if AVI_READER
		} else if (IS_OPTION("avi")) {
			conf_set->input.type = NV_ENC_INPUT_AVI;
#endif
#if AVS_READER
		} else if (IS_OPTION("avs")) {
			conf_set->input.type = NV_ENC_INPUT_AVS;
#endif
		} else if (IS_OPTION("cqp")) {
			i_arg++;
			int a[3] = { 0 };
			if (   3 == _stscanf_s(argv[i_arg], _T("%d:%d:%d"), &a[0], &a[1], &a[2])
				|| 3 == _stscanf_s(argv[i_arg], _T("%d/%d/%d"), &a[0], &a[1], &a[2])
				|| 3 == _stscanf_s(argv[i_arg], _T("%d.%d.%d"), &a[0], &a[1], &a[2])
				|| 3 == _stscanf_s(argv[i_arg], _T("%d,%d,%d"), &a[0], &a[1], &a[2])) {
				conf_set->encConfig.rcParams.rateControlMode = NV_ENC_PARAMS_RC_CONSTQP;
				conf_set->encConfig.rcParams.constQP.qpIntra  = a[0];
				conf_set->encConfig.rcParams.constQP.qpInterP = a[1];
				conf_set->encConfig.rcParams.constQP.qpInterB = a[2];
			} else if (1 == _stscanf_s(argv[i_arg], _T("%d"), &a[0])) {
				conf_set->encConfig.rcParams.rateControlMode = NV_ENC_PARAMS_RC_CONSTQP;
				conf_set->encConfig.rcParams.constQP.qpIntra  = a[0];
				conf_set->encConfig.rcParams.constQP.qpInterP = a[0];
				conf_set->encConfig.rcParams.constQP.qpInterB = a[0];
			} else {
				_ftprintf(stderr, _T("不正な値が指定されています。 %s : %s\n"), option_name, argv[i_arg]);
				return -1;
			}
		} else if (IS_OPTION("vbr")) {
			i_arg++;
			int value = 0;
			if (1 == _stscanf_s(argv[i_arg], _T("%d"), &value)) {
				conf_set->encConfig.rcParams.rateControlMode = NV_ENC_PARAMS_RC_VBR;
				conf_set->encConfig.rcParams.averageBitRate = value * 1000;
			} else {
				_ftprintf(stderr, _T("不正な値が指定されています。 %s : %s\n"), option_name, argv[i_arg]);
				return -1;
			}
		} else if (IS_OPTION("cbr")) {
			i_arg++;
			int value = 0;
			if (1 == _stscanf_s(argv[i_arg], _T("%d"), &value)) {
				conf_set->encConfig.rcParams.rateControlMode = NV_ENC_PARAMS_RC_CBR;
				conf_set->encConfig.rcParams.averageBitRate = value * 1000;
				conf_set->encConfig.rcParams.maxBitRate = value * 1000;
			} else {
				_ftprintf(stderr, _T("不正な値が指定されています。 %s : %s\n"), option_name, argv[i_arg]);
				return -1;
			}
		} else if (IS_OPTION("gop-len")) {
			i_arg++;
			int value = 0;
			if (1 == _stscanf_s(argv[i_arg], _T("%d"), &value)) {
				conf_set->encConfig.gopLength = value;
			} else {
				_ftprintf(stderr, _T("不正な値が指定されています。 %s : %s\n"), option_name, argv[i_arg]);
				return -1;
			}
		} else if (IS_OPTION("bframes")) {
			i_arg++;
			int value = 0;
			if (1 == _stscanf_s(argv[i_arg], _T("%d"), &value)) {
				conf_set->encConfig.frameIntervalP = value + 1;
			} else {
				_ftprintf(stderr, _T("不正な値が指定されています。 %s : %s\n"), option_name, argv[i_arg]);
				return -1;
			}
		} else if (IS_OPTION("max-bitrate")) {
			i_arg++;
			int value = 0;
			if (1 == _stscanf_s(argv[i_arg], _T("%d"), &value)) {
				conf_set->encConfig.rcParams.maxBitRate = value * 1000;
			} else {
				_ftprintf(stderr, _T("不正な値が指定されています。 %s : %s\n"), option_name, argv[i_arg]);
				return -1;
			}
		} else if (IS_OPTION("vbv-bufsize")) {
			i_arg++;
			int value = 0;
			if (1 == _stscanf_s(argv[i_arg], _T("%d"), &value)) {
				conf_set->encConfig.rcParams.vbvBufferSize = value * 1000;
			} else {
				_ftprintf(stderr, _T("不正な値が指定されています。 %s : %s\n"), option_name, argv[i_arg]);
				return -1;
			}
		} else if (IS_OPTION("ref")) {
			i_arg++;
			int value = 0;
			if (1 == _stscanf_s(argv[i_arg], _T("%d"), &value)) {
				codecPrm[NV_ENC_H264].h264Config.maxNumRefFrames = value;
				codecPrm[NV_ENC_HEVC].hevcConfig.maxNumRefFramesInDPB = value;
			} else {
				_ftprintf(stderr, _T("不正な値が指定されています。 %s : %s\n"), option_name, argv[i_arg]);
				return -1;
			}
		} else if (IS_OPTION("mv-precision")) {
			i_arg++;
			int value = 0;
			if (get_list_value(list_mv_presicion, argv[i_arg], &value)) {
				conf_set->encConfig.mvPrecision = (NV_ENC_MV_PRECISION)value;
			} else {
				_ftprintf(stderr, _T("不正な値が指定されています。 %s : %s\n"), option_name, argv[i_arg]);
				return -1;
			}
		} else if (IS_OPTION("interlaced")) {
			conf_set->picStruct = NV_ENC_PIC_STRUCT_FIELD_TOP_BOTTOM;
		} else if (IS_OPTION("cavlc")) {
			codecPrm[NV_ENC_H264].h264Config.entropyCodingMode = NV_ENC_H264_ENTROPY_CODING_MODE_CAVLC;
		} else if (IS_OPTION("cabac")) {
			codecPrm[NV_ENC_H264].h264Config.entropyCodingMode = NV_ENC_H264_ENTROPY_CODING_MODE_CABAC;
		} else if (IS_OPTION("no-deblock")) {
			codecPrm[NV_ENC_H264].h264Config.disableDeblockingFilterIDC = 1;
		} else if (IS_OPTION("deblock")) {
			codecPrm[NV_ENC_H264].h264Config.disableDeblockingFilterIDC = 0;
		} else if (IS_OPTION("fullrange")) {
			codecPrm[NV_ENC_H264].h264Config.h264VUIParameters.videoFullRangeFlag = 1;
		} else if (IS_OPTION("videoformat")) {
			i_arg++;
			int value = 0;
			if (get_list_value(list_videoformat, argv[i_arg], &value)) {
				codecPrm[NV_ENC_H264].h264Config.h264VUIParameters.videoFormat = value;
			} else {
				_ftprintf(stderr, _T("不正な値が指定されています。 %s : %s\n"), option_name, argv[i_arg]);
				return -1;
			}
		} else if (IS_OPTION("colormatrix")) {
			i_arg++;
			int value = 0;
			if (get_list_value(list_colormatrix, argv[i_arg], &value)) {
				codecPrm[NV_ENC_H264].h264Config.h264VUIParameters.colourMatrix = value;
			} else {
				_ftprintf(stderr, _T("不正な値が指定されています。 %s : %s\n"), option_name, argv[i_arg]);
				return -1;
			}
		} else if (IS_OPTION("colorprim")) {
			i_arg++;
			int value = 0;
			if (get_list_value(list_colorprim, argv[i_arg], &value)) {
				codecPrm[NV_ENC_H264].h264Config.h264VUIParameters.colourPrimaries = value;
			} else {
				_ftprintf(stderr, _T("不正な値が指定されています。 %s : %s\n"), option_name, argv[i_arg]);
				return -1;
			}
		} else if (IS_OPTION("transfer")) {
			i_arg++;
			int value = 0;
			if (get_list_value(list_transfer, argv[i_arg], &value)) {
				codecPrm[NV_ENC_H264].h264Config.h264VUIParameters.transferCharacteristics = value;
			} else {
				_ftprintf(stderr, _T("不正な値が指定されています。 %s : %s\n"), option_name, argv[i_arg]);
				return -1;
			}
		} else if (IS_OPTION("level")) {
			i_arg++;
			bool flag = false;
			int value = 0;
			if (get_list_value(list_avc_level, argv[i_arg], &value)) {
				codecPrm[NV_ENC_H264].h264Config.level = value;
				flag = true;
			}
			if (get_list_value(list_hevc_level, argv[i_arg], &value)) {
				codecPrm[NV_ENC_HEVC].hevcConfig.level = value;
				flag = true;
			}
			if (!flag) {
				if (1 == _stscanf_s(argv[i_arg], _T("%d"), &value)) {
					codecPrm[NV_ENC_H264].h264Config.level = value;
					codecPrm[NV_ENC_HEVC].hevcConfig.level = value;
				} else {
					_ftprintf(stderr, _T("不正な値が指定されています。 %s : %s\n"), option_name, argv[i_arg]);
					return -1;
				}
			}
		} else if (IS_OPTION("profile")) {
			i_arg++;
			bool flag = false;
			GUID zero = { 0 };
			GUID result_guid = get_guid_from_name(argv[i_arg], h264_profile_names);
			if (0 != memcmp(&result_guid, &zero, sizeof(result_guid))) {
				conf_set->encConfig.profileGUID = result_guid;
				flag = true;
			}
			int result = get_value_from_name(argv[i_arg], h265_profile_names);
			if (-1 != result) {
				codecPrm[NV_ENC_HEVC].hevcConfig.tier = result;
				flag = true;
			}
			if (!flag) {
				_ftprintf(stderr, _T("不正な値が指定されています。 %s : %s\n"), option_name, argv[i_arg]);
				return -1;
			}
		} else if (IS_OPTION("cu-max")) {
			i_arg++;
			int value = 0;
			if (get_list_value(list_hevc_cu_size, argv[i_arg], &value)) {
				codecPrm[NV_ENC_HEVC].hevcConfig.maxCUSize = (NV_ENC_HEVC_CUSIZE)value;
			} else {
				_ftprintf(stderr, _T("不正な値が指定されています。 %s : %s\n"), option_name, argv[i_arg]);
				return -1;
			}
		} else if (IS_OPTION("cu-min")) {
			i_arg++;
			int value = 0;
			if (1 == _stscanf_s(argv[i_arg], _T("%d"), &value)) {
				codecPrm[NV_ENC_HEVC].hevcConfig.minCUSize = (NV_ENC_HEVC_CUSIZE)value;
			} else {
				_ftprintf(stderr, _T("不正な値が指定されています。 %s : %s\n"), option_name, argv[i_arg]);
				return -1;
			}
		}
	}

#undef IS_OPTION
	//オプションチェック
	if (0 == conf_set->input.filename.length()) {
		_ftprintf(stderr, _T("入力ファイルが正しく指定されていません。\n"));
		return -1;
	}
	if (0 == conf_set->outputFilename.length()) {
		_ftprintf(stderr, _T("出力ファイルが正しく指定されていません。\n"));
		return -1;
	}

	return 0;
}

int _tmain(int argc, TCHAR **argv) {

	_tsetlocale(LC_ALL, _T("japanese"));

	InEncodeVideoParam encPrm = { 0 };
	NV_ENC_CODEC_CONFIG codecPrm[2] = { 0 };
	encPrm.encConfig = NVEncCore::DefaultParam();
	encPrm.inputBuffer = 16;
	encPrm.picStruct = NV_ENC_PIC_STRUCT_FRAME;
	encPrm.preset = 0;
	codecPrm[NV_ENC_H264] = NVEncCore::DefaultParamH264();
	codecPrm[NV_ENC_HEVC] = NVEncCore::DefaultParamHEVC();

	if (parse_cmd(&encPrm, codecPrm, argc, argv)) {
		return 1;
	}

	encPrm.encConfig.encodeCodecConfig = codecPrm[encPrm.codec];

	NVEncCore nvEnc;

	nvEnc.Initialize(&encPrm);
	nvEnc.InitEncode(&encPrm);
	nvEnc.PrintEncodingParamsInfo(NV_LOG_INFO);
	nvEnc.Encode();

	return 0;
}
