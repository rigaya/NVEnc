#include <Windows.h>
#include <tchar.h>
#include <algorithm>
#include <vector>
#include <cstdio>
#include "nvenc_version.h"
#include "nvenc_param.h"
#include "nvenc_util.h"

typedef struct INT2 {
	int x;
	int y;
} INT2;

typedef struct FLOAT2 {
	float x;
	float y;
} FLOAT2;

typedef struct OptionInfo {
	TCHAR *long_name;
	TCHAR *short_name;
	DWORD type;
	const CX_DESC *list;
	size_t p_offset;
} OptionInfo;

enum {
	OPTION_TYPE_BOOL = 1,
	OPTION_TYPE_BOOL_REVERSE,
	OPTION_TYPE_INT,
	OPTION_TYPE_FLOAT,
	OPTION_TYPE_INT2,
	OPTION_TYPE_FLOAT2,
	OPTION_TYPE_BOOL2_REVERSE,
	OPTION_TYPE_LIST,
	OPTION_TYPE_CHAR,
	OPTION_TYPE_CQP,
	OPTION_TYPE_CBR,
	OPTION_TYPE_VBR,
	OPTION_TYPE_SHOW,
};

typedef struct CMD_ARG {
	int arg_type;       //LONGかSHORTか
	TCHAR *option_name;  //オプション名(最初の"--"なし)
	TCHAR *value;        //オプションの値
	BOOL ret;           //構造体に読み込まれたかどうか
} CMD_ARG;

enum {
	ARG_TYPE_UNKNOWN = 0,
	ARG_TYPE_LONG    = 1,
	ARG_TYPE_SHORT   = 2,
};

static const std::vector<DWORD> OPTION_NO_VALUE = { 
	OPTION_TYPE_BOOL,
	OPTION_TYPE_BOOL_REVERSE,
	OPTION_TYPE_BOOL2_REVERSE,
};

static const std::vector<OptionInfo> nvenc_options = {
	{ _T("help"),             _T("h"), OPTION_TYPE_SHOW, (CX_DESC *)_T("0"),  NULL },
	{ _T("help"),             _T("?"), OPTION_TYPE_SHOW, (CX_DESC *)_T("0"),  NULL },
	{ _T("help"),             _T("H"), OPTION_TYPE_SHOW, (CX_DESC *)_T("0"),  NULL },
	{ _T("version"),          _T("v"), OPTION_TYPE_SHOW, (CX_DESC *)_T("1"),  NULL },
	{ _T("check-hw"),          _T(""), OPTION_TYPE_SHOW, (CX_DESC *)_T("2"),  NULL },
	{ _T("check-environment"), _T(""), OPTION_TYPE_SHOW, (CX_DESC *)_T("3"),  NULL },
	{ _T("check-features"),    _T(""), OPTION_TYPE_SHOW, (CX_DESC *)_T("4"),  NULL },
	{ _T("input"),            _T("i"), OPTION_TYPE_CHAR,               NULL,  offsetof(InEncodeVideoParam, input.filename) },
	{ _T("output"),           _T("o"), OPTION_TYPE_CHAR,               NULL,  offsetof(InEncodeVideoParam, outputFilename) },
	{ _T("fps"),               _T(""), OPTION_TYPE_INT2,               NULL,  offsetof(InEncodeVideoParam, input.scale) },
	{ _T("input-res"),         _T(""), OPTION_TYPE_INT2,               NULL,  offsetof(InEncodeVideoParam, input.width) },
	{ _T("cqp"),               _T(""), OPTION_TYPE_CQP,                NULL,  NULL },
	{ _T("cbr"),               _T(""), OPTION_TYPE_CBR,                NULL,  NULL },
	{ _T("vbr"),               _T(""), OPTION_TYPE_VBR,                NULL,  NULL },
	{ _T("gop-length"),        _T(""), OPTION_TYPE_INT,                NULL,  offsetof(InEncodeVideoParam, encConfig.gopLength) },
	{ _T("b-frames"),          _T(""), OPTION_TYPE_INT,                NULL,  offsetof(InEncodeVideoParam, encConfig.frameIntervalP) },
	{ _T("mv-precision"),      _T(""), OPTION_TYPE_LIST,  list_mv_presicion,  offsetof(InEncodeVideoParam, encConfig.mvPrecision) },
	{ _T("max-bitrate"),       _T(""), OPTION_TYPE_INT,                NULL,  offsetof(InEncodeVideoParam, encConfig.rcParams.maxBitRate) },
	{ _T("vbv-bufsize"),       _T(""), OPTION_TYPE_INT,                NULL,  offsetof(InEncodeVideoParam, encConfig.rcParams.vbvBufferSize) },
	{ _T("ref"),               _T(""), OPTION_TYPE_INT,                NULL,  offsetof(InEncodeVideoParam, encConfig.encodeCodecConfig.h264Config.maxNumRefFrames) },
	{ _T("no-interlaced"),     _T(""), OPTION_TYPE_BOOL_REVERSE,       NULL,  offsetof(InEncodeVideoParam, picStruct) },
	{ _T("interlaced"),        _T(""), OPTION_TYPE_LIST,    list_interlaced,  offsetof(InEncodeVideoParam, picStruct) },
};

static void show_version() {
	static const TCHAR *const ENABLED_INFO[] = { _T("disabled"), _T("enabled") };
#ifdef _M_IX86
	_ftprintf(stdout, _T("NVEncC (x86) %s by rigaya, build %s %s\n"), VER_STR_FILEVERSION_TCHAR, _T(__DATE__), _T(__TIME__));
#else
	_ftprintf(stdout, _T("NVEncC (x64) %s by rigaya, build %s %s\n"), VER_STR_FILEVERSION_TCHAR, _T(__DATE__), _T(__TIME__));
#endif
	_ftprintf(stdout, _T("\n"));
}

static void show_help() {
	show_version();

	_ftprintf(stdout, _T("Usage: NVEncC.exe [Options] -i <filename> -o <filename>\n"));
		_ftprintf(stdout, _T("\n")
			_T("input can be %s%s%sraw YUV or YUV4MPEG2(y4m) format.\n")
			_T("when raw(default), fps, input-res are also necessary.\n")
			_T("\n")
			_T("output format will be raw H.264/AVC ES.\n")
			_T("\n")
			_T("Example:\n")
			_T("  NVEncC -i \"<avsfilename>\" -o \"<outfilename>\"\n")
			_T("  avs2pipemod -y4mp \"<avsfile>\" | NVEncC --y4m -i - -o \"<outfilename>\"\n")
			_T("\n")
			_T("Options: \n")
			_T("-h,-? --help                      show help\n")
			_T("-v,--version                      show version info\n")
			_T("\n")
			_T("-i,--input <filename>             set input file name\n")
			_T("-o,--output <filename>            set ouput file name\n")
			_T("\n")
			_T(" Input formats (will be estimated from extension if not set.)\n")
			_T("   --raw                          set input as raw format\n")
			_T("   --y4m                          set input as y4m format\n")
			//_T("   --avi                          set input as avi format\n")
			//_T("   --avs                          set input as avs format\n")
			//_T("   --vpy                          set input as vpy format\n")
			//_T("   --vpy-mt                       set input as vpy format in multi-thread\n")
			_T("\n")
			_T("   --interlaced <string>          set as interlaced\n")
			_T("                                    - tff, bff")
			_T("   --fps <int>/<int> or <float>   video frame rate (frames per second)\n")
			_T("\n")
			_T("   --input-res <int>x<int>        input resolution\n")
			_T("   --check-hw                     check if QuickSyncVideo is available\n")
			_T("   --check-lib                    check lib API version installed\n")
			_T("   --check-features               check encode features\n")
			_T("   --check-environment            check environment info\n"));
}

static void show_hw() {

}

static void show_environment_info() {
	TCHAR buf[1024];
	getEnviromentInfo(buf, _countof(buf));
	_ftprintf(stderr, "%s\n", buf);
}

static void print_nvenc_features() {
	NVEncParam nvParam;

	TCHAR buf[1024];
	getEnviromentInfo(buf, _countof(buf));

	auto nvEncCaps = nvParam.GetNVEncCapability(0);

	size_t max_length = 0;
	std::for_each(nvEncCaps.begin(), nvEncCaps.end(), [&max_length](const NVEncCap& x) { max_length = (std::max)(max_length, _tcslen(x.name)); });

	_ftprintf(stderr, "%s\n", buf);

	for (auto cap : nvEncCaps) {
		_ftprintf(stderr, _T("%s"), cap.name);
		for (size_t i = _tcslen(cap.name); i <= max_length; i++)
			_ftprintf(stderr, _T(" "));
		_ftprintf(stderr, _T("%d\n"), cap.value);
	}
}

static inline BOOL check_range(int value, int min, int max) {
	return (min <= value && value <= max);
}

static BOOL auo_strtol(int *i, const TCHAR *str, DWORD len) {
	TCHAR *eptr = NULL;
	int v;
	BOOL ret = TRUE;
	if (len == NULL) len = ULONG_MAX;
	if (*str != _T('{')) {
		v = _tcstol(str, &eptr, 0);
		if (*eptr == _T('\0') || (DWORD)(eptr - str) == len) { *i = v; } else { ret = FALSE; }
	} else {
		str++;
		BOOL multi = (*str == _T('*'));
		v = _tcstol(str + multi, &eptr, 0);
		if (*eptr == _T('}')) { (multi) ? *i *= v : *i += v; } else { ret = FALSE; }
	}
	return ret;
}

static BOOL auo_strtof(float *f, const TCHAR *str, DWORD len) {
	TCHAR *eptr = NULL;
	float v;
	BOOL ret = TRUE;
	if (len == NULL) len = ULONG_MAX;
	if (*str != _T('{')) {
		v = (float)_tcstod(str, &eptr);
		if (*eptr == _T('\0') || (DWORD)(eptr - str) == len) { *f = v; } else { ret = FALSE; }
	} else {
		str++;
		BOOL multi = (*str == _T('*'));
		v = (float)_tcstod(str + multi, &eptr);
		if (*eptr == _T('}')) { (multi) ? *f *= v : *f += v; } else { ret = FALSE; }
	}
	return ret;
}

static BOOL auo_parse_int(int *i, const TCHAR *value, DWORD len) {
	BOOL ret;
	if ((ret = auo_strtol(i, value, len)) == FALSE) {
		size_t len = _tcslen(value);
		if (*value ==_T('[') && value[len-1] == _T(']')) {
			const TCHAR *a, *b, *c;
			if ((a = _tcsstr(value, _T("if>"))) != NULL && (b = _tcsstr(value, _T("else"))) != NULL) {
				int v;
				c = a + _tcslen(_T("if>"));
				ret |= auo_strtol(&v, c, b-c);
				b += _tcslen(_T("else"));
				if (*i > v)
					c = value+1, len = a - c;
				else
					c = b, len = (value + len - 1) - c;
				ret &= auo_strtol(i, c, len);
			}
		}
	}
	return ret;
}

static BOOL auo_parse_float(float *f, const TCHAR *value, DWORD len) {
	BOOL ret;
	if ((ret = auo_strtof(f, value, len)) == FALSE) {
		size_t len = _tcslen(value);
		if (*value == _T('[') && value[len-1] == _T(']')) {
			const TCHAR *a, *b, *c;
			if ((a = _tcsstr(value, _T("if>"))) != NULL && (b = _tcsstr(value, _T("else"))) != NULL) {
				float v;
				c = a + _tcslen(_T("if>"));
				ret |= auo_strtof(&v, c, b-c);
				b += _tcslen(_T("else"));
				if (*f > v)
					c = value+1, len = a - c;
				else
					c = b, len = (value + len - 1) - c;
				ret &= auo_strtof(f, c, len);
			}
		}
	}
	return ret;
}

//以下部分的にwarning C4100を黙らせる
//C4100 : 引数は関数の本体部で 1 度も参照されません。
#pragma warning( push )
#pragma warning( disable: 4100 )

static BOOL set_bool(void *b, const TCHAR *value, const CX_DESC *list) {
	BOOL ret = TRUE;
	if (value) {
		int i = -1;
		if (FALSE != (ret = auo_parse_int(&i, value, NULL)))
			if (FALSE != (ret = check_range(i, FALSE, TRUE)))
				*(int *)b = i;
	} else {
		*(BOOL*)b = TRUE;
	}
	return ret;
}

static BOOL set_bool_reverse(void *b, const TCHAR *value, const CX_DESC *list) {
	BOOL ret = TRUE;
	if (value) {
		int i = -1;
		if (FALSE != (ret = auo_parse_int(&i, value, NULL)))
			if (FALSE != (ret = check_range(i, FALSE, TRUE)))
				*(int *)b = !i;
	} else {
		*(BOOL*)b = FALSE;
	}
	return ret;
}
static BOOL set_int(void *i, const TCHAR *value, const CX_DESC *list) {
	return auo_parse_int((int *)i, value, NULL);
}

static BOOL set_float(void *f, const TCHAR *value, const CX_DESC *list) {
	return auo_parse_float((float *)f, value, NULL);
}

static BOOL set_char(void *s, const TCHAR *value, const CX_DESC *list) {
	tstring *str = (tstring *)s;
	*str = value;
	return TRUE;
}

static BOOL set_int2(void *i, const TCHAR *value, const CX_DESC *list) {
	const size_t len = _tcslen(value);
	//一度値をコピーして分析
	BOOL ret = FALSE;
	for (size_t j = 0; j < len; j++) {
		if (value[j] == _T(':') || value[j] == _T('|') || value[j] == _T(',') || value[j] == _T('/') || value[j] == _T('x')) {
			ret = TRUE;
			if (!(j == _tcslen(_T("<unset>")) && _tcsncicmp(value, _T("<unset>"), _tcslen(_T("<unset>"))) == NULL))
				ret &= auo_parse_int(&((INT2 *)i)->x, value, j);
			if (_tcsicmp(&value[j+1], _T("<unset>")) != NULL)
				ret &= auo_parse_int(&((INT2 *)i)->y, &value[j+1], 0);
			break;
		}
	}
	return ret;
}

static BOOL set_bool2_reverse(void *b, const TCHAR *value, const CX_DESC *list) {
	BOOL ret = TRUE;
	if (value) {
		INT2 i_value = { 0, 0 };
		if (FALSE != (ret = set_int2(&i_value, value, NULL))) {
			if (i_value.x == 0) ((INT2*)b)->x = FALSE;
			if (i_value.y == 0) ((INT2*)b)->y = FALSE;
		}
	} else {
		((INT2*)b)->x = FALSE;
		((INT2*)b)->y = FALSE;
	}
	return TRUE;
}

static BOOL set_float2(void *f, const TCHAR *value, const CX_DESC *list) {
	const size_t len = _tcslen(value);
	BOOL ret = FALSE;
	for (size_t j = 0; j < len; j++) {
		if (value[j] == _T(':') || value[j] == _T('|') || value[j] == _T(',') || value[j] == _T('/') || value[j] == _T('x')) {
			ret = TRUE;
			if (!(j == _tcslen(_T("<unset>")) && _tcsncicmp(value, _T("<unset>"), _tcslen(_T("<unset>"))) == NULL))
				ret &= auo_parse_float(&((FLOAT2 *)f)->x, value, j);
			if (_tcsicmp(&value[j+1], _T("<unset>")) != NULL)
				ret &= auo_parse_float(&((FLOAT2 *)f)->y, &value[j+1], 0);
			break;
		}
	}
	return ret;
}

static BOOL set_list(void *i, const TCHAR *value, const CX_DESC *list) {
	BOOL ret = FALSE;
	for (int j = 0; list[j].desc; j++) {
		if (NULL == _tcsicmp(value, list[j].desc)) {
			*(int*)i = list[j].value;
			ret = TRUE;
			break;
		}
	}
	//数値での指定に対応
	if (!ret) {
		int k = -1;
		if (FALSE != (ret = auo_parse_int(&k, value, NULL))) {
			//取得した数が、listに存在するか確認する
			ret = FALSE;
			for (int i_check = 0; list[i_check].desc; i_check++) {
				if (list[i_check].value == k) {
					*(int*)i = k;
					ret = TRUE;
					break;
				}
			}
		}
	}
	return ret; 
}

static BOOL set_cqp(void *cx, const TCHAR *value, const CX_DESC *list) {
	BOOL ret = TRUE;
	int a[3] = { 0 };
	InEncodeVideoParam *prm = (InEncodeVideoParam *)cx;
	if (   3 == _stscanf_s(value, _T("%d:%d:%d"), &a[0], &a[1], &a[2])
		|| 3 == _stscanf_s(value, _T("%d/%d/%d"), &a[0], &a[1], &a[2])
		|| 3 == _stscanf_s(value, _T("%d.%d.%d"), &a[0], &a[1], &a[2])
		|| 3 == _stscanf_s(value, _T("%d,%d,%d"), &a[0], &a[1], &a[2])) {
		prm->encConfig.rcParams.rateControlMode = NV_ENC_PARAMS_RC_CONSTQP;
		prm->encConfig.rcParams.constQP.qpIntra  = a[0];
		prm->encConfig.rcParams.constQP.qpInterP = a[1];
		prm->encConfig.rcParams.constQP.qpInterB = a[2];
	} else if (1 == _stscanf_s(value, _T("%d"), &a[0])) {
		prm->encConfig.rcParams.rateControlMode = NV_ENC_PARAMS_RC_CONSTQP;
		prm->encConfig.rcParams.constQP.qpIntra  = a[0];
		prm->encConfig.rcParams.constQP.qpInterP = a[0];
		prm->encConfig.rcParams.constQP.qpInterB = a[0];
	} else {
		ret = FALSE;
	}
	return ret;
}

static BOOL set_cbr(void *cx, const TCHAR *value, const CX_DESC *list) {
	int i = 0;
	BOOL ret = auo_parse_int(&i, value, NULL);
	if (FALSE != ret) {
		InEncodeVideoParam *prm = (InEncodeVideoParam *)cx;
		prm->encConfig.rcParams.rateControlMode = NV_ENC_PARAMS_RC_CBR;
		prm->encConfig.rcParams.averageBitRate = i * 1000;
		prm->encConfig.rcParams.maxBitRate = i * 1000;
	}
	return ret;
}

static BOOL set_vbr(void *cx, const TCHAR *value, const CX_DESC *list) {
	int i = 0;
	BOOL ret = auo_parse_int(&i, value, NULL);
	if (FALSE != ret) {
		InEncodeVideoParam *prm = (InEncodeVideoParam *)cx;
		prm->encConfig.rcParams.rateControlMode = NV_ENC_PARAMS_RC_VBR;
		prm->encConfig.rcParams.averageBitRate = i * 1000;
	}
	return ret;
}

static BOOL show(void *cx, const TCHAR *value, const CX_DESC *list) {
	const TCHAR *show_num = (const TCHAR *)list;
	int show_id = strtol(show_num, NULL, 0);
	switch (show_id) {
	case 1: show_version(); break;
	case 2: show_hw(); break;
	case 3: show_environment_info(); break;
	case 4: print_nvenc_features(); break;
	default:
	case 0: show_help(); break;
	}
	return TRUE;
}

#pragma warning( pop )

typedef BOOL (*SET_VALUE) (void *cx, const char *value, const CX_DESC *list);
const SET_VALUE set_value[] = {
	NULL,
	set_bool,
	set_bool_reverse,
	set_int,
	set_float,
	set_int2,
	set_float2,
	set_bool2_reverse,
	set_list,
	set_char,
	set_cqp,
	set_cbr,
	set_vbr,
	show,
};

static inline BOOL option_has_no_value(DWORD type) {
	return OPTION_NO_VALUE.end() != std::find(OPTION_NO_VALUE.begin(), OPTION_NO_VALUE.end(), type);
}

int parse_cmd(InEncodeVideoParam *conf_set, int argc, TCHAR **argv) {
	using namespace std;
	vector<CMD_ARG> arg_list;
	//コマンドラインを整理
	for (int i_arg = 1; i_arg < argc; i_arg++) {
		if (argv[i_arg][0] != _T('-'))
			continue;
		CMD_ARG cmd_opt = { 0 };
		cmd_opt.arg_type = (argv[i_arg][1] == _T('-')) ? ARG_TYPE_LONG : ARG_TYPE_SHORT;
		cmd_opt.option_name = argv[i_arg] + 1 + (cmd_opt.arg_type == ARG_TYPE_LONG);
		if (argv[i_arg+1][0] != _T('-')) {
			i_arg++;
			cmd_opt.value = argv[i_arg];
		}
		arg_list.push_back(cmd_opt);
	}

	auto option_has_no_value = [](DWORD type) {
		for (int i = 0; OPTION_NO_VALUE[i]; i++) {
			if (type == OPTION_NO_VALUE[i])
				return TRUE;
		}
		return FALSE;
	};

	//コマンドラインからのオプションを取得
	for (auto prm : arg_list) {
		for (auto option : nvenc_options) {
			if (0 == ((prm.arg_type == ARG_TYPE_LONG) ? _tcscmp(prm.option_name, option.long_name)
				                                      : _tcsncmp(prm.option_name, option.short_name, 1))) {
				prm.ret = (option_has_no_value(option.type) == FALSE && prm.value == NULL) ? FALSE : TRUE;
				if (FALSE == prm.ret) {
					_ftprintf(stderr, _T("オプションの指定方法に誤りがあります。: -%s%s\n"),
						(prm.arg_type == ARG_TYPE_LONG) ? _T("-") : _T(""), prm.option_name);
					return -1;
				} else {
					prm.ret = set_value[option.type]((void *)((BYTE *)conf_set + option.p_offset), prm.value, option.list);
					if (FALSE == prm.ret) {
						_ftprintf(stderr, _T("オプションの値が読み取れませんでした。-%s%s %s\n"),
						(prm.arg_type == ARG_TYPE_LONG) ? _T("-") : _T(""), prm.option_name, prm.value);
						return -1;
					}
					if (OPTION_TYPE_SHOW == option.type)
						return 1;
				}
				break;
			}
		}
		if (FALSE == prm.ret) {
			_ftprintf(stderr, _T("誤ったオプションが指定されています。: -%s%s\n"),
				(prm.arg_type == ARG_TYPE_LONG) ? _T("-") : _T(""), prm.option_name);
			return -1;
		}
	}

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

	InEncodeVideoParam encPrm = { 0 };
	encPrm.encConfig = NVEncCore::DefaultParam();
	encPrm.inputBuffer = 3;
	encPrm.picStruct = NV_ENC_PIC_STRUCT_FRAME;
	encPrm.preset = 0;

	if (parse_cmd(&encPrm, argc, argv)) {
		return 1;
	}

	NVEncCore nvEnc;

	nvEnc.Initialize(&encPrm);
	nvEnc.InitEncode(&encPrm);

	nvEnc.Encode();

	return 0;
}
