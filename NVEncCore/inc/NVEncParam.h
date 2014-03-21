#pragma once
#ifndef __NVENC_PARAM_H__
#define __NVENC_PARAM_H_

#include <limits.h>
#include "CNVEncoder.h"

typedef struct {
	char *desc;
	int value;
} CX_DESC;

const CX_DESC list_avc_profile[] = {
	{ "Baseline", 66 },
	{ "Main",     77 },
	{ "High",     100 },
	{ NULL, NULL }
};

const CX_DESC list_avc_level[] = { 
	{ "auto", 0   },
	{ "1",    10  },
	{ "1b",   9   },
	{ "1.1",  11  },
	{ "1.2",  12  },
	{ "1.3",  13  },
	{ "2",    20  },
	{ "2.1",  21  },
	{ "2.2",  22  },
	{ "3",    30  },
	{ "3.1",  31  },
	{ "3.2",  32  },
	{ "4",    40  },
	{ "4.1",  41  },
	{ "4.2",  42  },
	{ "5",    50  },
	{ "5.1",  51  },
	{ NULL, NULL }
};

const int COLOR_VALUE_AUTO = INT_MAX;
const int HD_HEIGHT_THRESHOLD = 720;
const int HD_INDEX = 2;
const int SD_INDEX = 3;
const CX_DESC list_colorprim[] = {
	{ "undef",     2  },
	{ "auto",      COLOR_VALUE_AUTO },
	{ "bt709",     1  },
	{ "smpte170m", 6  },
	{ "bt470m",    4  },
	{ "bt470bg",   5  },
	{ "smpte240m", 7  },
	{ "film",      8  },
	{ NULL, NULL }
};
const CX_DESC list_transfer[] = {
	{ "undef",     2  },
	{ "auto",      COLOR_VALUE_AUTO },
	{ "bt709",     1  },
	{ "smpte170m", 6  },
	{ "bt470m",    4  },
	{ "bt470bg",   5  },
	{ "smpte240m", 7  },
	{ "linear",    8  },
	{ "log100",    9  },
	{ "log316",    10 },
	{ NULL, NULL }
};
const CX_DESC list_colormatrix[] = {
	{ "undef",     2  },
	{ "auto",      COLOR_VALUE_AUTO },
	{ "bt709",     1  },
	{ "smpte170m", 6  },
	{ "bt470bg",   5  },
	{ "smpte240m", 7  },
	{ "YCgCo",     8  },
	{ "fcc",       4  },
	{ "GBR",       0  },
	{ NULL, NULL }
};
const CX_DESC list_videoformat[] = {
	{ "undef",     5  },
	{ "ntsc",      2  },
	{ "component", 0  },
	{ "pal",       1  },
	{ "secam",     3  },
	{ "mac",       4  },
	{ NULL, NULL } 
};

const CX_DESC list_mv_presicion[] = {
	{ "full-pel", NV_ENC_MV_PRECISION_FULL_PEL },
	{ "half-pel", NV_ENC_MV_PRECISION_HALF_PEL },
	{ "Q-pel",    NV_ENC_MV_PRECISION_QUARTER_PEL },
	{ NULL, NULL }
};

const CX_DESC list_nvenc_rc_method[] = {
	{ "CQP - 固定量子化量", NV_ENC_PARAMS_RC_CONSTQP },
	{ "CBR - 固定ビットレート", NV_ENC_PARAMS_RC_CBR },
	{ "VBR - 可変ビットレート", NV_ENC_PARAMS_RC_VBR },
	{ "VBR_MINQP - 下限QP付き可変ビットレート", NV_ENC_PARAMS_RC_VBR_MINQP },
	{ NULL, NULL }
};

const CX_DESC list_interlaced[] = {
	{ "Progressive", NV_ENC_PIC_STRUCT_FRAME },
	{ "tff",         NV_ENC_PIC_STRUCT_TOP_FIELD },
	{ "bff",         NV_ENC_PIC_STRUCT_BOTTOM_FIELD },
	{ NULL, NULL }
};

static int get_cx_index(const CX_DESC * list, int v) {
	for (int i = 0; list[i].desc; i++)
		if (list[i].value == v)
			return i;
	return 0;
}

static int PARSE_ERROR_FLAG = INT_MIN;
static int get_value_from_chr(const CX_DESC *list, const char *chr) {
	for (int i = 0; list[i].desc; i++)
		if (_stricmp(list[i].desc, chr) == 0)
			return list[i].value;
	return PARSE_ERROR_FLAG;
}

#endif //__NVENC_PARAM_H__
