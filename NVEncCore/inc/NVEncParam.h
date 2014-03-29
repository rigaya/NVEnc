//  -----------------------------------------------------------------------------------------
//    NVEnc by rigaya
//  -----------------------------------------------------------------------------------------
//   ソースコードについて
//   ・無保証です。
//   ・本ソースコードを使用したことによるいかなる損害・トラブルについてrigayaは責任を負いません。
//   以上に了解して頂ける場合、本ソースコードの使用、複製、改変、再頒布を行って頂いて構いません。
//  -----------------------------------------------------------------------------------------

#pragma once
#ifndef __NVENC_PARAM_H__
#define __NVENC_PARAM_H_

#include <limits.h>
#include "CNVEncoder.h"
#include "NVEncCore.h"

typedef struct {
	TCHAR *desc;
	int value;
} CX_DESC;

const CX_DESC list_avc_profile[] = {
	{ _T("Baseline"), 66 },
	{ _T("Main"),     77 },
	{ _T("High"),     100 },
	{ NULL, NULL }
};

const CX_DESC list_avc_level[] = { 
	{ _T("auto"), 0   },
	{ _T("1"),    10  },
	{ _T("1b"),   9   },
	{ _T("1.1"),  11  },
	{ _T("1.2"),  12  },
	{ _T("1.3"),  13  },
	{ _T("2"),    20  },
	{ _T("2.1"),  21  },
	{ _T("2.2"),  22  },
	{ _T("3"),    30  },
	{ _T("3.1"),  31  },
	{ _T("3.2"),  32  },
	{ _T("4"),    40  },
	{ _T("4.1"),  41  },
	{ _T("4.2"),  42  },
	{ _T("5"),    50  },
	{ _T("5.1"),  51  },
	{ NULL, NULL }
};

const int COLOR_VALUE_AUTO = INT_MAX;
const int HD_HEIGHT_THRESHOLD = 720;
const int HD_INDEX = 2;
const int SD_INDEX = 3;
const CX_DESC list_colorprim[] = {
	{ _T("undef"),     2  },
	{ _T("auto"),      COLOR_VALUE_AUTO },
	{ _T("bt709"),     1  },
	{ _T("smpte170m"), 6  },
	{ _T("bt470m"),    4  },
	{ _T("bt470bg"),   5  },
	{ _T("smpte240m"), 7  },
	{ _T("film"),      8  },
	{ NULL, NULL }
};
const CX_DESC list_transfer[] = {
	{ _T("undef"),     2  },
	{ _T("auto"),      COLOR_VALUE_AUTO },
	{ _T("bt709"),     1  },
	{ _T("smpte170m"), 6  },
	{ _T("bt470m"),    4  },
	{ _T("bt470bg"),   5  },
	{ _T("smpte240m"), 7  },
	{ _T("linear"),    8  },
	{ _T("log100"),    9  },
	{ _T("log316"),    10 },
	{ NULL, NULL }
};
const CX_DESC list_colormatrix[] = {
	{ _T("undef"),     2  },
	{ _T("auto"),      COLOR_VALUE_AUTO },
	{ _T("bt709"),     1  },
	{ _T("smpte170m"), 6  },
	{ _T("bt470bg"),   5  },
	{ _T("smpte240m"), 7  },
	{ _T("YCgCo"),     8  },
	{ _T("fcc"),       4  },
	{ _T("GBR"),       0  },
	{ NULL, NULL }
};
const CX_DESC list_videoformat[] = {
	{ _T("undef"),     5  },
	{ _T("ntsc"),      2  },
	{ _T("component"), 0  },
	{ _T("pal"),       1  },
	{ _T("secam"),     3  },
	{ _T("mac"),       4  },
	{ NULL, NULL } 
};

const CX_DESC list_mv_presicion[] = {
	{ _T("full-pel"), NV_ENC_MV_PRECISION_FULL_PEL    },
	{ _T("half-pel"), NV_ENC_MV_PRECISION_HALF_PEL    },
	{ _T("Q-pel"),    NV_ENC_MV_PRECISION_QUARTER_PEL },
	{ NULL, NULL }
};

const CX_DESC list_nvenc_rc_method[] = {
	{ _T("CQP - 固定量子化量"),                     NV_ENC_PARAMS_RC_CONSTQP   },
	{ _T("CBR - 固定ビットレート"),                 NV_ENC_PARAMS_RC_CBR       },
	{ _T("VBR - 可変ビットレート"),                 NV_ENC_PARAMS_RC_VBR       },
	{ _T("VBR_MINQP - 下限QP付き可変ビットレート"), NV_ENC_PARAMS_RC_VBR_MINQP },
	{ NULL, NULL }
};

const CX_DESC list_interlaced[] = {
	{ _T("Progressive"), NV_ENC_PIC_STRUCT_FRAME            },
	{ _T("tff"),         NV_ENC_PIC_STRUCT_FIELD_TOP_BOTTOM },
	{ _T("bff"),         NV_ENC_PIC_STRUCT_FIELD_TOP_BOTTOM },
	{ NULL, NULL }
};
const CX_DESC list_entropy_coding[] = {
	{ _T("auto"),  NV_ENC_H264_ENTROPY_CODING_MODE_AUTOSELECT },
	{ _T("cabac"), NV_ENC_H264_ENTROPY_CODING_MODE_CABAC      },
	{ _T("cavlc"), NV_ENC_H264_ENTROPY_CODING_MODE_CAVLC      },
	{ NULL, NULL }
};

const CX_DESC list_bdirect[] = {
	{ _T("auto"),     NV_ENC_H264_BDIRECT_MODE_AUTOSELECT },
	{ _T("disabled"), NV_ENC_H264_BDIRECT_MODE_DISABLE    },
	{ _T("temporal"), NV_ENC_H264_BDIRECT_MODE_TEMPORAL   },
	{ _T("spatial"),  NV_ENC_H264_BDIRECT_MODE_SPATIAL    },
	{ NULL, NULL }
};

const CX_DESC list_fmo[] = {
	{ _T("auto"),     NV_ENC_H264_FMO_AUTOSELECT },
	{ _T("enabled"),  NV_ENC_H264_FMO_ENABLE     },
	{ _T("disabled"), NV_ENC_H264_FMO_DISABLE    },
	{ NULL, NULL }
};
const CX_DESC list_adapt_transform[] = {
	{ _T("auto"),     NV_ENC_H264_ADAPTIVE_TRANSFORM_AUTOSELECT },
	{ _T("disabled"), NV_ENC_H264_ADAPTIVE_TRANSFORM_DISABLE    },
	{ _T("enabled"),  NV_ENC_H264_ADAPTIVE_TRANSFORM_ENABLE     },
	{ NULL, NULL }
};

const CX_DESC list_preset[] = {
	{ _T("fast"),    NV_ENC_PRESET_HP      },
	{ _T("default"), NV_ENC_PRESET_DEFAULT },
	{ _T("best"),    NV_ENC_PRESET_HQ      },
	{ _T("bluray"),  NV_ENC_PRESET_BD      },
	{ NULL, NULL }
};
const CX_DESC list_preset_ja[] = {
	{ _T("高速"),   NV_ENC_PRESET_HP       },
	{ _T("標準"),   NV_ENC_PRESET_DEFAULT  },
	{ _T("高品質"), NV_ENC_PRESET_HQ       },
	{ _T("Bluray"), NV_ENC_PRESET_BD       },
	{ NULL, NULL }
};

static const TCHAR *get_desc(const CX_DESC * list, int v) {
	for (int i = 0; list[i].desc; i++)
		if (list[i].value == v)
			return list[i].desc;
	return NULL;
}

static int get_cx_index(const CX_DESC * list, int v) {
	for (int i = 0; list[i].desc; i++)
		if (list[i].value == v)
			return i;
	return 0;
}

static int get_cx_index(const CX_DESC * list, const char *chr) {
	for (int i = 0; list[i].desc; i++)
		if (0 == strcmp(list[i].desc, chr))
			return i;
	return 0;
}

static int get_cx_value(const CX_DESC * list, const char *chr) {
	for (int i = 0; list[i].desc; i++)
		if (0 == strcmp(list[i].desc, chr))
			return list[i].value;
	return 0;
}

static int PARSE_ERROR_FLAG = INT_MIN;
static int get_value_from_chr(const CX_DESC *list, const char *chr) {
	for (int i = 0; list[i].desc; i++)
		if (_stricmp(list[i].desc, chr) == 0)
			return list[i].value;
	return PARSE_ERROR_FLAG;
}

static int get_value(int id, const std::vector<NVEncCap>& capList) {
	for (auto cap_info : capList) {
		if (cap_info.id == id)
			return cap_info.value;
	}
	return 0;
}

class NVEncParam : public NVEncCore
{
public:
	NVEncParam();
	~NVEncParam();

	int createCacheAsync(int deviceID);
	std::vector<NV_ENC_CONFIG> GetCachedNVEncH264Preset();
	std::vector<NVEncCap> GetCachedNVEncCapability();
	
	std::vector<NV_ENC_CONFIG> GetNVEncH264Preset(int deviceID);
	std::vector<NVEncCap> GetNVEncCapability(int deviceID);
protected:
	static unsigned int __stdcall createCacheLoader(void *prm);
	void createCache(int deviceID);

	int NVEncParam::OpenEncoder(int deviceID);
	int mCurrentDeviceID;
	std::vector<NV_ENC_CONFIG> m_presetCache;
	std::vector<NVEncCap> m_capsCache;

	HANDLE thCreateCache;
};

#endif //__NVENC_PARAM_H__
