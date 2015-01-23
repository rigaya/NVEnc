//  -----------------------------------------------------------------------------------------
//    NVEnc by rigaya
//  -----------------------------------------------------------------------------------------
//   ソースコードについて
//   ・無保証です。
//   ・本ソースコードを使用したことによるいかなる損害・トラブルについてrigayaは責任を負いません。
//   以上に了解して頂ける場合、本ソースコードの使用、複製、改変、再頒布を行って頂いて構いません。
//  -----------------------------------------------------------------------------------------

#pragma once

#include <limits.h>
#include "nvEncodeAPI.h"
#include "NVEncoderPerf.h"
#include "NVEncCore.h"

typedef struct {
	TCHAR *desc;
	int value;
} CX_DESC;

typedef struct {
    GUID id;
    TCHAR *desc;
    unsigned int value;
} guid_desc;

const guid_desc h264_profile_names[] = {
    { NV_ENC_H264_PROFILE_BASELINE_GUID, _T("Baseline"), 66 },
    { NV_ENC_H264_PROFILE_MAIN_GUID,     _T("Main"),     77 },
    { NV_ENC_H264_PROFILE_HIGH_GUID,     _T("High"),    100 },
    //{ NV_ENC_H264_PROFILE_STEREO_GUID,   _T("Stereo"),  128 }
};

const guid_desc h265_profile_names[] = {
    { NV_ENC_HEVC_PROFILE_MAIN_GUID, _T("Main"), NV_ENC_TIER_HEVC_MAIN },
    //{ NV_ENC_HEVC_PROFILE_HIGH_GUID, _T("High"), NV_ENC_TIER_HEVC_HIGH },
};

const guid_desc preset_names[] = {
    { NV_ENC_PRESET_DEFAULT_GUID,              _T("Default Preset"),                           0 },
    { NV_ENC_PRESET_LOW_LATENCY_DEFAULT_GUID,  _T("Low Latancy Default Preset"),               1 },
    { NV_ENC_PRESET_HP_GUID,                   _T("High Performance (HP) Preset"),             2 },
    { NV_ENC_PRESET_HQ_GUID,                   _T("High Quality (HQ) Preset"),                 3 },
    { NV_ENC_PRESET_BD_GUID,                   _T("Blue Ray Preset"),                          4 },
    { NV_ENC_PRESET_LOW_LATENCY_HQ_GUID,       _T("Low Latancy High Quality (HQ) Preset"),     5 },
    { NV_ENC_PRESET_LOW_LATENCY_HP_GUID,       _T("Low Latancy High Performance (HP) Preset"), 6 }
};

const guid_desc list_nvenc_codecs[] = {
	{ NV_ENC_CODEC_H264_GUID, _T("H.264/AVC"),  NV_ENC_H264 },
	{ NV_ENC_CODEC_HEVC_GUID, _T("H.265/HEVC"), NV_ENC_HEVC },
};
const CX_DESC list_nvenc_codecs_for_opt[] = {
	{ _T("h264"), NV_ENC_H264 },
	{ _T("avc"),  NV_ENC_H264 },
	{ _T("hevc"), NV_ENC_HEVC },
	{ _T("h265"), NV_ENC_HEVC },
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

const CX_DESC list_hevc_level[] = { 
	{ _T("auto"), 0   },
	{ _T("1"),    NV_ENC_LEVEL_HEVC_1   },
	{ _T("2"),    NV_ENC_LEVEL_HEVC_2   },
	{ _T("2.1"),  NV_ENC_LEVEL_HEVC_21  },
	{ _T("3"),    NV_ENC_LEVEL_HEVC_3   },
	{ _T("3.1"),  NV_ENC_LEVEL_HEVC_31  },
	{ _T("4"),    NV_ENC_LEVEL_HEVC_4   },
	{ _T("4.1"),  NV_ENC_LEVEL_HEVC_41  },
	{ _T("5"),    NV_ENC_LEVEL_HEVC_5   },
	{ _T("5.1"),  NV_ENC_LEVEL_HEVC_51  },
	{ _T("5.2"),  NV_ENC_LEVEL_HEVC_52  },
	{ _T("6"),    NV_ENC_LEVEL_HEVC_6   },
	{ _T("6.1"),  NV_ENC_LEVEL_HEVC_61  },
	{ _T("6.2"),  NV_ENC_LEVEL_HEVC_62  },
	{ NULL, NULL }
};

const CX_DESC list_hevc_cu_size[] = { 
	{ _T("auto"), NV_ENC_HEVC_CUSIZE_AUTOSELECT },
	{ _T("8"),    NV_ENC_HEVC_CUSIZE_8x8        },
	{ _T("16"),   NV_ENC_HEVC_CUSIZE_16x16      },
	{ _T("32"),   NV_ENC_HEVC_CUSIZE_32x32      },
	{ _T("64"),   NV_ENC_HEVC_CUSIZE_64x64      },
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

const CX_DESC nvenc_interface_names[] = {
    { _T("CUDA"),      NV_ENC_CUDA },
    { _T("DirectX9"),  NV_ENC_DX9  },
    { _T("DirectX10"), NV_ENC_DX10 },
    { _T("DirectX11"), NV_ENC_DX11 },
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
	//{ _T("VBR_MINQP - 下限QP付き可変ビットレート"), NV_ENC_PARAMS_RC_VBR_MINQP },
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
#if 0
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
#endif

template<size_t count>
static const TCHAR *get_name_from_guid(GUID guid, const guid_desc (&desc)[count]) {
	for (int i = 0; i < count; i++) {
		if (0 == memcmp(&desc[i].id, &guid, sizeof(GUID))) {
			return desc[i].desc;
		}
	}
	return _T("Unknown");
};

template<size_t count>
static int get_value_from_guid(GUID guid, const guid_desc (&desc)[count]) {
	for (int i = 0; i < count; i++) {
		if (0 == memcmp(&desc[i].id, &guid, sizeof(GUID))) {
			return desc[i].value;
		}
	}
	return 0;
};

template<size_t count>
static GUID get_guid_from_value(int value, const guid_desc (&desc)[count]) {
	for (int i = 0; i < count; i++) {
		if (desc[i].value == (uint32_t)value) {
			return desc[i].id;
		}
	}
	return GUID{ 0 };
};

template<size_t count>
static GUID get_guid_from_name(const TCHAR *name, const guid_desc (&desc)[count]) {
	for (int i = 0; i < count; i++) {
		if (0 == _tcsicmp(name, desc[i].desc)) {
			return desc[i].id;
		}
	}
	return GUID{ 0 };
};

template<size_t count>
static int get_value_from_name(const TCHAR *name, const guid_desc (&desc)[count]) {
	for (int i = 0; i < count; i++) {
		if (0 == _tcsicmp(name, desc[i].desc)) {
			return desc[i].value;
		}
	}
	return -1;
};

template<size_t count>
static int get_index_from_value(int value, const guid_desc (&desc)[count]) {
	for (int i = 0; i < count; i++) {
		if (desc[i].value == (uint32_t)value) {
			return i;
		}
	}
	return -1;
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

static int get_cx_index(const CX_DESC * list, const TCHAR *chr) {
	for (int i = 0; list[i].desc; i++)
		if (0 == _tcscmp(list[i].desc, chr))
			return i;
	return 0;
}

static int get_cx_value(const CX_DESC * list, const TCHAR *chr) {
	for (int i = 0; list[i].desc; i++)
		if (0 == _tcscmp(list[i].desc, chr))
			return list[i].value;
	return 0;
}

static int PARSE_ERROR_FLAG = INT_MIN;
static int get_value_from_chr(const CX_DESC *list, const TCHAR *chr) {
	for (int i = 0; list[i].desc; i++)
		if (_tcsicmp(list[i].desc, chr) == 0)
			return list[i].value;
	return PARSE_ERROR_FLAG;
}

class NVEncParam : public NVEncCore
{
public:
	NVEncParam();
	~NVEncParam();

	//featureリストの作成を開始 (非同期)
	int createCacheAsync(int deviceID);

	//featureリストを取得 (取得できるまで待機)
	const std::vector<NVEncCodecFeature>& GetCachedNVEncCapability();
	
	//featureリストからHEVCのリストを取得 (HEVC非対応ならnullptr)
	static const NVEncCodecFeature *GetHEVCFeatures(const std::vector<NVEncCodecFeature>& codecFeatures);
	//featureリストからHEVCのリストを取得 (H.264対応ならnullptr)
	static const NVEncCodecFeature *GetH264Features(const std::vector<NVEncCodecFeature>& codecFeatures);

	//H.264が使用可能かどうかを取得 (取得できるまで待機)
	bool H264Available();
	//HEVCが使用可能かどうかを取得 (取得できるまで待機)
	bool HEVCAvailable();
protected:
	//createCacheを非同期実行するスレッド用
	static unsigned int __stdcall createCacheLoader(void *prm);
	//featureの取得を実行
	int createCache(int deviceID);

	int m_nTargetDeviceID;   //対象デバイスID
	NVEncCore *m_pNVEncCore; //NVEncCoreのインスタンス (スレッド終了時にdelete)

	HANDLE m_hThCreateCache;      //featureリスト作成用スレッドのハンドル
	HANDLE m_hEvCreateCache;      //featureリストの作成終了のイベント (ManualReset)
	HANDLE m_hEvCreateCodecCache; //codecのみのリスト作成終了のイベント (ManualReset)
	bool m_bH264; //H.264が使用可能かどうか (m_hEvCreateCodecCache後に有効)
	bool m_bHEVC; //HEVCが使用可能かどうか (m_hEvCreateCodecCache後に有効)

	//featureリスト
	//コーデックの有無はm_hEvCreateCodecCache後に有効
	//フルリストはm_hEvCreateCodecCache後に有効
	std::vector<NVEncCodecFeature> m_EncodeFeatures;
};
