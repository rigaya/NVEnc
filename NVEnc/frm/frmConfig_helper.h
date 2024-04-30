// -----------------------------------------------------------------------------------------
// x264guiEx/x265guiEx/svtAV1guiEx/ffmpegOut/QSVEnc/NVEnc/VCEEnc by rigaya
// -----------------------------------------------------------------------------------------
// The MIT License
//
// Copyright (c) 2010-2022 rigaya
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.
//
// --------------------------------------------------------------------------------------------

#pragma once

using namespace System;
using namespace System::Data;
using namespace System::Threading;
using namespace System::IO;
using namespace System::Collections::Generic;

#define HIDE_MPEG2

namespace NVEnc {

    ref class LocalSettings
    {
    public:
        String^ vidEncName;
        String^ vidEncPath;
        List<String^>^ audEncName;
        List<String^>^ audEncExeName;
        List<String^>^ audEncPath;
        String^ MP4MuxerExeName;
        String^ MP4MuxerPath;
        String^ MKVMuxerExeName;
        String^ MKVMuxerPath;
        String^ TC2MP4ExeName;
        String^ TC2MP4Path;
        String^ MPGMuxerExeName;
        String^ MPGMuxerPath;
        String^ MP4RawExeName;
        String^ MP4RawPath;
        String^ CustomTmpDir;
        String^ CustomAudTmpDir;
        String^ CustomMP4TmpDir;
        String^ LastAppDir;
        String^ LastBatDir;

        LocalSettings() {
            audEncName = gcnew List<String^>();
            audEncExeName = gcnew List<String^>();
            audEncPath = gcnew List<String^>();
        }
        ~LocalSettings() {
            delete audEncName;
            delete audEncExeName;
            delete audEncPath;
        }
    };

    value struct ExeControls
    {
        String^ Name;
        String^ Path;
        const char* args;
    };

    value struct TrackBarNU {
        TrackBar^ TB;
        NumericUpDown^ NU;
    };

    value struct VidEncInfo {
        bool hwencAvail;
        bool h264Enc;
        bool hevcEnc;
        bool av1Enc;
        List<String^>^ devices;
    };
    /*
    ref class NVEncParamCache
    {
    public:
        DataTable^ dataTableNVEncFeatures;
        std::vector<NV_ENC_CONFIG> presetConfigs;
        std::vector<NVEncCap> nvencCapabilities;
        Thread^ thPresetConfig;
        Thread^ thFeatures;
        bool presetConfigCacheAvailable;
        bool featuresCacheAvaialble;

        NVEncParamCache() {
            dataTableNVEncFeatures = gcnew DataTable();
            dataTableNVEncFeatures->Columns->Add(L"機能");
            dataTableNVEncFeatures->Columns->Add(L"サポート");

            presetConfigCacheAvailable = false;
            featuresCacheAvaialble = false;

            thPresetConfig = gcnew Thread(gcnew ThreadStart(this, &NVEncParamCache::createPresetConfigCache));
            thFeatures = gcnew Thread(gcnew ThreadStart(this, &NVEncParamCache::createFeaturesCache));
            thPresetConfig->Start();
            thFeatures->Start();
        }
        ~NVEncParamCache() {
            delete dataTableNVEncFeatures;
            delete thFeatures;
            delete thPresetConfig;
        }

        void createPresetConfigCache() {
            NVEncParam param;
            presetConfig = param.GetNVEncH264Preset(0);
        }

        void createFeaturesCache() {
            NVEncParam param;
            auto nvencCapabilities = param.GetNVEncCapability(0);
            for (auto cap : nvencCapabilities) {
                DataRow^ drb = dataTableQsvFeatures->NewRow();
                drb[0] = String(cap.name).ToString();
                drb[1] = String(cap.value).ToString();
                dataTableQsvFeatures->Rows->Add(drb);
            }
        }

        std::vector<NV_ENC_CONFIG> getPresetConfigCache() {
            if (!presetConfigCacheAvailable) {
                thPresetConfig->Join();
                presetConfigCacheAvailable = true;
            }
            return presetConfig;
        }

        std::vector<NV_ENC_CONFIG> getFeaturesCache() {
            if (!featuresCacheAvaialble) {
                thFeatures->Join();
                featuresCacheAvaialble = true;
            }
            return nvencCapabilities;
        }

        DataTable^ getFeatureTable() {
            return dataTableNVEncFeatures;
        }
    };*/
};

static const WCHAR *use_default_exe_path = L"exe_files内の実行ファイルを自動選択";

const int fcgTBQualityTimerLatency = 600;
const int fcgTBQualityTimerPeriod = 40;
const int fcgTXCmdfulloffset = 57;
const int fcgCXAudioEncModeSmallWidth = 189;
const int fcgCXAudioEncModeLargeWidth = 237;

static const ENC_OPTION_STR aspect_desc[] = {
    { NULL, AUO_CONFIG_CX_ASPECT_SAR, L"SAR比を指定 (デフォルト)" },
    { NULL, AUO_CONFIG_CX_ASPECT_DAR, L"画面比から自動計算"       },
    { NULL, AUO_MES_UNKNOWN, NULL }
};

static const ENC_OPTION_STR tempdir_desc[] = {
    { NULL, AUO_CONFIG_CX_TEMPDIR_OUTDIR, L"出力先と同じフォルダ (デフォルト)" },
    { NULL, AUO_CONFIG_CX_TEMPDIR_SYSTEM, L"システムの一時フォルダ"            },
    { NULL, AUO_CONFIG_CX_TEMPDIR_CUSTOM, L"カスタム"                          },
    { NULL, AUO_MES_UNKNOWN, NULL }
};

static const ENC_OPTION_STR audtempdir_desc[] = {
    { NULL, AUO_CONFIG_CX_AUDTEMP_DEFAULT, L"変更しない" },
    { NULL, AUO_CONFIG_CX_AUDTEMP_CUSTOM,  L"カスタム"   },
    { NULL, AUO_MES_UNKNOWN, NULL }
};

static const ENC_OPTION_STR mp4boxtempdir_desc[] = {
    { NULL, AUO_CONFIG_CX_MP4BOXTEMP_DEFAULT, L"指定しない" },
    { NULL, AUO_CONFIG_CX_MP4BOXTEMP_CUSTOM,  L"カスタム"   },
    { NULL, AUO_MES_UNKNOWN, NULL }
};

#if ENCODER_QSV
static const ENC_OPTION_STR2 list_interlaced_mfx_gui[] = {
    { AUO_CONFIG_CX_INTERLACE_PROGRESSIVE, L"progressive",     MFX_PICSTRUCT_PROGRESSIVE },
    { AUO_CONFIG_CX_INTERLACE_TFF,         L"interlaced(tff)", MFX_PICSTRUCT_FIELD_TFF   },
    { AUO_CONFIG_CX_INTERLACE_BFF,         L"interlaced(bff)", MFX_PICSTRUCT_FIELD_BFF   },
    { AUO_MES_UNKNOWN, NULL, 0 }
};
#endif

static const ENC_OPTION_STR audio_enc_timing_desc[] = {
    { NULL, AUO_CONFIG_CX_AUD_ENC_ORDER_AFTER,    L"後"   },
    { NULL, AUO_CONFIG_CX_AUD_ENC_ORDER_BEFORE,   L"前"   },
    { NULL, AUO_CONFIG_CX_AUD_ENC_ORDER_PARALLEL, L"同時" },
    { NULL, AUO_MES_UNKNOWN, NULL }
};

static const ENC_OPTION_STR2 list_deinterlace_gui[] = {
    { AUO_CONFIG_CX_DEINTERLACE_NONE,   L"なし",                   0 },
#if ENCODER_QSV
    { AUO_CONFIG_CX_DEINTERLACE_NORMAL, L"インタレ解除 (通常)",     MFX_DEINTERLACE_NORMAL      },
    { AUO_CONFIG_CX_DEINTERLACE_IT,     L"インタレ解除 (24fps化)",  MFX_DEINTERLACE_IT          },
    { AUO_CONFIG_CX_DEINTERLACE_BOB,    L"インタレ解除 (Bob化)",    MFX_DEINTERLACE_BOB         },
#elif ENCODER_NVENC
#elif ENCODER_VCEENC
#else
    static_assert(false);
#endif
    { AUO_CONFIG_CX_DEINTERLACE_AFS,    L"自動フィールドシフト",    100 },
    { AUO_CONFIG_CX_DEINTERLACE_NNEDI,  L"nnedi",                   101 },
    { AUO_CONFIG_CX_DEINTERLACE_YADIF,  L"yadif",                   102 },
    { AUO_MES_UNKNOWN, NULL, NULL }
};

static const ENC_OPTION_STR list_vpp_afs_analyze[] = {
    { NULL, AUO_CONFIG_CX_AFS_ANALYZE0, L"0 - 解除なし"         },
    { NULL, AUO_CONFIG_CX_AFS_ANALYZE1, L"1 - フィールド三重化" },
    { NULL, AUO_CONFIG_CX_AFS_ANALYZE2, L"2 - 縞検出二重化"     },
    { NULL, AUO_CONFIG_CX_AFS_ANALYZE3, L"3 - 動き検出二重化"   },
    { NULL, AUO_CONFIG_CX_AFS_ANALYZE4, L"4 - 動き検出補間"     },
    { NULL, AUO_MES_UNKNOWN, NULL},
};

static const ENC_OPTION_STR2 list_vpp_nnedi_pre_screen_gui[] = {
    { AUO_CONFIG_CX_NNEDI_PRESCREEN_NONE,           L"none",           VPP_NNEDI_PRE_SCREEN_NONE           },
    { AUO_CONFIG_CX_NNEDI_PRESCREEN_ORIGINAL,       L"original",       VPP_NNEDI_PRE_SCREEN_ORIGINAL       },
    { AUO_CONFIG_CX_NNEDI_PRESCREEN_NEW,            L"new",            VPP_NNEDI_PRE_SCREEN_NEW            },
    { AUO_CONFIG_CX_NNEDI_PRESCREEN_ORIGINAL_BLOCK, L"original_block", VPP_NNEDI_PRE_SCREEN_ORIGINAL_BLOCK },
    { AUO_CONFIG_CX_NNEDI_PRESCREEN_NEW_BLOCK,      L"new_block",      VPP_NNEDI_PRE_SCREEN_NEW_BLOCK      },
    { AUO_MES_UNKNOWN, NULL, NULL }
};

static const ENC_OPTION_STR2 list_vpp_yadif_mode_gui[] = {
    { AUO_CONFIG_CX_YADIF_MODE_NORMAL, L"normal",        VPP_YADIF_MODE_AUTO     },
    { AUO_CONFIG_CX_YADIF_MODE_BOB,    L"bob",           VPP_YADIF_MODE_BOB_AUTO },
    { AUO_MES_UNKNOWN, NULL, NULL }
};

static const ENC_OPTION_STR2 list_vpp_denoise_dct_step_gui[] = {
    { AUO_CONFIG_CX_DENOISE_DCT_STEP_1, L"1 - 高品質",  1 },
    { AUO_CONFIG_CX_DENOISE_DCT_STEP_2, L"2",           2 },
    { AUO_CONFIG_CX_DENOISE_DCT_STEP_4, L"4",           4 },
    { AUO_CONFIG_CX_DENOISE_DCT_STEP_8, L"8 - 高速",    8 },
    { AUO_MES_UNKNOWN, NULL, NULL }
};
static_assert(_countof(list_vpp_denoise_dct_step) == _countof(list_vpp_denoise_dct_step_gui), "list_vpp_denoise_dct_step size");

static const ENC_OPTION_STR2 list_mv_presicion_ja[] = {
    { AUO_CONFIG_CX_MV_PREC_AUTO,   L"自動",        NV_ENC_MV_PRECISION_DEFAULT     },
    { AUO_CONFIG_CX_MV_PREC_FULL,   L"1画素精度",   NV_ENC_MV_PRECISION_FULL_PEL    },
    { AUO_CONFIG_CX_MV_PREC_HALF,   L"1/2画素精度", NV_ENC_MV_PRECISION_HALF_PEL    },
    { AUO_CONFIG_CX_MV_PREC_QUATER, L"1/4画素精度", NV_ENC_MV_PRECISION_QUARTER_PEL },
    { AUO_MES_UNKNOWN, NULL, 0 }
};

static const ENC_OPTION_STR2 list_encmode[] = {
#if ENCODER_QSV
    { AUO_CONFIG_CX_RC_CBR,    L"ビットレート指定 - CBR",           MFX_RATECONTROL_CBR    },
    { AUO_CONFIG_CX_RC_VBR,    L"ビットレート指定 - VBR",           MFX_RATECONTROL_VBR    },
    { AUO_CONFIG_CX_RC_AVBR,   L"ビットレート指定 - AVBR",          MFX_RATECONTROL_AVBR   },
    { AUO_CONFIG_CX_RC_QVBR,   L"ビットレート指定 - QVBR",          MFX_RATECONTROL_QVBR   },
    { AUO_CONFIG_CX_RC_CQP,    L"固定量子化量 (CQP)",               MFX_RATECONTROL_CQP    },
    { AUO_CONFIG_CX_RC_LA,     L"先行探索レート制御",               MFX_RATECONTROL_LA     },
    { AUO_CONFIG_CX_RC_LA_HRD, L"先行探索レート制御 (HRD準拠)",     MFX_RATECONTROL_LA_HRD },
    { AUO_CONFIG_CX_RC_ICQ,    L"固定品質モード",                   MFX_RATECONTROL_ICQ    },
    { AUO_CONFIG_CX_RC_LA_ICQ, L"先行探索付き固定品質モード",       MFX_RATECONTROL_LA_ICQ },
    { AUO_CONFIG_CX_RC_VCM,    L"ビデオ会議モード",                 MFX_RATECONTROL_VCM    },
#elif ENCODER_NVENC
    { AUO_CONFIG_CX_RC_CQP,    L"CQP - 固定量子化量",               NV_ENC_PARAMS_RC_CONSTQP   },
    { AUO_CONFIG_CX_RC_CBR,    L"CBR - 固定ビットレート",           NV_ENC_PARAMS_RC_CBR       },
    { AUO_CONFIG_CX_RC_VBR,    L"VBR - 可変ビットレート",           NV_ENC_PARAMS_RC_VBR       },
    { AUO_CONFIG_CX_RC_QVBR,   L"QVBR - 固定品質",                  NV_ENC_PARAMS_RC_QVBR      },
#elif ENCODER_VCEENC
#else
    static_assert(false);
#endif
    { AUO_MES_UNKNOWN, NULL, NULL }
};

static const ENC_OPTION_STR2 list_vpp_deband_gui[] = {
    { AUO_CONFIG_CX_DEBAND_0, L"0 - 1点参照",  0 },
    { AUO_CONFIG_CX_DEBAND_1, L"1 - 2点参照",  1 },
    { AUO_CONFIG_CX_DEBAND_2, L"2 - 4点参照",  2 },
    { AUO_MES_UNKNOWN, NULL, 0 }
};

#if ENCODER_QSV
static const ENC_OPTION_STR2 list_rotate_angle_ja[] = {
    { AUO_MES_UNKNOWN,   L"0°",  MFX_ANGLE_0    },
    { AUO_MES_UNKNOWN,  L"90°",  MFX_ANGLE_90   },
    { AUO_MES_UNKNOWN, L"180°",  MFX_ANGLE_180  },
    { AUO_MES_UNKNOWN, L"270°",  MFX_ANGLE_270  },
    { AUO_MES_UNKNOWN, NULL, 0 }
};

static const ENC_OPTION_STR2 list_out_enc_codec[] = {
    { AUO_MES_UNKNOWN, L"H.264 / AVC",  MFX_CODEC_AVC  },
    { AUO_MES_UNKNOWN, L"H.265 / HEVC", MFX_CODEC_HEVC },
#ifndef HIDE_MPEG2
    { AUO_MES_UNKNOWN, L"MPEG2", MFX_CODEC_MPEG2 },
#endif
    //{ AUO_MES_UNKNOWN,"VC-1", MFX_CODEC_VC1 },
    { AUO_MES_UNKNOWN, L"VP9", MFX_CODEC_VP9 },
    { AUO_MES_UNKNOWN, L"AV1", MFX_CODEC_AV1 },
    { AUO_MES_UNKNOWN, NULL, NULL }
};
//下記は一致していないといけない
static_assert(_countof(list_out_enc_codec)-1/*NULLの分*/ == _countof(CODEC_LIST_AUO));
#endif

static const ENC_OPTION_STR2 list_log_level_jp[] = {
    { AUO_CONFIG_CX_LOG_LEVEL_INFO,  L"通常",                  RGY_LOG_INFO  },
    { AUO_CONFIG_CX_LOG_LEVEL_MORE,  L"音声/muxのログも表示 ", RGY_LOG_MORE  },
    { AUO_CONFIG_CX_LOG_LEVEL_DEBUG, L"デバッグ用出力も表示 ", RGY_LOG_DEBUG },
    { AUO_MES_UNKNOWN, NULL, NULL }
};


//メモ表示用 RGB
const int StgNotesColor[][3] = {
    {  80,  72,  92 },
    { 120, 120, 120 }
};

const WCHAR * const DefaultStgNotes = L"メモ...";