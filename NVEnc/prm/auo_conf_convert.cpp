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

#include <string.h>
#include <stdio.h>
#include <stddef.h>
#include <iomanip>
#include <sstream>
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <Windows.h>
#include <shlwapi.h>
#pragma comment(lib, "shlwapi.lib")
#include "auo_util.h"
#include "auo_conf.h"
#include "NVEncCmd.h"

static const int NV_ENC_H264 = 0;
static const int NV_ENC_HEVC = 1;

enum {
    NPPI_INTER_MAX = NPPI_INTER_LANCZOS3_ADVANCED,
    RESIZE_CUDA_TEXTURE_BILINEAR,
    RESIZE_CUDA_TEXTURE_NEAREST,
    RESIZE_CUDA_SPLINE16,
    RESIZE_CUDA_SPLINE36,
    RESIZE_CUDA_SPLINE64,
    RESIZE_CUDA_LANCZOS2,
    RESIZE_CUDA_LANCZOS3,
    RESIZE_CUDA_LANCZOS4,
};

const CX_DESC list_nppi_resize[] = {
    { _T("default"),       NPPI_INTER_UNDEFINED },
#if !defined(_M_IX86) || FOR_AUO
    { _T("nn"),            NPPI_INTER_NN },
    { _T("npp_linear"),    NPPI_INTER_LINEAR },
    { _T("cubic"),         NPPI_INTER_CUBIC },
    { _T("cubic_bspline"), NPPI_INTER_CUBIC2P_BSPLINE },
    { _T("cubic_catmull"), NPPI_INTER_CUBIC2P_CATMULLROM },
    { _T("cubic_b05c03"),  NPPI_INTER_CUBIC2P_B05C03 },
    { _T("super"),         NPPI_INTER_SUPER },
    { _T("lanczos"),       NPPI_INTER_LANCZOS },
#endif
    //{ _T("lanczons3"),     NPPI_INTER_LANCZOS3_ADVANCED },
    { _T("bilinear"),      RESIZE_CUDA_TEXTURE_BILINEAR },
    { _T("nearest"),       RESIZE_CUDA_TEXTURE_NEAREST },
    { _T("spline16"),      RESIZE_CUDA_SPLINE16 },
    { _T("spline36"),      RESIZE_CUDA_SPLINE36 },
    { _T("spline64"),      RESIZE_CUDA_SPLINE64 },
    { _T("lanczos2"),      RESIZE_CUDA_LANCZOS2 },
    { _T("lanczos3"),      RESIZE_CUDA_LANCZOS3 },
    { _T("lanczos4"),      RESIZE_CUDA_LANCZOS4 },
    { NULL, 0 }
};

typedef struct _NV_ENC_CONFIG_SVC_TEMPORAL
{
    uint32_t         numTemporalLayers;                /**< [in]: Max temporal layers. Valid value range is [1,::NV_ENC_CAPS_NUM_MAX_TEMPORAL_LAYERS] */
    uint32_t         basePriorityID;                   /**< [in]: Priority id of the base layer. Default is 0. Priority Id is increased by 1 for each consecutive temporal layers. */
    uint32_t         reserved1[254];                   /**< [in]: Reserved and should be set to 0 */
    void*            reserved2[64];                    /**< [in]: Reserved and should be set to NULL */
} NV_ENC_CONFIG_SVC_TEMPORAL;

/**
 * MVC encoder configuration parameters
 */
typedef struct _NV_ENC_CONFIG_MVC
{
    uint32_t         reserved1[256];                   /**< [in]: Reserved and should be set to 0 */
    void*            reserved2[64];                    /**< [in]: Reserved and should be set to NULL */
} NV_ENC_CONFIG_MVC;

typedef union _NV_ENC_CONFIG_H264_EXT
{
    NV_ENC_CONFIG_SVC_TEMPORAL  svcTemporalConfig;     /**< [in]: SVC encode config*/
    NV_ENC_CONFIG_MVC           mvcConfig;             /**< [in]: MVC encode config*/
    uint32_t                    reserved1[254];        /**< [in]: Reserved and should be set to 0 */
    void*                       reserved2[64];         /**< [in]: Reserved and should be set to NULL */
} NV_ENC_CONFIG_H264_EXT;

typedef struct _NV_ENC_CONFIG_H264_OLD
{
    uint32_t enableTemporalSVC         :1;                          /**< [in]: Set to 1 to enable SVC temporal*/
    uint32_t enableStereoMVC           :1;                          /**< [in]: Set to 1 to enable stereo MVC*/
    uint32_t hierarchicalPFrames       :1;                          /**< [in]: Set to 1 to enable hierarchical PFrames */
    uint32_t hierarchicalBFrames       :1;                          /**< [in]: Set to 1 to enable hierarchical BFrames */
    uint32_t outputBufferingPeriodSEI  :1;                          /**< [in]: Set to 1 to write SEI buffering period syntax in the bitstream */
    uint32_t outputPictureTimingSEI    :1;                          /**< [in]: Set to 1 to write SEI picture timing syntax in the bitstream */
    uint32_t outputAUD                 :1;                          /**< [in]: Set to 1 to write access unit delimiter syntax in bitstream */
    uint32_t disableSPSPPS             :1;                          /**< [in]: Set to 1 to disable writing of Sequence and Picture parameter info in bitstream */
    uint32_t outputFramePackingSEI     :1;                          /**< [in]: Set to 1 to enable writing of frame packing arrangement SEI messages to bitstream */
    uint32_t outputRecoveryPointSEI    :1;                          /**< [in]: Set to 1 to enable writing of recovery point SEI message */
    uint32_t enableIntraRefresh        :1;                          /**< [in]: Set to 1 to enable gradual decoder refresh or intra refresh. If the GOP structure uses B frames this will be ignored */
    uint32_t enableConstrainedEncoding :1;                          /**< [in]: Set this to 1 to enable constrainedFrame encoding where each slice in the constarined picture is independent of other slices
                                                                               Check support for constrained encoding using ::NV_ENC_CAPS_SUPPORT_CONSTRAINED_ENCODING caps. */
    uint32_t repeatSPSPPS              :1;                          /**< [in]: Set to 1 to enable writing of Sequence and Picture parameter for every IDR frame */
    uint32_t enableVFR                 :1;                          /**< [in]: Set to 1 to enable variable frame rate. */
    uint32_t enableLTR                 :1;                          /**< [in]: Set to 1 to enable LTR support and auto-mark the first */
    uint32_t reservedBitFields         :17;                         /**< [in]: Reserved bitfields and must be set to 0 */
    uint32_t level;                                                 /**< [in]: Specifies the encoding level. Client is recommended to set this to NV_ENC_LEVEL_AUTOSELECT in order to enable the NvEncodeAPI interface to select the correct level. */
    uint32_t idrPeriod;                                             /**< [in]: Specifies the IDR interval. If not set, this is made equal to gopLength in NV_ENC_CONFIG.Low latency application client can set IDR interval to NVENC_INFINITE_GOPLENGTH so that IDR frames are not inserted automatically. */
    uint32_t separateColourPlaneFlag;                               /**< [in]: Set to 1 to enable 4:4:4 separate colour planes */
    uint32_t disableDeblockingFilterIDC;                            /**< [in]: Specifies the deblocking filter mode. Permissible value range: [0,2] */
    uint32_t numTemporalLayers;                                     /**< [in]: Specifies max temporal layers to be used for hierarchical coding. Valid value range is [1,::NV_ENC_CAPS_NUM_MAX_TEMPORAL_LAYERS] */
    uint32_t spsId;                                                 /**< [in]: Specifies the SPS id of the sequence header. Currently reserved and must be set to 0. */
    uint32_t ppsId;                                                 /**< [in]: Specifies the PPS id of the picture header. Currently reserved and must be set to 0. */
    NV_ENC_H264_ADAPTIVE_TRANSFORM_MODE adaptiveTransformMode;      /**< [in]: Specifies the AdaptiveTransform Mode. Check support for AdaptiveTransform mode using ::NV_ENC_CAPS_SUPPORT_ADAPTIVE_TRANSFORM caps. */
    NV_ENC_H264_FMO_MODE                fmoMode;                    /**< [in]: Specified the FMO Mode. Check support for FMO using ::NV_ENC_CAPS_SUPPORT_FMO caps. */
    NV_ENC_H264_BDIRECT_MODE            bdirectMode;                /**< [in]: Specifies the BDirect mode. Check support for BDirect mode using ::NV_ENC_CAPS_SUPPORT_BDIRECT_MODE caps.*/
    NV_ENC_H264_ENTROPY_CODING_MODE     entropyCodingMode;          /**< [in]: Specifies the entropy coding mode. Check support for CABAC mode using ::NV_ENC_CAPS_SUPPORT_CABAC caps. */
    NV_ENC_STEREO_PACKING_MODE          stereoMode;                 /**< [in]: Specifies the stereo frame packing mode which is to be signalled in frame packing arrangement SEI */
    NV_ENC_CONFIG_H264_EXT              h264Extension;              /**< [in]: Specifies the H264 extension config */
    uint32_t                            intraRefreshPeriod;         /**< [in]: Specifies the interval between successive intra refresh if enableIntrarefresh is set and one time intraRefresh configuration is desired.
                                                                               When this is specified only first IDR will be encoded and no more key frames will be encoded. Client should set PIC_TYPE = NV_ENC_PIC_TYPE_INTRA_REFRESH
                                                                               for first picture of every intra refresh period. */
    uint32_t                            intraRefreshCnt;            /**< [in]: Specifies the number of frames over which intra refresh will happen */
    uint32_t                            maxNumRefFrames;            /**< [in]: Specifies the DPB size used for encoding. Setting it to 0 will let driver use the default dpb size.
                                                                               The low latency application which wants to invalidate reference frame as an error resilience tool
                                                                               is recommended to use a large DPB size so that the encoder can keep old reference frames which can be used if recent
                                                                               frames are invalidated. */
    uint32_t                            sliceMode;                  /**< [in]: This parameter in conjunction with sliceModeData specifies the way in which the picture is divided into slices
                                                                               sliceMode = 0 MB based slices, sliceMode = 1 Byte based slices, sliceMode = 2 MB row based slices, sliceMode = 3, numSlices in Picture
                                                                               When forceIntraRefreshWithFrameCnt is set it will have priority over sliceMode setting
                                                                               When sliceMode == 0 and sliceModeData == 0 whole picture will be coded with one slice */
    uint32_t                            sliceModeData;              /**< [in]: Specifies the parameter needed for sliceMode. For:
                                                                               sliceMode = 0, sliceModeData specifies # of MBs in each slice (except last slice)
                                                                               sliceMode = 1, sliceModeData specifies maximum # of bytes in each slice (except last slice)
                                                                               sliceMode = 2, sliceModeData specifies # of MB rows in each slice (except last slice)
                                                                               sliceMode = 3, sliceModeData specifies number of slices in the picture. Driver will divide picture into slices optimally */
    NV_ENC_CONFIG_H264_VUI_PARAMETERS   h264VUIParameters;          /**< [in]: Specifies the H264 video usability info pamameters */
    uint32_t                            ltrNumFrames;               /**< [in]: Specifies the number of LTR frames used. Additionally, encoder will mark the first numLTRFrames base layer reference frames within each IDR interval as LTR */
    uint32_t                            ltrTrustMode;               /**< [in]: Specifies the LTR operating mode. Set to 0 to disallow encoding using LTR frames until later specified. Set to 1 to allow encoding using LTR frames unless later invalidated.*/
    uint32_t                            reserved1[272];             /**< [in]: Reserved and must be set to 0 */
    void*                               reserved2[64];              /**< [in]: Reserved and must be set to NULL */
} NV_ENC_CONFIG_H264_OLD;


typedef struct CONF_NVENC_OLD {
    NV_ENC_CONFIG enc_config;
    NV_ENC_PIC_STRUCT pic_struct;
    int preset;
    int deviceID;
    int inputBuffer;
    int par[2];
} CONF_NVENC_OLD;



typedef struct CONF_NVENC_OLD3 {
    NV_ENC_CONFIG enc_config;
    NV_ENC_PIC_STRUCT pic_struct;
    int preset;
    int deviceID;
    int inputBuffer;
    int par[2];
    int codec;
    NV_ENC_CODEC_CONFIG codecConfig[2];
    int bluray;
    int weightp;
    int perf_monitor;
    int cuda_schedule;
} CONF_NVENC_OLD3;

typedef struct {
    BOOL vpp_perf_monitor;
    BOOL resize_enable;
    int resize_interp;
    int resize_width;
    int resize_height;
    VppKnn knn;
    VppPmd pmd;
    VppDeband deband;
    VppAfs afs;
    VppUnsharp unsharp;
    VppEdgelevel edgelevel;
    VppTweak tweak;
} CONF_VPP_OLD3;

typedef struct {
    BOOL afs;                      //自動フィールドシフトの使用
    BOOL auo_tcfile_out;           //auo側でタイムコードを出力する
    BOOL log_debug;
} CONF_VIDEO_OLD3; //動画用設定

typedef struct {
    int  encoder;             //使用する音声エンコーダ
    int  enc_mode;            //使用する音声エンコーダの設定
    int  bitrate;             //ビットレート指定モード
    BOOL use_2pass;           //音声2passエンコードを行う
    BOOL use_wav;             //パイプを使用せず、wavを出力してエンコードを行う
    BOOL faw_check;           //FAWCheckを行う
    int  priority;            //音声エンコーダのCPU優先度(インデックス)
    BOOL minimized;           //音声エンコーダを最小化で実行
    int  aud_temp_dir;        //音声専用一時フォルダ
    int  audio_encode_timing; //音声を先にエンコード
    int  delay_cut;           //エンコード遅延の削除
} CONF_AUDIO_OLD3; //音声用設定

typedef struct {
    BOOL disable_mp4ext;  //mp4出力時、外部muxerを使用する
    BOOL disable_mkvext;  //mkv出力時、外部muxerを使用する
    int  mp4_mode;        //mp4 外部muxer用追加コマンドの設定
    int  mkv_mode;        //mkv 外部muxer用追加コマンドの設定
    BOOL minimized;       //muxを最小化で実行
    int  priority;        //mux優先度(インデックス)
    int  mp4_temp_dir;    //mp4box用一時ディレクトリ
    BOOL apple_mode;      //Apple用モード(mp4系専用)
    BOOL disable_mpgext;  //mpg出力時、外部muxerを使用する
    int  mpg_mode;        //mpg 外部muxer用追加コマンドの設定
} CONF_MUX_OLD3; //muxer用設定

typedef struct {
    //BOOL disable_guicmd;         //GUIによるコマンドライン生成を停止(CLIモード)
    int  temp_dir;               //一時ディレクトリ
    BOOL out_audio_only;         //音声のみ出力
    char notes[128];             //メモ
    DWORD run_bat;                //バッチファイルを実行するかどうか
    DWORD dont_wait_bat_fin;      //バッチファイルの処理終了待機をするかどうか
    union {
        char batfiles[4][512];        //バッチファイルのパス
        struct {
            char before_process[512]; //エンコ前バッチファイルのパス
            char after_process[512];  //エンコ後バッチファイルのパス
            char before_audio[512];   //音声エンコ前バッチファイルのパス
            char after_audio[512];    //音声エンコ後バッチファイルのパス
        } batfile;
    };
} CONF_OTHER_OLD3;

typedef struct {
    char        conf_name[CONF_NAME_BLOCK_LEN];  //保存時に使用
    int         size_all;                        //保存時: CONF_VCEOUTの全サイズ / 設定中、エンコ中: CONF_INITIALIZED
    int         head_size;                       //ヘッダ部分の全サイズ
    int         block_count;                     //ヘッダ部を除いた設定のブロック数
    int         block_size[CONF_BLOCK_MAX];      //各ブロックのサイズ
    size_t      block_head_p[CONF_BLOCK_MAX];    //各ブロックのポインタ位置
    CONF_NVENC_OLD3 nvenc;                       //nvencについての設定
    CONF_VIDEO_OLD3 vid;                         //その他動画についての設定
    CONF_AUDIO_OLD3 aud;                         //音声についての設定
    CONF_MUX_OLD3   mux;                         //muxについての設定
    CONF_OTHER_OLD3 oth;                         //その他の設定
    CONF_VPP_OLD3   vpp;                         //vppについての設定
} CONF_GUIEX_OLD3;

const int conf_block_data_old[] = {
    sizeof(CONF_NVENC_OLD3),
    sizeof(CONF_VIDEO_OLD3),
    sizeof(CONF_AUDIO_OLD3),
    sizeof(CONF_MUX_OLD3),
    sizeof(CONF_OTHER_OLD3),
    sizeof(CONF_VPP_OLD3)
};

const size_t conf_block_pointer_old[] = {
    offsetof(CONF_GUIEX_OLD3, nvenc),
    offsetof(CONF_GUIEX_OLD3, vid),
    offsetof(CONF_GUIEX_OLD3, aud),
    offsetof(CONF_GUIEX_OLD3, mux),
    offsetof(CONF_GUIEX_OLD3, oth),
    offsetof(CONF_GUIEX_OLD3, vpp)
};

void write_conf_header_old(CONF_GUIEX_OLD3 *save_conf) {
    sprintf_s(save_conf->conf_name, sizeof(save_conf->conf_name), CONF_NAME);
    save_conf->size_all = sizeof(CONF_GUIEX_OLD3);
    save_conf->head_size = CONF_HEAD_SIZE;
    save_conf->block_count = _countof(conf_block_data_old);
    for (int i = 0; i < _countof(conf_block_data_old); ++i) {
        save_conf->block_size[i] = conf_block_data_old[i];
        save_conf->block_head_p[i] = conf_block_pointer_old[i];
    }
}

void init_CONF_GUIEX_old(CONF_GUIEX_OLD3 *conf) {
    ZeroMemory(conf, sizeof(CONF_GUIEX_OLD3));
    write_conf_header_old(conf);
    conf->nvenc.deviceID = -1;
    conf->nvenc.enc_config = DefaultParam();
    conf->nvenc.codecConfig[NV_ENC_H264] = DefaultParamH264();
    conf->nvenc.codecConfig[NV_ENC_HEVC] = DefaultParamHEVC();
    conf->nvenc.pic_struct = NV_ENC_PIC_STRUCT_FRAME;
    conf->nvenc.preset = NVENC_PRESET_DEFAULT;
    conf->nvenc.enc_config.rcParams.maxBitRate = DEFAULT_MAX_BITRATE; //NVEnc.auoではデフォルト値としてセットする
    conf->nvenc.cuda_schedule = DEFAULT_CUDA_SCHEDULE;
    conf->size_all = CONF_INITIALIZED;
    conf->vpp.resize_width = 1280;
    conf->vpp.resize_height = 720;
    conf->vpp.resize_interp = RESIZE_CUDA_SPLINE36;
    conf->vpp.knn = VppKnn();
    conf->vpp.pmd = VppPmd();
    conf->vpp.deband = VppDeband();
    conf->vpp.afs = VppAfs();
    conf->vpp.unsharp = VppUnsharp();
    conf->vpp.edgelevel = VppEdgelevel();
    conf->vpp.tweak = VppTweak();
}

int guiEx_config::stgv3_block_size() {
    return 7;
}

void guiEx_config::convert_nvencstg_to_nvencstgv4(CONF_GUIEX_OLD *conf, const void *dat) {
    CONF_GUIEX_OLD3 confv3;
    const CONF_GUIEX_OLD3 *old_data = (const CONF_GUIEX_OLD3 *)dat;
    init_CONF_GUIEX_old(&confv3);

    //まずそのままコピーするブロックはそうする
#define COPY_BLOCK(block, block_idx) { memcpy(&confv3.block, ((BYTE *)old_data) + old_data->block_head_p[block_idx], std::min((int)sizeof(confv3.block), old_data->block_size[block_idx])); }
    COPY_BLOCK(nvenc, 0);
    COPY_BLOCK(vid, 1);
    COPY_BLOCK(aud, 2);
    COPY_BLOCK(mux, 3);
    COPY_BLOCK(oth, 4);
#undef COPY_BLOCK

    CONF_NVENC_OLD *old = (CONF_NVENC_OLD *)(((BYTE *)old_data) + old_data->block_head_p[0]);
    NV_ENC_CONFIG_H264_OLD * h264old = (NV_ENC_CONFIG_H264_OLD *)&old->enc_config.encodeCodecConfig.h264Config;

    memset(&confv3.nvenc.codecConfig[NV_ENC_H264].h264Config, 0, sizeof(confv3.nvenc.codecConfig[NV_ENC_H264].h264Config));
#define COPY_H264_STG(name) { confv3.nvenc.codecConfig[NV_ENC_H264].h264Config.name = h264old->name; }
    //COPY_H264_STG(enableTemporalSVC);
    COPY_H264_STG(enableStereoMVC);
    COPY_H264_STG(hierarchicalPFrames);
    COPY_H264_STG(hierarchicalBFrames);
    COPY_H264_STG(outputBufferingPeriodSEI);
    COPY_H264_STG(outputPictureTimingSEI);
    COPY_H264_STG(outputAUD);
    COPY_H264_STG(disableSPSPPS);
    COPY_H264_STG(outputFramePackingSEI);
    COPY_H264_STG(outputRecoveryPointSEI);
    COPY_H264_STG(enableIntraRefresh);
    COPY_H264_STG(enableConstrainedEncoding);
    COPY_H264_STG(repeatSPSPPS);
    COPY_H264_STG(enableVFR);
    COPY_H264_STG(enableLTR);
    COPY_H264_STG(reservedBitFields);
    COPY_H264_STG(level);
    COPY_H264_STG(idrPeriod);
    COPY_H264_STG(separateColourPlaneFlag);
    COPY_H264_STG(disableDeblockingFilterIDC);
    COPY_H264_STG(numTemporalLayers);
    COPY_H264_STG(spsId);
    COPY_H264_STG(ppsId);
    COPY_H264_STG(adaptiveTransformMode);
    COPY_H264_STG(fmoMode);
    COPY_H264_STG(bdirectMode);
    COPY_H264_STG(entropyCodingMode);
    COPY_H264_STG(stereoMode);
    COPY_H264_STG(intraRefreshPeriod);
    COPY_H264_STG(intraRefreshCnt);
    COPY_H264_STG(maxNumRefFrames);
    COPY_H264_STG(sliceMode);
    COPY_H264_STG(h264VUIParameters);
    COPY_H264_STG(ltrNumFrames);
    COPY_H264_STG(ltrTrustMode);

    convert_nvencstgv2_to_nvencstgv3(&confv3);
    convert_nvencstgv3_to_nvencstgv4(conf, &confv3);
}

void guiEx_config::convert_nvencstgv2_to_nvencstgv3(void *_conf) {
    static const DWORD OLD_FLAG_AFTER  = 0x01;
    static const DWORD OLD_FLAG_BEFORE = 0x02;
    CONF_GUIEX_OLD3 *conf = (CONF_GUIEX_OLD3 *)_conf;

    char bat_path_before_process[1024];
    char bat_path_after_process[1024];
    strcpy_s(bat_path_after_process,  conf->oth.batfiles[0]);
    strcpy_s(bat_path_before_process, conf->oth.batfiles[2]);

    DWORD old_run_bat_flags = conf->oth.run_bat;
    conf->oth.run_bat  = 0x00;
    conf->oth.run_bat |= (old_run_bat_flags & OLD_FLAG_BEFORE) ? RUN_BAT_BEFORE_PROCESS : 0x00;
    conf->oth.run_bat |= (old_run_bat_flags & OLD_FLAG_AFTER)  ? RUN_BAT_AFTER_PROCESS  : 0x00;

    memset(&conf->oth.batfiles[0], 0, sizeof(conf->oth.batfiles));
    strcpy_s(conf->oth.batfile.before_process, bat_path_before_process);
    strcpy_s(conf->oth.batfile.after_process,  bat_path_after_process);
    strcpy_s(conf->conf_name, CONF_NAME_OLD_3);
}

void guiEx_config::convert_nvencstgv2_to_nvencstgv4(CONF_GUIEX_OLD *conf, const void *dat) {
    CONF_GUIEX_OLD3 confv3 = { 0 };
    memcpy(&confv3, dat, sizeof(confv3));
    convert_nvencstgv2_to_nvencstgv3(&confv3);
    convert_nvencstgv3_to_nvencstgv4(conf, &confv3);
}

#pragma warning (push)
#pragma warning (disable: 4127)
#pragma warning (disable: 4063) //warning C4063 : case '16' は '_NV_ENC_PARAMS_RC_MODE' の switch の値として正しくありません。
tstring gen_cmd_old3(const CONF_GUIEX_OLD3 *conf) {
    std::basic_stringstream<TCHAR> cmd;

#define OPT_FLOAT(str, opt, prec)  cmd << _T(" ") << (str) << _T(" ") << std::setprecision(prec) << (opt);
#define OPT_NUM(str, opt)  cmd << _T(" ") << (str) << _T(" ") << (int)(opt);
#define OPT_NUM_HEVC(str, codec, opt) cmd << _T(" ") << (str) << (codec) << _T(" ") << (int)(opt);
#define OPT_NUM_H264(str, codec, opt) cmd << _T(" ") << (str) << (codec) << _T(" ") << (int)(opt);
#define OPT_GUID(str, opt, list)  cmd << _T(" ") << (str) << _T(" ") << get_name_from_guid((opt), list);
#define OPT_GUID_HEVC(str, codec, opt, list) cmd << _T(" ") << (str) << (codec) << _T(" ") << get_name_from_value((opt), list);
#define OPT_LST(str, opt, list)  cmd << _T(" ") << (str) << _T(" ") << get_chr_from_value(list, (opt));
#define OPT_LST_HEVC(str, codec, opt, list) cmd << _T(" ") << (str) << (codec) << _T(" ") << get_chr_from_value(list, (opt));
#define OPT_LST_H264(str, codec, opt, list)  cmd << _T(" ") << (str) << (codec) << _T(" ") << get_chr_from_value(list, (opt));
#define OPT_QP(str, qp, enable) { \
    if (enable) { \
        cmd << _T(" ") << (str) << _T(" "); \
    } else { \
        cmd << _T(" ") << (str) << _T(" 0;"); \
    } \
    if ((qp.qpIntra) == (qp.qpInterP) && (qp.qpIntra) == (qp.qpInterB)) { \
        cmd << (int)(qp.qpIntra); \
    } else { \
        cmd << (int)(qp.qpIntra) << _T(":") << (int)(qp.qpInterP) << _T(":") << (int)(qp.qpInterB); \
    } \
}
#define OPT_BOOL(str_true, str_false, opt)  cmd << _T(" ") << ((opt) ? (str_true) : (str_false));
#define OPT_BOOL_HEVC(str_true, str_false, codec, opt) { \
    cmd << _T(" "); \
    if (opt) { \
        if (_tcslen(str_true)) cmd << (str_true) << (codec); \
    } else { \
        if (_tcslen(str_false)) cmd << (str_false) << (codec); \
    } \
}
#define OPT_BOOL_H264(str_true, str_false, codec, opt) { \
    cmd << _T(" "); \
    if (opt) { \
        if (_tcslen(str_true)) cmd << (str_true) << (codec); \
    } else { \
        if (_tcslen(str_false)) cmd << (str_false) << (codec); \
    } \
}
#define OPT_CHAR(str, opt) if ((opt) && _tcslen(opt)) cmd << _T(" ") << str << _T(" ") << (opt);
#define OPT_STR(str, opt) if (opt.length() > 0) cmd << _T(" ") << str << _T(" ") << (opt.c_str());

    OPT_NUM(_T("-d"), conf->nvenc.deviceID);
    cmd << _T(" -c ") << get_chr_from_value(list_nvenc_codecs_for_opt, conf->nvenc.codec);
    if (conf->nvenc.pic_struct == NV_ENC_PIC_STRUCT_FIELD_TOP_BOTTOM) {
        cmd << _T(" --interlace tff");
    } else if (conf->nvenc.pic_struct == NV_ENC_PIC_STRUCT_FIELD_BOTTOM_TOP) {
        cmd << _T(" --interlace bff");
    }
    switch (conf->nvenc.enc_config.rcParams.rateControlMode) {
    case NV_ENC_PARAMS_RC_CBR:
    case NV_ENC_PARAMS_RC_CBR_HQ:
    case NV_ENC_PARAMS_RC_VBR:
    case NV_ENC_PARAMS_RC_VBR_HQ: {
        OPT_QP(_T("--cqp"), conf->nvenc.enc_config.rcParams.constQP, true);
    } break;
    case NV_ENC_PARAMS_RC_CONSTQP:
    default: {
        cmd << _T(" --vbr ") << conf->nvenc.enc_config.rcParams.averageBitRate / 1000;
    } break;
    }
    switch (conf->nvenc.enc_config.rcParams.rateControlMode) {
    case NV_ENC_PARAMS_RC_CBR: {
        cmd << _T(" --cbr ") << conf->nvenc.enc_config.rcParams.averageBitRate / 1000;
    } break;
    case NV_ENC_PARAMS_RC_CBR_HQ: {
        cmd << _T(" --cbrhq ") << conf->nvenc.enc_config.rcParams.averageBitRate / 1000;
    } break;
    case NV_ENC_PARAMS_RC_VBR: {
        cmd << _T(" --vbr ") << conf->nvenc.enc_config.rcParams.averageBitRate / 1000;
    } break;
    case NV_ENC_PARAMS_RC_VBR_HQ: {
        cmd << _T(" --vbrhq ") << conf->nvenc.enc_config.rcParams.averageBitRate / 1000;
    } break;
    case NV_ENC_PARAMS_RC_CONSTQP:
    default: {
        OPT_QP(_T("--cqp"), conf->nvenc.enc_config.rcParams.constQP, true);
    } break;
    }
    if (conf->nvenc.enc_config.rcParams.rateControlMode != NV_ENC_PARAMS_RC_CONSTQP) {
        OPT_NUM(_T("--vbv-bufsize"), conf->nvenc.enc_config.rcParams.vbvBufferSize / 1000);
        float val = conf->nvenc.enc_config.rcParams.targetQuality + conf->nvenc.enc_config.rcParams.targetQualityLSB / 256.0f;
        cmd << _T(" --vbr-quality ") << std::setprecision(2) << val;
    }
    OPT_NUM(_T("--max-bitrate"), conf->nvenc.enc_config.rcParams.maxBitRate / 1000);
    if (conf->nvenc.enc_config.rcParams.initialRCQP.qpIntra > 0
        && conf->nvenc.enc_config.rcParams.initialRCQP.qpInterP > 0
        && conf->nvenc.enc_config.rcParams.initialRCQP.qpInterB > 0) {
        OPT_QP(_T("--qp-init"), conf->nvenc.enc_config.rcParams.initialRCQP, conf->nvenc.enc_config.rcParams.enableInitialRCQP);
    }
    OPT_QP(_T("--qp-min"), conf->nvenc.enc_config.rcParams.minQP, conf->nvenc.enc_config.rcParams.enableMinQP);
    if (std::min(conf->nvenc.enc_config.rcParams.maxQP.qpIntra,
        std::min(conf->nvenc.enc_config.rcParams.maxQP.qpInterP, conf->nvenc.enc_config.rcParams.maxQP.qpInterB))
        > std::max(conf->nvenc.enc_config.rcParams.constQP.qpIntra,
            std::max(conf->nvenc.enc_config.rcParams.constQP.qpInterP, conf->nvenc.enc_config.rcParams.constQP.qpInterB))) {
        OPT_QP(_T("--qp-max"), conf->nvenc.enc_config.rcParams.maxQP, conf->nvenc.enc_config.rcParams.enableMaxQP);
    }
    OPT_NUM(_T("--lookahead"), conf->nvenc.enc_config.rcParams.lookaheadDepth);
    OPT_BOOL(_T("--no-i-adapt"), _T(""), conf->nvenc.enc_config.rcParams.disableIadapt);
    OPT_BOOL(_T("--no-b-adapt"), _T(""), conf->nvenc.enc_config.rcParams.disableBadapt);
    OPT_BOOL(_T("--strict-gop"), _T(""), conf->nvenc.enc_config.rcParams.strictGOPTarget);
    OPT_NUM(_T("--gop-len"), conf->nvenc.enc_config.gopLength);
    OPT_NUM(_T("-b"), conf->nvenc.enc_config.frameIntervalP-1);
    OPT_NUM(_T("--weightp"), conf->nvenc.weightp);
    OPT_BOOL(_T("--aq"), _T("--no-aq"), conf->nvenc.enc_config.rcParams.enableAQ);
    OPT_BOOL(_T("--aq-temporal"), _T(""), conf->nvenc.enc_config.rcParams.enableTemporalAQ);
    OPT_NUM(_T("--aq-strength"), conf->nvenc.enc_config.rcParams.aqStrength);
    OPT_LST(_T("--mv-precision"), conf->nvenc.enc_config.mvPrecision, list_mv_presicion);
    if (conf->nvenc.par[0] > 0 && conf->nvenc.par[1] > 0) {
        cmd << _T(" --sar ") << conf->nvenc.par[0] << _T(":") << conf->nvenc.par[1];
    } else if (conf->nvenc.par[0] < 0 && conf->nvenc.par[1] < 0) {
        cmd << _T(" --par ") << -1 * conf->nvenc.par[0] << _T(":") << -1 * conf->nvenc.par[1];
    }

    //hevc
    OPT_LST_HEVC(_T("--level"), _T(":hevc"), conf->nvenc.codecConfig[NV_ENC_HEVC].hevcConfig.level, list_hevc_level);
    OPT_GUID_HEVC(_T("--profile"), _T(":hevc"), conf->nvenc.codecConfig[NV_ENC_HEVC].hevcConfig.tier, h265_profile_names);
    OPT_NUM_HEVC(_T("--ref"), _T(""), conf->nvenc.codecConfig[NV_ENC_HEVC].hevcConfig.maxNumRefFramesInDPB);
    cmd << _T(" --output-depth ") << conf->nvenc.codecConfig[NV_ENC_HEVC].hevcConfig.reserved3  /*pixelBitDepthMinus8*/ + 8;
    OPT_BOOL_HEVC(_T("--fullrange"), _T(""), _T(":hevc"), conf->nvenc.codecConfig[NV_ENC_HEVC].hevcConfig.hevcVUIParameters.videoFullRangeFlag);
    OPT_LST_HEVC(_T("--videoformat"), _T(":hevc"), conf->nvenc.codecConfig[NV_ENC_HEVC].hevcConfig.hevcVUIParameters.videoFormat, list_videoformat);
    OPT_LST_HEVC(_T("--colormatrix"), _T(":hevc"), conf->nvenc.codecConfig[NV_ENC_HEVC].hevcConfig.hevcVUIParameters.colourMatrix, list_colormatrix);
    OPT_LST_HEVC(_T("--colorprim"), _T(":hevc"), conf->nvenc.codecConfig[NV_ENC_HEVC].hevcConfig.hevcVUIParameters.colourPrimaries, list_colorprim);
    OPT_LST_HEVC(_T("--transfer"), _T(":hevc"), conf->nvenc.codecConfig[NV_ENC_HEVC].hevcConfig.hevcVUIParameters.transferCharacteristics, list_transfer);
    OPT_LST_HEVC(_T("--cu-max"), _T(""), conf->nvenc.codecConfig[NV_ENC_HEVC].hevcConfig.maxCUSize, list_hevc_cu_size);
    OPT_LST_HEVC(_T("--cu-min"), _T(""), conf->nvenc.codecConfig[NV_ENC_HEVC].hevcConfig.minCUSize, list_hevc_cu_size);

    //h264
    OPT_LST_H264(_T("--level"), _T(":h264"), conf->nvenc.codecConfig[NV_ENC_H264].h264Config.level, list_avc_level);
    OPT_GUID(_T("--profile"), conf->nvenc.enc_config.profileGUID, h264_profile_names);
    OPT_NUM_H264(_T("--ref"), _T(""), conf->nvenc.codecConfig[NV_ENC_H264].h264Config.maxNumRefFrames);
    OPT_LST_H264(_T("--direct"), _T(""), conf->nvenc.codecConfig[NV_ENC_H264].h264Config.bdirectMode, list_bdirect);
    OPT_LST_H264(_T("--adapt-transform"), _T(""), conf->nvenc.codecConfig[NV_ENC_H264].h264Config.adaptiveTransformMode, list_adapt_transform);
    OPT_BOOL_H264(_T("--fullrange"), _T(""), _T(":h264"), conf->nvenc.codecConfig[NV_ENC_H264].h264Config.h264VUIParameters.videoFullRangeFlag);
    OPT_LST_H264(_T("--videoformat"), _T(":h264"), conf->nvenc.codecConfig[NV_ENC_H264].h264Config.h264VUIParameters.videoFormat, list_videoformat);
    OPT_LST_H264(_T("--colormatrix"), _T(":h264"), conf->nvenc.codecConfig[NV_ENC_H264].h264Config.h264VUIParameters.colourMatrix, list_colormatrix);
    OPT_LST_H264(_T("--colorprim"), _T(":h264"), conf->nvenc.codecConfig[NV_ENC_H264].h264Config.h264VUIParameters.colourPrimaries, list_colorprim);
    OPT_LST_H264(_T("--transfer"), _T(":h264"), conf->nvenc.codecConfig[NV_ENC_H264].h264Config.h264VUIParameters.transferCharacteristics, list_transfer);
    cmd << _T(" --") << get_chr_from_value(list_entropy_coding, conf->nvenc.codecConfig[NV_ENC_H264].h264Config.entropyCodingMode);
    OPT_BOOL(_T("--bluray"), _T(""), conf->nvenc.bluray);
    OPT_BOOL_H264(_T("--no-deblock"), _T("--deblock"), _T(""), conf->nvenc.codecConfig[NV_ENC_H264].h264Config.disableDeblockingFilterIDC);

    std::basic_stringstream<TCHAR> tmp;

    OPT_LST(_T("--vpp-resize"), conf->vpp.resize_interp, list_nppi_resize);

#define ADD_FLOAT(str, opt, prec) tmp << _T(",") << (str) << _T("=") << std::setprecision(prec) << (opt);
#define ADD_NUM(str, opt) tmp << _T(",") << (str) << _T("=") << (opt);
#define ADD_LST(str, opt, list) tmp << _T(",") << (str) << _T("=") << get_chr_from_value(list, (opt));
#define ADD_BOOL(str, opt) tmp << _T(",") << (str) << _T("=") << ((opt) ? (_T("true")) : (_T("false")));
#define ADD_CHAR(str, opt) tmp << _T(",") << (str) << _T("=") << (opt);
#define ADD_STR(str, opt) tmp << _T(",") << (str) << _T("=") << (opt.c_str());

    //afs
    tmp.str(tstring());
    if (!conf->vpp.afs.enable) {
        tmp << _T(",enable=false");
    }
    ADD_NUM(_T("top"), conf->vpp.afs.clip.top);
    ADD_NUM(_T("bottom"), conf->vpp.afs.clip.bottom);
    ADD_NUM(_T("left"), conf->vpp.afs.clip.left);
    ADD_NUM(_T("right"), conf->vpp.afs.clip.right);
    ADD_NUM(_T("method_switch"), conf->vpp.afs.method_switch);
    ADD_NUM(_T("coeff_shift"), conf->vpp.afs.coeff_shift);
    ADD_NUM(_T("thre_shift"), conf->vpp.afs.thre_shift);
    ADD_NUM(_T("thre_deint"), conf->vpp.afs.thre_deint);
    ADD_NUM(_T("thre_motion_y"), conf->vpp.afs.thre_Ymotion);
    ADD_NUM(_T("thre_motion_c"), conf->vpp.afs.thre_Cmotion);
    ADD_NUM(_T("level"), conf->vpp.afs.analyze);
    ADD_BOOL(_T("shift"), conf->vpp.afs.shift);
    ADD_BOOL(_T("drop"), conf->vpp.afs.drop);
    ADD_BOOL(_T("smooth"), conf->vpp.afs.smooth);
    ADD_BOOL(_T("24fps"), conf->vpp.afs.force24);
    ADD_BOOL(_T("tune"), conf->vpp.afs.tune);
    ADD_BOOL(_T("rff"), conf->vpp.afs.rff);
    ADD_BOOL(_T("timecode"), conf->vpp.afs.timecode);
    ADD_BOOL(_T("log"), conf->vpp.afs.log);
    if (!tmp.str().empty()) {
        cmd << _T(" --vpp-afs ") << tmp.str().substr(1);
    } else if (conf->vpp.afs.enable) {
        cmd << _T(" --vpp-afs");
    }

    //knn
    tmp.str(tstring());
    tmp << _T(",enable=false");
    ADD_NUM(_T("radius"), conf->vpp.knn.radius);
    ADD_FLOAT(_T("strength"), conf->vpp.knn.strength, 3);
    ADD_FLOAT(_T("lerp"), conf->vpp.knn.lerpC, 3);
    ADD_FLOAT(_T("th_weight"), conf->vpp.knn.weight_threshold, 3);
    ADD_FLOAT(_T("th_lerp"), conf->vpp.knn.lerp_threshold, 3);
    if (!tmp.str().empty()) {
        cmd << _T(" --vpp-knn ") << tmp.str().substr(1);
    } else if (conf->vpp.knn.enable) {
        cmd << _T(" --vpp-knn");
    }

    //pmd
    tmp.str(tstring());
    tmp << _T(",enable=false");
    ADD_NUM(_T("apply_count"), conf->vpp.pmd.applyCount);
    ADD_FLOAT(_T("strength"), conf->vpp.pmd.strength, 3);
    ADD_FLOAT(_T("threshold"), conf->vpp.pmd.threshold, 3);
    ADD_NUM(_T("useexp"), conf->vpp.pmd.useExp);
    if (!tmp.str().empty()) {
        cmd << _T(" --vpp-pmd ") << tmp.str().substr(1);
    } else if (conf->vpp.pmd.enable) {
        cmd << _T(" --vpp-pmd");
    }

    //unsharp
    tmp.str(tstring());
    if (!conf->vpp.unsharp.enable) {
        tmp << _T(",enable=false");
    }
    ADD_NUM(_T("radius"), conf->vpp.unsharp.radius);
    ADD_FLOAT(_T("weight"), conf->vpp.unsharp.weight, 3);
    ADD_FLOAT(_T("threshold"), conf->vpp.unsharp.threshold, 3);
    if (!tmp.str().empty()) {
        cmd << _T(" --vpp-unsharp ") << tmp.str().substr(1);
    } else if (conf->vpp.unsharp.enable) {
        cmd << _T(" --vpp-unsharp");
    }

    //edgelevel
    tmp.str(tstring());
    if (!conf->vpp.edgelevel.enable) {
        tmp << _T(",enable=false");
    }
    ADD_FLOAT(_T("strength"), conf->vpp.edgelevel.strength, 3);
    ADD_FLOAT(_T("threshold"), conf->vpp.edgelevel.threshold, 3);
    ADD_FLOAT(_T("black"), conf->vpp.edgelevel.black, 3);
    ADD_FLOAT(_T("white"), conf->vpp.edgelevel.white, 3);
    if (!tmp.str().empty()) {
        cmd << _T(" --vpp-edgelevel ") << tmp.str().substr(1);
    } else if (conf->vpp.edgelevel.enable) {
        cmd << _T(" --vpp-edgelevel");
    }

    //tweak
    tmp.str(tstring());
    if (!conf->vpp.tweak.enable) {
        tmp << _T(",enable=false");
    }
    ADD_FLOAT(_T("brightness"), conf->vpp.tweak.brightness, 3);
    ADD_FLOAT(_T("contrast"), conf->vpp.tweak.contrast, 3);
    ADD_FLOAT(_T("gamma"), conf->vpp.tweak.gamma, 3);
    ADD_FLOAT(_T("saturation"), conf->vpp.tweak.saturation, 3);
    ADD_FLOAT(_T("hue"), conf->vpp.tweak.hue, 3);
    if (!tmp.str().empty()) {
        cmd << _T(" --vpp-tweak ") << tmp.str().substr(1);
    } else if (conf->vpp.tweak.enable) {
        cmd << _T(" --vpp-tweak");
    }

    //deband
    tmp.str(tstring());
    if (!conf->vpp.deband.enable) {
        tmp << _T(",enable=false");
    }
    ADD_NUM(_T("range"), conf->vpp.deband.range);
    if (conf->vpp.deband.threY == conf->vpp.deband.threCb
        && conf->vpp.deband.threY == conf->vpp.deband.threCr) {
        ADD_NUM(_T("thre"), conf->vpp.deband.threY);
    } else {
        ADD_NUM(_T("thre_y"), conf->vpp.deband.threY);
        ADD_NUM(_T("thre_cb"), conf->vpp.deband.threCb);
        ADD_NUM(_T("thre_cr"), conf->vpp.deband.threCr);
    }
    if (conf->vpp.deband.ditherY == conf->vpp.deband.ditherC) {
        ADD_NUM(_T("dither"), conf->vpp.deband.ditherY);
    } else {
        ADD_NUM(_T("dither_y"), conf->vpp.deband.ditherY);
        ADD_NUM(_T("dither_c"), conf->vpp.deband.ditherC);
    }
    ADD_NUM(_T("sample"), conf->vpp.deband.sample);
    ADD_BOOL(_T("blurfirst"), conf->vpp.deband.blurFirst);
    ADD_BOOL(_T("rand_each_frame"), conf->vpp.deband.randEachFrame);
    if (!tmp.str().empty()) {
        cmd << _T(" --vpp-deband ") << tmp.str().substr(1);
    } else if (conf->vpp.deband.enable) {
        cmd << _T(" --vpp-deband");
    }
    if (conf->vid.log_debug) {
        cmd << _T(" --log-level debug");
    }
    if (conf->nvenc.perf_monitor) {
        cmd << _T(" --perf-monitor");
    }
    if (conf->vpp.vpp_perf_monitor) {
        cmd << _T(" --vpp-perf-monitor");
    }
    return cmd.str();
}

static void init_CONF_GUIEX_OLD(CONF_GUIEX_OLD *conf, BOOL use_highbit) {
    ZeroMemory(conf, sizeof(CONF_GUIEX_OLD));
    guiEx_config::write_conf_header(&conf->header);
    conf->vid.resize_width = 1280;
    conf->vid.resize_height = 720;
    conf->aud.ext.encoder = 15;
    conf->aud.in.encoder = 1;
    conf->aud.use_internal = 1;
#if ENCODER_QSVENC || ENCODER_NVENC || ENCODER_VCEENC || ENCODER_FFMPEG
    conf->mux.use_internal = TRUE;
#else
    conf->mux.use_internal = FALSE;
#endif
    conf->header.size_all = CONF_INITIALIZED;
}

void guiEx_config::convert_nvencstgv3_to_nvencstgv4(CONF_GUIEX_OLD *conf, const void *dat) {
    init_CONF_GUIEX_OLD(conf, FALSE);

    //いったん古い設定ファイル構造に合わせこむ
    CONF_GUIEX_OLD3 conf_old;
    init_CONF_GUIEX_old(&conf_old);
    for (int i = 0; i < ((CONF_GUIEX_OLD3 *)dat)->block_count; ++i) {
        BYTE *filedat = (BYTE *)dat + ((CONF_GUIEX_OLD3 *)dat)->block_head_p[i];
        BYTE *dst = (BYTE *)&conf_old + conf_block_pointer_old[i];
        memcpy(dst, filedat, std::min(((CONF_GUIEX_OLD3 *)dat)->block_size[i], conf_block_data_old[i]));
    }

    //まずそのままコピーするブロックはそうする
#define COPY_BLOCK(block, block_idx) { memcpy(&conf->block, ((BYTE *)&conf_old) + conf_old.block_head_p[block_idx], std::min((int)sizeof(conf->block), conf_old.block_size[block_idx])); }
    COPY_BLOCK(aud, 2);
    COPY_BLOCK(mux, 3);
    COPY_BLOCK(oth, 4);
#undef COPY_BLOCK

    conf->enc.codec_rgy      = conf_old.nvenc.codec == NV_ENC_HEVC ? RGY_CODEC_HEVC : RGY_CODEC_H264;
    conf->vid.auo_tcfile_out = conf_old.vid.auo_tcfile_out;
    conf->vid.afs            = conf_old.vid.afs;
    conf->vid.resize_enable  = conf_old.vpp.resize_enable;
    conf->vid.resize_width   = conf_old.vpp.resize_width;
    conf->vid.resize_height  = conf_old.vpp.resize_height;

    conf_old.nvenc.codecConfig[NV_ENC_H264].h264Config.adaptiveTransformMode = NV_ENC_H264_ADAPTIVE_TRANSFORM_AUTOSELECT;
    conf_old.nvenc.codecConfig[NV_ENC_H264].h264Config.bdirectMode = NV_ENC_H264_BDIRECT_MODE_AUTOSELECT;
    conf_old.nvenc.codecConfig[NV_ENC_HEVC].hevcConfig.maxCUSize = NV_ENC_HEVC_CUSIZE_AUTOSELECT;
    conf_old.nvenc.codecConfig[NV_ENC_HEVC].hevcConfig.minCUSize = NV_ENC_HEVC_CUSIZE_AUTOSELECT;

    //古い設定ファイルからコマンドラインへ
    //ここではデフォルトパラメータを考慮せず、すべての情報の文字列化を行う
    auto cmd_old = gen_cmd_old3(&conf_old);

    //一度パラメータに戻し、再度コマンドラインに戻すことでデフォルトパラメータの削除を行う
    InEncodeVideoParam enc_prm;
    NV_ENC_CODEC_CONFIG codec_prm[RGY_CODEC_NUM] = { 0 };
    codec_prm[RGY_CODEC_H264] = DefaultParamH264();
    codec_prm[RGY_CODEC_HEVC] = DefaultParamHEVC();
    codec_prm[RGY_CODEC_AV1]  = DefaultParamAV1();
    parse_cmd(&enc_prm, codec_prm, cmd_old.c_str());

    //うまく保存されていないことがある
    enc_prm.encConfig.mvPrecision = NV_ENC_MV_PRECISION_DEFAULT;
    codec_prm[RGY_CODEC_H264].h264Config.adaptiveTransformMode = NV_ENC_H264_ADAPTIVE_TRANSFORM_AUTOSELECT;
    codec_prm[RGY_CODEC_H264].h264Config.bdirectMode = NV_ENC_H264_BDIRECT_MODE_AUTOSELECT;
    codec_prm[RGY_CODEC_HEVC].hevcConfig.maxCUSize = NV_ENC_HEVC_CUSIZE_AUTOSELECT;
    codec_prm[RGY_CODEC_HEVC].hevcConfig.minCUSize = NV_ENC_HEVC_CUSIZE_AUTOSELECT;

    strcpy_s(conf->enc.cmd, tchar_to_string(gen_cmd(&enc_prm, codec_prm, true)).c_str());
}

#pragma warning (pop)