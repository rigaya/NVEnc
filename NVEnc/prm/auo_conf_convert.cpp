// -----------------------------------------------------------------------------------------
// NVEnc by rigaya
// -----------------------------------------------------------------------------------------
// The MIT License
//
// Copyright (c) 1999-2016 rigaya
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
// ------------------------------------------------------------------------------------------

#include <string.h>
#include <stdio.h>
#include <stddef.h>
#include <Windows.h>
#include <shlwapi.h>
#pragma comment(lib, "shlwapi.lib")
#include "auo_util.h"
#include "auo_conf.h"

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

void guiEx_config::convert_nvencstg_to_nvencstgv3(CONF_GUIEX *conf, const void *dat) {
    const CONF_GUIEX *old_data = (const CONF_GUIEX *)dat;
    init_CONF_GUIEX(conf, FALSE);

    //まずそのままコピーするブロックはそうする
#define COPY_BLOCK(block, block_idx) { memcpy(&conf->block, ((BYTE *)old_data) + old_data->block_head_p[block_idx], min(sizeof(conf->block), old_data->block_size[block_idx])); }
    COPY_BLOCK(nvenc, 0);
    COPY_BLOCK(vid, 1);
    COPY_BLOCK(aud, 2);
    COPY_BLOCK(mux, 3);
    COPY_BLOCK(oth, 4);
#undef COPY_BLOCK

    CONF_NVENC_OLD *old = (CONF_NVENC_OLD *)(((BYTE *)old_data) + old_data->block_head_p[0]);
    NV_ENC_CONFIG_H264_OLD * h264old = (NV_ENC_CONFIG_H264_OLD *)&old->enc_config.encodeCodecConfig.h264Config;

    memset(&conf->nvenc.codecConfig[NV_ENC_H264].h264Config, 0, sizeof(conf->nvenc.codecConfig[NV_ENC_H264].h264Config));
#define COPY_H264_STG(name) { conf->nvenc.codecConfig[NV_ENC_H264].h264Config.name = h264old->name; }
    COPY_H264_STG(enableTemporalSVC);
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

    convert_nvencstgv2_to_nvencstgv3(conf);
}

void guiEx_config::convert_nvencstgv2_to_nvencstgv3(CONF_GUIEX *conf) {
    static const DWORD OLD_FLAG_AFTER  = 0x01;
    static const DWORD OLD_FLAG_BEFORE = 0x02;

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

