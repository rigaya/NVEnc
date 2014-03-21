/*
 * Copyright 1993-2013 NVIDIA Corporation.  All rights reserved.
 *
 * NOTICE TO LICENSEE:
 *
 * This source code and/or documentation ("Licensed Deliverables") are
 * subject to NVIDIA intellectual property rights under U.S. and
 * international Copyright laws.
 *
 * These Licensed Deliverables contained herein is PROPRIETARY and
 * CONFIDENTIAL to NVIDIA and is being provided under the terms and
 * conditions of a form of NVIDIA software license agreement by and
 * between NVIDIA and Licensee ("License Agreement") or electronically
 * accepted by Licensee.  Notwithstanding any terms or conditions to
 * the contrary in the License Agreement, reproduction or disclosure
 * of the Licensed Deliverables to any third party without the express
 * written consent of NVIDIA is prohibited.
 *
 * ALL NVIDIA DESIGN SPECIFICATIONS, REFERENCE BOARDS, FILES, DRAWINGS,
 * DIAGNOSTICS, LISTS, AND OTHER DOCUMENTS (TOGETHER AND SEPARATELY,
 * gMATERIALSh) ARE BEING PROVIDED gAS IS.h WITHOUT EXPRESS OR IMPLIED
 * WARRANTY OF ANY KIND.  NVIDIA DISCLAIMS ALL WARRANTIES WITH REGARD
 * TO THESE LICENSED DELIVERABLES, INCLUDING ALL IMPLIED WARRANTIES OF
 * MERCHANTABILITY, NONINFRINGEMENT, AND FITNESS FOR A PARTICULAR PURPOSE.
 * NOTWITHSTANDING ANY TERMS OR CONDITIONS TO THE CONTRARY IN THE LICENSE
 * AGREEMENT, IN NO EVENT SHALL NVIDIA BE LIABLE FOR ANY SPECIAL, INDIRECT,
 * INCIDENTAL, OR CONSEQUENTIAL DAMAGES, OR ANY DAMAGES WHATSOEVER RESULTING
 * FROM LOSS OF USE, DATA OR PROFITS, WHETHER IN AN ACTION OF CONTRACT,
 * NEGLIGENCE OR OTHER TORTIOUS ACTION, ARISING OUT OF OR IN CONNECTION
 * WITH THE USE OR PERFORMANCE OF THESE LICENSED DELIVERABLES.
 *
 * Information furnished is believed to be accurate and reliable. However,
 * NVIDIA assumes no responsibility for the consequences of use of such
 * information nor for any infringement of patents or other rights of
 * third parties, which may result from its use.  No License is granted
 * by implication or otherwise under any patent or patent rights of NVIDIA
 * Corporation.  Specifications mentioned in the software are subject to
 * change without notice. This publication supersedes and replaces all
 * other information previously supplied.
 *
 * NVIDIA Corporation products are not authorized for use as critical
 * components in life support devices or systems without express written
 * approval of NVIDIA Corporation.
 *
 * U.S. Government End Users.  These Licensed Deliverables are a
 * "commercial item" as that term is defined at 48 C.F.R. 2.101 (OCT
 * 1995), consisting of "commercial computer software" and "commercial
 * computer software documentation" as such terms are used in 48
 * C.F.R. 12.212 (SEPT 1995) and is provided to the U.S. Government
 * only as a commercial end item.  Consistent with 48 C.F.R.12.212 and
 * 48 C.F.R. 227.7202-1 through 227.7202-4 (JUNE 1995), all
 * U.S. Government End Users acquire the Licensed Deliverables with
 * only those rights set forth herein.
 *
 * Any use of the Licensed Deliverables in individual and commercial
 * software must include, in the user documentation and internal
 * comments to the code, the above Disclaimer and U.S. Government End
 * Users Notice.
 */


#ifndef _XCODEVID_H_
#define _XCODEVID_H_

#include "xcodeutil.h"

class INvVideoEncoder;
class INvVideoEncoderClient;
class INvCudaContext;


// Compression Standard
enum NvVideoCompressionStd
{
    NVCS_Unknown=-1,
    NVCS_H264=4      // 14496-10
};

// Profiles
enum NVEProfile
{
    NVEProfile_Default=0,   // Unspecified
    NVEProfile_Simple,      // Simple / Baseline profile
    NVEProfile_Main,        // Main profile
    NVEProfile_High,        // High / Advanced profile
    NVEProfile_Stereo,      // Stereo MVC profile
};

// Conformance type for a given compression standard (only affect internal settings - may not be compliant
// if external parameters are incorrect)
enum NVECompliance
{
    NVEStd_Generic=0,   // Generic
    NVEStd_VideoCD,     // VideoCD (MPEG-1)
    NVEStd_SVCD,        // SVCD
    NVEStd_DVD,         // DVD Video
    NVEStd_HDTV,        // HDTV
    NVEStd_iPod,        // Apple iPod (H.264)
    NVEStd_PSP,         // Sony PSP (H.264)
    NVEStd_AVCHD,       // AVCHD (H.264)
    NVEStd_BD,          // Blu-ray Disc (H.264)
    NVEStd_ZuneHD,      // Zune HD (H.264)
    NVEStd_FlipCam,     // Flip Camera (H.264)
};

// Motion estimation search algorithm
enum NvMeAlg
{
    NVENC_MOTION_SEARCH_ALG_MRST2       = 0x00,   // Best Quality
    NVENC_MOTION_SEARCH_ALG_ZERO_MV     = 0x01,   // Motion vectors are set to zero
    NVENC_MOTION_SEARCH_ALG_ESA         = 0x02,   // Exhaustive SAD-based search
    NVENC_MOTION_SEARCH_ALG_DIAMOND     = 0x04,   // Small diamond search
    NVENC_MOTION_SEARCH_ALG_HEX         = 0x08,   // Hexagon search
    NVENC_MOTION_SEARCH_ALG_UMH         = 0x10,   // UMH Mixed search
    NVENC_MOTION_SEARCH_ALG_SESA        = 0x20,   // Subsampled ESA
};


// Definitions for video_format
enum
{
    NVEVideoFormat_Component=0,
    NVEVideoFormat_PAL,
    NVEVideoFormat_NTSC,
    NVEVideoFormat_SECAM,
    NVEVideoFormat_MAC,
    NVEVideoFormat_Unspecified,
    NVEVideoFormat_Reserved6,
    NVEVideoFormat_Reserved7
};

// Definitions for color_primaries
enum
{
    NVEColorPrimaries_Forbidden=0,
    NVEColorPrimaries_BT709,
    NVEColorPrimaries_Unspecified,
    NVEColorPrimaries_Reserved,
    NVEColorPrimaries_BT470M,
    NVEColorPrimaries_BT470BG,
    NVEColorPrimaries_SMPTE170M,
    NVEColorPrimaries_SMPTE240M,
    NVEColorPrimaries_GenericFilm
};

// Definitions for transfer_characteristics
enum
{
    NVETransferCharacteristics_Forbidden=0,
    NVETransferCharacteristics_BT709,
    NVETransferCharacteristics_Unspecified,
    NVETransferCharacteristics_Reserved,
    NVETransferCharacteristics_BT470M,
    NVETransferCharacteristics_BT470BG,
    NVETransferCharacteristics_SMPTE170M,
    NVETransferCharacteristics_SMPTE240M,
    NVETransferCharacteristics_Linear,
    NVETransferCharacteristics_Log100,
    NVETransferCharacteristics_Log316
};

// Definitions for matrix_coefficients
enum
{
    NVEMatrixCoefficients_Forbidden=0,
    NVEMatrixCoefficients_BT709,
    NVEMatrixCoefficients_Unspecified,
    NVEMatrixCoefficients_Reserved,
    NVEMatrixCoefficients_FCC,
    NVEMatrixCoefficients_BT470BG,
    NVEMatrixCoefficients_SMPTE170M,
    NVEMatrixCoefficients_SMPTE240M
};


// Picture type
#define TYPE_IFRAME     1
#define TYPE_PFRAME     2
#define TYPE_BFRAME     3

// Maximum raw sequence header length (all codecs)
#define MAX_SEQ_HDR_LEN     (512)   // 512 bytes
// Minimum raw sequence header length (all codecs)
#define MIN_SEQ_HDR_LEN     (100)   // 100 bytes

// Video Encoder Parameters
typedef struct _NvVidEncParams
{
    INvVideoEncoderClient *pClient; // Interface for sending compressed data and misc statistics
    S32 lWidth;                     // Horizontal size
    S32 lHeight;                    // Vertical size
    int bProgSeq;                   // Progressive Sequence
    int iBotFldFirst;               // 0 = unknown (sample-level flag) 1 = always top field first 2 = always bottom field first
    int nGOPLength;                 // Max length of GOP in frames
    int nPFrameDistance;            // 0=I-only, 1=IPP, 2=IBP, 3=IBBP
    int nNumSlices;                 // Number of Slices
    int bClosedGOP;                 // Use only closed GOPs
    int nDynamicGOP;                // Dynamic GOP structure (0:fixed, >=1:new GOP on scene changes, >=2:auto B-frames)
    int bFixedPPS;                  // Used fixed picture-level parameters for the entire sequence
    int bAlwaysReconstruct;         // Always reconstruct the coded frames
    int bMeasureDistortion;         // Send distortion statistics for PSNR computations
    int bDisableDeblock;            // Disable deblocking
    int bDisableCabac;              // Disable CABAC (H.264-specific, forces CAVLC)
    int nNaluFramingType;           // NAL unit framing: 0 = start codes,1,2,4 = big-endian length prefix of size 1, 2, or 4 bytes
    int bDisableSPSPPS;             // Disable SPS/PPS reporting in the bitstream
    int nGPUOffloadLevel;           // if nGPUOffloadLevel <12 pixel procssing on CPU, if nGPUOffloadLevel >= 12 pixel processing on GPU
    int bMultiGPU;                  // 0: force disable 1: enable when deemed fit
    int bLowLatency;                // Low latency (may be used by RC to limit instantaneous bitrate)
    S32 lDARWidth, lDARHeight;      // Display Aspect Ratio = lDARWidth : lDARHeight
    NVFrameRateDescriptor sFrameRateDescriptor; // Frame rate structure
    NVECompliance eConform;         // Additional restrictions for encoding
    NVEProfile eProfile;            // Profile restriction
    S32 iLevel;                     // Levels (0xff: autoselect)
    S32 lBitrateAvg;                // Average bit rate
    S32 lBitratePeak;               // Peak Bitrate for VBR (ignored for CBR)
    S32 lVBRMode;                   // VBRMode (0:CBR, >0:VBR, <0:CQ_VBR)
    S32 lVideoFormat;               // Video Format (NVEVideoFormat_XXX)
    S32 lColorPrimaries;            // Colour Primaries (NVEColorPrimaries_XXX)
    S32 lTransferCharacteristics;   // Transfer Characteristics
    S32 lMatrixCoefficients;        // Matrix Coefficients
    U32 ulInitialTimeCode;          // Initial timecode in microseconds
    NvMeAlg eSearchAlg;             // Motion Estimation search algorithm
    S32 lMESearchWidth;             // ME Search Window width (+/- lMESearchWidth/2)
    S32 lMESearchHeight;            // ME Search Window height (+/- lMESearchHeight/2)
    S32 lTotalFrames;               // Total number of frames in the sequence, 0=unknown (infinite)
    S32 lNumThreads;                // Number of threads to use in the encoder (0=not threaded)
    INvCudaContext *pCudaContext;   // CUDA context (when using cuda-based ME)
    S32 lPFrameMinQp;               // Force P-frame min Qp value (codec-specific scale, default=0=auto)
    S32 lBFrameMinQp;               // Force B-frame min Qp value (codec-specific scale, default=0=auto)
    void *pPrivateEncodeData;       // private encode params to for msenc based encoding
    S32  lPrivateEncodeDataSize;    // size of private encode data passed to msenc
} NvVidEncParams;


// Parameters to EncodeFrame
typedef struct _NVEPicture
{
    IPicBuf *pPicIn;    // Input picture parameters (picture size must be nCodedWidth x nCodedHeight)
    int nRepeatCount;   // 0=not repeated, >=1 repeat count (repeat new frame)
    int nDroppedFrames; // # of dropped frames before this frame (repeat previous frame)
    int bLast;          // Is this frame the last frame ?
    int bChapterPoint;      // force IDR  (this should be set to 1 to force pps, 2 to force sps, and 3 to force IDR
    int bForceKeyFrame;        // force I frame
    int view_id;
} NVEPicture;


// Sequence Information
typedef struct _NVESequenceInfo
{
    int nDisplayWidth;          // Displayed Horizontal Size
    int nDisplayHeight;         // Displayed Vertical Size
    int nCodedWidth;            // Coded Picture Width
    int nCodedHeight;           // Coded Picture Height
    int bProgSeq;               // Progressive Sequence
    NVFrameRateDescriptor sFrameRateDescriptor; // Actual Encoding Frame Rate
    // Rate Control
    S32 lVBVUnderflowCnt;       // VBV Underflow errors
    S32 lVBVOverflowCnt;        // VBV Overflow errors
    S64 llTotalBytes;           // Total bytes coded in the sequence
    // Raw sequence header information (TODO)
    S32 cbSequenceHeader;       // Number of bytes in SequenceHeaderData
    U8 SequenceHeaderData[MAX_SEQ_HDR_LEN]; // Raw sequence header data (codec-specific)
} NVESequenceInfo;


// Information passed to OnBeginFrame
typedef struct _NVEBeginFrameInfo
{
    S32 nFrameNumber;                   // Frame Number
    S32 nPicType;                       // Picture Type
} NVEBeginFrameInfo;


// Compression Statistics passed to OnEndFrame
typedef struct _NVEEndFrameInfo
{
    S32 nFrameNumber;                   // Frame Number
    S32 nPicType;                       // Picture Type
    S32 nPicSize;                       // Compressed picture size (in bytes)
    // The following fields are only present if full reconstruction was requested (bAlwaysReconstruct=true)
    const U8 *pNV12Recon[2];            // Ptr to luma and chroma planes of reconstructed frame
    S32 nReconWidth;                    // Width of reconstructed frame
    S32 nReconHeight;                   // Height of reconstructed frame
    S32 nReconPitch;                    // Pitch of reconstructed frame
    // The following fields are only valid if SNR was requested (bMeasureDistortion=true)
    S64 llLumaTSE;                      // Total Square Error of luminance plane (Y)
    S64 llChromaTSE;                    // Total Square Error of chrominance plane (UV)
    // Rate control statistics
    S32 lPicSizePredicted;              // Predicted picture size (in bytes)
    S32 lIntraBlockBits;                // Bits used by intra blocks (quantized coefficients)
    S32 lInterBlockBits;                // Bits used by inter blocks (quantized coefficients)
    S32 lIntraMBBits;                   // Total bits in intra macroblocks
    S32 lInterMBBits;                   // Total bits in inter macroblocks
    S32 lIntraMBCnt;                    // Number of intra macroblocks in this picture
    float flTextureComplexity;          // Texture complexity (intra energy)
    float flMotionComplexity;           // Predicted motion complexity (inter_energy / intra_energy)
    float flVBVLevelPeak;               // Peak bitrate VBV fullness (normalized) before coding this picture
    float flVBVLevelAvg;                // Avg. bitrate VBV fullness (normalized) before coding this picture
    float flQuantTarget;                // Target quantization
    float flQuantActual;                // Actual quantization
    double mse[3];                      // Mean square error reporting to common layer for PSNR calc
} NVEEndFrameInfo;


// Interface to allow the video encoder to communicate with the client
// Sequence of calls from encoder to client:
// - OnBeginFrame
// - Acquire/Release Bitstream (data for current frame)
// - OnEndFrame
//
class INvVideoEncoderClient
{
    public:
        virtual unsigned char *AcquireBitstream(int *pBufferSize) = 0;
        virtual void ReleaseBitstream(int nBytesInBuffer) = 0;
        virtual void OnBeginFrame(const NVEBeginFrameInfo *pbfi) = 0;
        virtual void OnEndFrame(const NVEEndFrameInfo *pefi) = 0;

    protected:
        virtual ~INvVideoEncoderClient() {}
};


// High-level interface to video encoder
class INvVideoEncoder: public virtual INvRefCount
{
    public:
        virtual bool Initialize(const NvVidEncParams *pnvep) = 0;
        virtual bool Deinitialize() = 0;
        virtual bool EncodeFrame(NVEPicture *pFrmIn) = 0;
        virtual void GetSequenceInfo(NVESequenceInfo *pnvsi) = 0;
        virtual bool IsCudaEncoder() = 0;
};

extern bool XCODEAPI CreateNvVideoEncoder(INvVideoEncoder **ppobj, NvVideoCompressionStd eCompression);


/////////////////////////////////////////////////////////////////////////////////////////
//
// Decoder API
//
/////////////////////////////////////////////////////////////////////////////////////////


typedef struct _NVMPEG2PictureData
{
    IPicBuf *pForwardRef;           // Forward reference (P/B-frames)
    IPicBuf *pBackwardRef;          // Backward reference (B-frames)
    int picture_coding_type;        // TYPE_?FRAME
    int full_pel_forward_vector;
    int full_pel_backward_vector;
    int f_code[2][2];
    int intra_dc_precision;
    int frame_pred_frame_dct;
    int concealment_motion_vectors;
    int q_scale_type;
    int intra_vlc_format;
    int alternate_scan;
    // Quantization matrices (raster order)
    unsigned char QuantMatrixIntra[64];
    unsigned char QuantMatrixInter[64];
} NVMPEG2PictureData;


typedef struct _NVH264DPBEntry
{
    IPicBuf *pPicBuf;       // ptr to reference frame
    int FrameIdx;           // frame_num(short-term) or LongTermFrameIdx(long-term)
    int is_long_term;       // 0=short term reference, 1=long term reference
    int not_existing;       // non-existing reference frame (corresponding PicIdx should be set to -1)
    int used_for_reference; // 0=unused, 1=top_field, 2=bottom_field, 3=both_fields
    int FieldOrderCnt[2];   // field order count of top and bottom fields
} NVH264DPBEntry;


typedef struct _NVH264PictureData
{
    // SPS
    int log2_max_frame_num_minus4;
    int pic_order_cnt_type;
    int log2_max_pic_order_cnt_lsb_minus4;
    int delta_pic_order_always_zero_flag;
    int frame_mbs_only_flag;
    int direct_8x8_inference_flag;
    int num_ref_frames;
    int residual_colour_transform_flag;
    int qpprime_y_zero_transform_bypass_flag;
    // PPS
    int num_ref_idx_l0_active_minus1;
    int num_ref_idx_l1_active_minus1;
    int weighted_pred_flag;
    int weighted_bipred_idc;
    int pic_init_qp_minus26;
    int redundant_pic_cnt_present_flag;
    unsigned char deblocking_filter_control_present_flag;
    unsigned char transform_8x8_mode_flag;
    unsigned char MbaffFrameFlag;
    unsigned char constrained_intra_pred_flag;
    unsigned char entropy_coding_mode_flag;
    unsigned char pic_order_present_flag;
    signed char chroma_qp_index_offset;
    signed char second_chroma_qp_index_offset;
    int frame_num;
    int CurrFieldOrderCnt[2];
    unsigned char fmo_aso_enable;
    unsigned char num_slice_groups_minus1;
    unsigned char slice_group_map_type;
    signed char pic_init_qs_minus26;
    unsigned int slice_group_change_rate_minus1;
    const unsigned char *pMb2SliceGroupMap;
    // DPB
    NVH264DPBEntry dpb[16+1];          // List of reference frames within the DPB

    // Quantization Matrices (raster-order)
    unsigned char WeightScale4x4[6][16];
    unsigned char WeightScale8x8[2][64];
    union
    {
        // MVC extension
        struct
        {
            int num_views_minus1;
            int view_id;
            unsigned char inter_view_flag;
            unsigned char num_inter_view_refs_l0;
            unsigned char num_inter_view_refs_l1;
            unsigned char MVCReserved8Bits;
            int InterViewRefsL0[16];
            int InterViewRefsL1[16];
        } mvcext;
        // SVC extension
        struct
        {
            unsigned char profile_idc;
            unsigned char level_idc;
            unsigned char DQId;
            unsigned char DQIdMax;
            unsigned char disable_inter_layer_deblocking_filter_idc;
            unsigned char ref_layer_chroma_phase_y_plus1;
            char  inter_layer_slice_alpha_c0_offset_div2;
            char  inter_layer_slice_beta_offset_div2;
            unsigned short DPBEntryValidFlag;

            union
            {
                struct
                {
                    unsigned char inter_layer_deblocking_filter_control_present_flag     : 1;
                    unsigned char extended_spatial_scalability_idc                       : 2;
                    unsigned char adaptive_tcoeff_level_prediction_flag                  : 1;
                    unsigned char slice_header_restriction_flag                          : 1;
                    unsigned char chroma_phase_x_plus1_flag                              : 1;
                    unsigned char chroma_phase_y_plus1                                   : 2;
                    unsigned char tcoeff_level_prediction_flag                           : 1;
                    unsigned char constrained_intra_resampling_flag                      : 1;
                    unsigned char ref_layer_chroma_phase_x_plus1_flag                    : 1;
                    unsigned char store_ref_base_pic_flag                                : 1;
                    unsigned char Reserved                                               : 4;
                } f;
                unsigned char ucBitFields[2];
            };

            union
            {
                short seq_scaled_ref_layer_left_offset;
                short scaled_ref_layer_left_offset;
            };
            union
            {
                short seq_scaled_ref_layer_top_offset;
                short scaled_ref_layer_top_offset;
            };
            union
            {
                short seq_scaled_ref_layer_right_offset;
                short scaled_ref_layer_right_offset;
            };
            union
            {
                short seq_scaled_ref_layer_bottom_offset;
                short scaled_ref_layer_bottom_offset;
            };
        } svcext;
    };
} NVH264PictureData;


typedef struct _NVVC1PictureData
{
    IPicBuf *pForwardRef;   // Forward reference (P/B-frames)
    IPicBuf *pBackwardRef;  // Backward reference (B-frames)
    int FrameWidth;         // Actual frame width
    int FrameHeight;        // Actual frame height
    // SEQUENCE
    int profile;
    int postprocflag;
    int pulldown;
    int interlace;
    int tfcntrflag;
    int finterpflag;
    int psf;
    int multires;
    int syncmarker;
    int rangered;
    int maxbframes;
    // ENTRYPOINT
    int panscan_flag;
    int refdist_flag;
    int extended_mv;
    int dquant;
    int vstransform;
    int loopfilter;
    int fastuvmc;
    int overlap;
    int quantizer;
    int extended_dmv;
    int range_mapy_flag;
    int range_mapy;
    int range_mapuv_flag;
    int range_mapuv;
    int rangeredfrm;    // range reduction state
} NVVC1PictureData;


typedef struct _NVMPEG4PictureData
{
    IPicBuf *pForwardRef;           // Forward reference (P/B-frames)
    IPicBuf *pBackwardRef;          // Backward reference (B-frames)
    // VOL
    int video_object_layer_width;
    int video_object_layer_height;
    int vop_time_increment_resolution;
    int vop_time_increment_bitcount;
    int resync_marker_disable;
    int quant_type;
    int vop_quant;
    int quarter_sample;
    int short_video_header;
    int divx_flags;
    // VOP
    int vop_coding_type;
    int vop_coded;
    int vop_rounding_type;
    int alternate_vertical_scan_flag;
    int interlaced;
    int vop_fcode_forward;
    int vop_fcode_backward;
    int trd[2];
    int trb[2];
    // shdeshpande : DivX GMC Concealment
    // Flag to prevent decoding of non I-VOPs during a GMC sequence
    // and indicate beginning / end of a GMC sequence.
    bool bGMCConceal;
    // Quantization matrices (raster order)
    unsigned char QuantMatrixIntra[64];
    unsigned char QuantMatrixInter[64];
} NVMPEG4PictureData;


typedef struct _NVDPictureData
{
    int PicWidthInMbs;      // Coded Frame Size
    int FrameHeightInMbs;   // Coded Frame Height
    IPicBuf *pCurrPic;      // Current picture (output)
    int field_pic_flag;     // 0=frame picture, 1=field picture
    int bottom_field_flag;  // 0=top field, 1=bottom field (ignored if field_pic_flag=0)
    int second_field;       // Second field of a complementary field pair
    int progressive_frame;  // Frame is progressive
    int top_field_first;    // Frame pictures only
    int repeat_first_field; // For 3:2 pulldown (number of additional fields, 2=frame doubling, 4=frame tripling)
    int ref_pic_flag;       // Frame is a reference frame
    int intra_pic_flag;     // Frame is entirely intra coded (no temporal dependencies)
    int chroma_format;      // Chroma Format (should match sequence info)
    int picture_order_count; // picture order count (if known)
    // Bitstream data
    unsigned int nBitstreamDataLen;        // Number of bytes in bitstream data buffer
    const unsigned char *pBitstreamData;   // Ptr to bitstream data for this picture (slice-layer)
    unsigned int nNumSlices;               // Number of slices in this picture
    const unsigned int *pSliceDataOffsets; // nNumSlices entries, contains offset of each slice within the bitstream data buffer
    // Codec-specific data
    union
    {
        NVMPEG2PictureData mpeg2;   // Also used for MPEG-1
        NVMPEG4PictureData mpeg4;
        NVH264PictureData h264;
        NVVC1PictureData vc1;
    } CodecSpecific;
} NVDPictureData;


// Packet input for parsing
typedef struct _NVDBitstreamPacket
{
    const U8 *pByteStream;  // Ptr to byte stream data
    S32 nDataLength;        // Data length for this packet
    int bEOS;               // TRUE if this is an End-Of-Stream packet (flush everything)
    int bPTSValid;          // TRUE if llPTS is valid (also used to detect frame boundaries for VC1 SP/MP)
    int bDiscontinuity;     // TRUE if DecMFT is signalling a discontinuity
    int bPartialParsing;    // 0: parse entire packet, 1: parse until next decode/display event
    S64 llPTS;              // Presentation Time Stamp for this packet (clock rate specified at initialization)
} NVDBitstreamPacket;


// Sequence information
typedef struct _NVDSequenceInfo
{
    NvVideoCompressionStd eCodec;   // Compression Standard
    NvFrameRate eFrameRate;         // Frame Rate stored in the bitstream
    int bProgSeq;                   // Progressive Sequence
    int nDisplayWidth;              // Displayed Horizontal Size
    int nDisplayHeight;             // Displayed Vertical Size
    int nCodedWidth;                // Coded Picture Width
    int nCodedHeight;               // Coded Picture Height
    U8 nChromaFormat;               // Chroma Format (0=4:0:0, 1=4:2:0, 2=4:2:2, 3=4:4:4)
    U8 uBitDepthLumaMinus8;         // Luma bit depth (0=8bit)
    U8 uBitDepthChromaMinus8;       // Chroma bit depth (0=8bit)
    U8 uPad;                        // For alignment
    S32 lBitrate;                   // Video bitrate (bps)
    S32 lDARWidth, lDARHeight;      // Display Aspect Ratio = lDARWidth : lDARHeight
    S32 lVideoFormat;               // Video Format (NVEVideoFormat_XXX)
    S32 lColorPrimaries;            // Colour Primaries (NVEColorPrimaries_XXX)
    S32 lTransferCharacteristics;   // Transfer Characteristics
    S32 lMatrixCoefficients;        // Matrix Coefficients
    S32 cbSequenceHeader;           // Number of bytes in SequenceHeaderData
    U8 SequenceHeaderData[MAX_SEQ_HDR_LEN]; // Raw sequence header data (codec-specific)
} NVDSequenceInfo;

// Slice-level information
enum slice_type_e
{
    P = 0,
    B = 1,
    I = 2,
    SP = 3,
    SI = 4,
};

typedef struct _NVH264SliceInfo
{
    U8 num_ref_idx_l0_active_minus1;
    U8 num_ref_idx_l1_active_minus1;
    U8 cabac_init_idc;
    S8 slice_qp_delta;
    U8 disable_deblocking_filter_idc;
    S8 slice_alpha_c0_offset_div2;
    S8 slice_beta_offset_div2;
    S8 RefPicList[2][32];   // [list][idx], format: dpb_index*2+bottom_field
} NVH264SliceInfo;

typedef struct _NVDSliceInfo
{
    int mbx;        // horizontal location of slice
    int mby;        // vertical location of slice
    int slice_type; // slice_type_e
    union
    {
        NVH264SliceInfo h264;
    } CodecSpecific;
} NVDSliceInfo;

enum
{
    NVD_CAPS_MVC = 0x01,
    NVD_CAPS_SVC = 0x02,
};

// Interface to allow decoder to communicate with the client
class INvVideoDecoderClient
{
    public:
        virtual S32 BeginSequence(const NVDSequenceInfo *pnvsi) = 0; // Returns max number of reference frames (always at least 2 for MPEG-2)
        virtual bool AllocPictureBuffer(IPicBuf **ppPicBuf) = 0;    // Returns a new IPicBuf interface
        virtual bool DecodePicture(NVDPictureData *pnvdp) = 0;      // Called when a picture is ready to be decoded
        virtual bool DisplayPicture(IPicBuf *pPicBuf, S64 llPTS) = 0; // Called when a picture is ready to be displayed
        virtual void UnhandledNALU(const U8 *pbData, S32 cbData) = 0; // Called for custom NAL parsing (not required)
        virtual U32 GetDecodeCaps()
        {
            return 0;    // NVD_CAPS_XXX
        }

    protected:
        virtual ~INvVideoDecoderClient() { }
};

// Initialization parameters for decoder class
typedef struct _NvVidDecParams
{
    INvVideoDecoderClient *pClient;     // should always be present if using parsing functionality
    S32 lReferenceClockRate;            // ticks per second of PTS clock (0=default=10000000=10Mhz)
    S32 lErrorThreshold;                // threshold for deciding to bypass of picture (0=do not decode, 100=always decode)
    NVDSequenceInfo *pExternalSeqInfo;  // optional external sequence header data from system layer
} NvVidDecParams;


// High-level interface to video decoder (Note that parsing and decoding functionality are decoupled from each other)
class INvVideoDecoder: public virtual INvRefCount
{
    public:
        virtual bool Initialize(NvVidDecParams *pnvdp) = 0;
        virtual bool Deinitialize() = 0;
        virtual bool DecodePicture(NVDPictureData *pnvdp) = 0;
        virtual bool ParseByteStream(const NVDBitstreamPacket *pck, int *pParsedBytes=NULL) = 0;
        virtual bool DecodeSliceInfo(NVDSliceInfo *psli, const NVDPictureData *pnvdp, int iSlice) = 0;
};

extern bool XCODEAPI CreateNvVideoDecoder(INvVideoDecoder **ppobj, NvVideoCompressionStd eCompression);

/////////////////////////////////////////////////////////////////////////////////////////

#endif // _XPREPROC_H_
