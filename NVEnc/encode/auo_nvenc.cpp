//  -----------------------------------------------------------------------------------------
//    NVEnc by rigaya
//  -----------------------------------------------------------------------------------------
//   ソースコードについて
//   ・無保証です。
//   ・本ソースコードを使用したことによるいかなる損害・トラブルについてrigayaは責任を負いません。
//   以上に了解して頂ける場合、本ソースコードの使用、複製、改変、再頒布を行って頂いて構いません。
//  -----------------------------------------------------------------------------------------

#include <Windows.h>
#include <Process.h>
#include <mmsystem.h>
#pragma comment(lib, "winmm.lib") 
#include <stdlib.h>
#include <stdio.h>
#include <tchar.h>

#include "output.h"
#include "vphelp_client.h"

#pragma warning( push )
#pragma warning( disable: 4127 )
#include "afs_client.h"
#pragma warning( pop )

#include "convert.h"

#include "auo.h"
#include "auo_frm.h"
#include "auo_pipe.h"
#include "auo_error.h"
#include "auo_conf.h"
#include "auo_util.h"
#include "auo_system.h"
#include "auo_version.h"

#include "auo_encode.h"
#include "auo_video.h"
#include "auo_audio_parallel.h"

#include "auo_nvenc.h"

AUO_RESULT aud_parallel_task(const OUTPUT_INFO *oip, PRM_ENC *pe);

static int calc_input_frame_size(int width, int height, int color_format) {
	width = (color_format == CF_RGB) ? (width+3) & ~3 : (width+1) & ~1;
	return width * height * COLORFORMATS[color_format].size;
}

BOOL setup_afsvideo(const OUTPUT_INFO *oip, CONF_GUIEX *conf, PRM_ENC *pe, BOOL auto_afs_disable) {
	//すでに初期化してある または 必要ない
	if (pe->afs_init || pe->video_out_type == VIDEO_OUTPUT_DISABLED || !conf->vid.afs)
		return TRUE;

	const int color_format = CF_YUY2;
	const int frame_size = calc_input_frame_size(oip->w, oip->h, color_format);
	//Aviutl(自動フィールドシフト)からの映像入力
	if (afs_vbuf_setup((OUTPUT_INFO *)oip, conf->vid.afs, frame_size, COLORFORMATS[color_format].FOURCC)) {
		pe->afs_init = TRUE;
		return TRUE;
	} else if (conf->vid.afs && auto_afs_disable) {
		afs_vbuf_release(); //一度解放
		warning_auto_afs_disable();
		conf->vid.afs = FALSE;
		//再度使用するmuxerをチェックする
		pe->muxer_to_be_used = check_muxer_to_be_used(conf, pe->video_out_type, (oip->flag & OUTPUT_INFO_FLAG_AUDIO) != 0);
		return TRUE;
	}
	//エラー
	error_afs_setup(conf->vid.afs, auto_afs_disable);
	return FALSE;
}

void close_afsvideo(PRM_ENC *pe) {
	if (!pe->afs_init || pe->video_out_type == VIDEO_OUTPUT_DISABLED)
		return;

	afs_vbuf_release();

	pe->afs_init = FALSE;
}

HRESULT CAuoNvEncoderH264::LoadCurrentFrame(unsigned char *yuvInput[3], HANDLE hInputYUVFile, unsigned int dwFrmIndex,
	unsigned int dwFileWidth, unsigned int dwFileHeight, unsigned int dwSurfWidth, unsigned int dwSurfHeight,
	bool bFieldPic, bool bTopField, int FrameQueueSize, int chromaFormatIdc) {

	if (FALSE != (pe->aud_parallel.abort = oip->func_is_abort()))
		return false;

	int i_frame = frames + pe->drop_count;

	if (i_frame >= oip->n) {
		oip->func_rest_time_disp(i_frame, oip->n);
		release_audio_parallel_events(pe);
		return false;
	}

	void *frame;
	if (conf->vid.afs) {
		BOOL drop = FALSE;
		for (;;) {
			if ((frame = afs_get_video((OUTPUT_INFO *)oip, i_frame, &drop, &jitter[i_frame + 1])) == NULL) {
				error_afs_get_frame();
				return false;
			}
			if (!drop)
				break;
			jitter[i_frame] = DROP_FRAME_FLAG;
			pe->drop_count++;
			i_frame++;
			if (i_frame >= oip->n) {
				oip->func_rest_time_disp(i_frame, oip->n);
				release_audio_parallel_events(pe);
				return false;
			}
		}
	} else {
		if ((frame = oip->func_get_video_ex(i_frame, COLORFORMATS[CF_YUY2].FOURCC)) == NULL) {
			error_afs_get_frame();
			return false;
		}
	}

	//convert yuy2->yv12
	for (int y = 0; y < height; y += 2) {
		BYTE *src = (BYTE *)frame + (y * width << 2);
		BYTE *dst_y = dat[0] + (y * width);
		BYTE *dst_u = dat[1] + (y * width >> 1);
		BYTE *dst_v = dat[2] + (y * width >> 1);
		for (int x = 0; x < width; x++, src += 4, dst_y += 2, dst_u++, dst_v++) {
			dst_y[0] = src[0  << 1];
			dst_y[1] = src[1  << 1];
			dst_y[width + 0] = src[(width + 0) << 1];
			dst_y[width + 1] = src[(width + 1) << 1];
			dst_u[0] = (src[1] + src[(width << 1) + 1] + 1) >> 1;
			dst_v[0] = (src[3] + src[(width << 1) + 3] + 1) >> 1;
		}
	}

	if (!(frames & 7)) {
		aud_parallel_task(oip, pe);
		oip->func_rest_time_disp(frames + pe->drop_count, oip->n);
		oip->func_update_preview();
	}

	frames++;
	return true;
}
void CAuoNvEncoderH264::passAuoInfo(const OUTPUT_INFO *_oip, CONF_GUIEX *_conf, PRM_ENC *_pe, int _frames, int *_jitter) {
	oip = _oip;
	conf = _conf;
	pe = _pe;
	frames = _frames;
	jitter = _jitter;
}
void CAuoNvEncoderH264::setAuoInfo(NV_ENC_CONFIG *nvenc, EncoderInputParams *nvenc2, const OUTPUT_INFO *_oip) {
	nvenc2->width = oip->w;
	nvenc2->height = oip->h;
	nvenc2->endFrame = oip->n;
	nvenc2->frameRateNum = oip->rate;
	nvenc2->frameRateDen = oip->scale;
	//自動設定
	if (nvenc->gopLength == UINT_MAX) {
		nvenc->gopLength = (int)(oip->rate / (double)oip->scale + 0.5) * 10;
	}
}
int CAuoNvEncoderH264::NVEncPrintf(int log_level, const char *format, ...) {
	log_level = clamp(log_level, LOG_INFO, LOG_ERROR);

	va_list args;
	va_start(args, format);

	int len = _vscprintf(format, args);
	char *const buffer = (char*)malloc((len+1) * sizeof(buffer[0])); // _vscprintf doesn't count terminating '\0'

	vsprintf_s(buffer, len+1, format, args);

	static const char *const LOG_LEVEL_STR[] ={ "info", "warning", "error" };
	const int mes_line_len = len+1 + strlen("nvenc [warning]: ");
	char *const mes_line = (char *)malloc(mes_line_len * sizeof(mes_line[0]));

	char *a, *b, *mes = buffer;
	char *const fin = mes + len+1; //null文字の位置
	while ((a = strchr(mes, '\n')) != NULL) {
		if ((b = strrchr(mes, '\r', a - mes - 2)) != NULL)
			mes = b + 1;
		*a = '\0';
		sprintf_s(mes_line, mes_line_len, "nvenc [%s]: %s", LOG_LEVEL_STR[log_level], mes);
		write_log_line(log_level, mes_line);
		mes = a + 1;
	}
	if ((a = strrchr(mes, '\r', fin - mes - 1)) != NULL) {
		b = a - 1;
		while (*b == ' ' || *b == '\r')
			b--;
		*(b+1) = '\0';
		if ((b = strrchr(mes, '\r', b - mes - 2)) != NULL)
			mes = b + 1;
		set_window_title(mes);
		mes = a + 1;
	}

	free(buffer);
	free(mes_line);
	return len;
}
int CAuoNvEncoderH264::NvPrintProgress(double progress, const char *format, ...) {
	va_list args;
	va_start(args, format);

	const int len = _vscprintf(format, args);
	char *const buffer = (char*)malloc((len+1) * sizeof(buffer[0])); // _vscprintf doesn't count terminating '\0'

	vsprintf_s(buffer, len+1, format, args);

	set_log_title_and_progress(buffer, progress);

	free(buffer);
	return len;
}
HRESULT CAuoNvEncoderH264::NVEncAvailable() {
	return OpenEncodeSession();
}

// Initialization code that checks the GPU encoders available and fills a table
unsigned int checkNumberEncoders(EncoderGPUInfo *encoderInfo)
{
    CUresult cuResult = CUDA_SUCCESS;
    CUdevice cuDevice = 0;

    char gpu_name[100];
    int  deviceCount = 0;
    int  SMminor = 0, SMmajor = 0;
    int  NVENC_devices = 0;

    NvPrintf("\n");

    // CUDA interfaces
    cuResult = cuInit(0);

    if (cuResult != CUDA_SUCCESS)
    {
        fprintf(stderr, ">> GetNumberEncoders() - cuInit() failed error:0x%x\n", cuResult);
        exit(EXIT_FAILURE);
    }

    checkCudaErrors(cuDeviceGetCount(&deviceCount));

    if (deviceCount == 0)
    {
        fprintf(stderr, ">> GetNumberEncoders() - reports no devices available that support CUDA\n");
        exit(EXIT_FAILURE);
    }
    else
    {
        NvPrintf(">> GetNumberEncoders() has detected %d CUDA capable GPU device(s) <<\n", deviceCount);

        for (int currentDevice=0; currentDevice < deviceCount; currentDevice++)
        {
            checkCudaErrors(cuDeviceGet(&cuDevice, currentDevice));
            checkCudaErrors(cuDeviceGetName(gpu_name, 100, cuDevice));
            checkCudaErrors(cuDeviceComputeCapability(&SMmajor, &SMminor, currentDevice));
            NvPrintf("  [ GPU #%d - < %s > has Compute SM %d.%d, NVENC %s ]\n",
                   currentDevice, gpu_name, SMmajor, SMminor,
                   (((SMmajor << 4) + SMminor) >= 0x30) ? "Available" : "Not Available");

            if (((SMmajor << 4) + SMminor) >= 0x30)
            {
                encoderInfo[NVENC_devices].device = currentDevice;
                strcpy(encoderInfo[NVENC_devices].gpu_name, gpu_name);
                NVENC_devices++;
            }
        }
    }

    return NVENC_devices;
}

bool nvencAvailable() {
	CAuoNvEncoderH264 test;
	return 0 == test.NVEncAvailable();
}

void getDefaultParam(NV_ENC_CONFIG *nvenc, EncoderInputParams *nvenc2) {
	memset(nvenc, 0, sizeof(NV_ENC_CONFIG));
	nvenc->version = NV_ENC_CONFIG_VER;
	nvenc->frameFieldMode = NV_ENC_PARAMS_FRAME_FIELD_MODE_FRAME;
	nvenc->frameIntervalP = 4; //Bフレーム 3
	nvenc->gopLength = UINT_MAX;
	nvenc->mvPrecision = NV_ENC_MV_PRECISION_QUARTER_PEL;
	nvenc->profileGUID = NV_ENC_H264_PROFILE_HIGH_GUID;
	nvenc->rcParams.rateControlMode = NV_ENC_PARAMS_RC_CONSTQP;
	nvenc->rcParams.averageBitRate = 3000000;
	nvenc->rcParams.maxBitRate = 15000000;
	nvenc->rcParams.vbvBufferSize = 15000000;
	nvenc->rcParams.vbvInitialDelay = 0;
	nvenc->rcParams.constQP.qpIntra = 22;
	nvenc->rcParams.constQP.qpInterP = 24;
	nvenc->rcParams.constQP.qpInterB = 26;
	nvenc->rcParams.initialRCQP = nvenc->rcParams.constQP;
	nvenc->rcParams.enableMinQP = 0;
	nvenc->rcParams.enableMaxQP = 0;
	nvenc->rcParams.minQP.qpIntra = 8;
	nvenc->rcParams.minQP.qpInterP = 10;
	nvenc->rcParams.minQP.qpInterB = 12;
	nvenc->rcParams.maxQP.qpIntra = 51;
	nvenc->rcParams.maxQP.qpInterP = 51;
	nvenc->rcParams.maxQP.qpInterB = 51;
	nvenc->rcParams.enableInitialRCQP = 0;
	nvenc->encodeCodecConfig.h264Config.level = 40;
	nvenc->encodeCodecConfig.h264Config.idrPeriod = nvenc->gopLength;
	nvenc->encodeCodecConfig.h264Config.maxNumRefFrames = 3;

	memset(&nvenc2, 0, sizeof(EncoderInputParams));
	nvenc2->enablePTD = 1;
	nvenc2->interfaceType = NV_ENC_CUDA;
	nvenc2->syncMode = 1;
	nvenc2->qualtiy_preset = NV_ENC_PRESET_HQ;
}
