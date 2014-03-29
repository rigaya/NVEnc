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
#include "ConvertCSP.h"

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

AuoEncodeStatus::AuoEncodeStatus() {

}

AuoEncodeStatus::~AuoEncodeStatus() {

}

void AuoEncodeStatus::UpdateDisplay(const TCHAR *mes) {
	set_log_title_and_progress(mes, (m_sData.frameOut + m_sData.frameDrop) / (double)m_sData.frameTotal);
}

void AuoEncodeStatus::WriteLine(const TCHAR *mes) {
	const char *HEADER = "nvenc [info]: ";
	int buf_len = strlen(mes) + 1 + strlen(HEADER);
	char *buf = (char *)calloc(buf_len, sizeof(buf[0]));
	if (buf) {
		memcpy(buf, HEADER, strlen(HEADER));
		memcpy(buf + strlen(HEADER), mes, strlen(mes) + 1);
		write_log_line(LOG_INFO, buf);
		free(buf);
	}
}

AuoInput::AuoInput() {
	oip = NULL;
	conf = NULL;
	pe = NULL;
	m_tmLastUpdate = timeGetTime();
	m_pause = FALSE;
	m_ConvCSPInfo = NULL;
}

AuoInput::~AuoInput() {
	Close();
}

void AuoInput::Close() {
	if (pe)
		close_afsvideo(pe);
	oip = NULL;
	conf = NULL;
	pe = NULL;
	m_iFrame = 0;
	m_ConvCSPInfo = NULL;
	disable_enc_control();
}
int AuoInput::Init(InputVideoInfo *inputPrm, EncodeStatus *pStatus) {
	Close();
	
	m_pStatus = pStatus;
	AuoInputInfo *info = reinterpret_cast<AuoInputInfo *>(inputPrm->otherPrm);
	memcpy(m_crop, inputPrm->crop, sizeof(m_crop));

	oip = info->oip;
	conf = info->conf;
	pe = info->pe;
	jitter = info->jitter;
	m_interlaced = info->interlaced;

	pStatus->m_sData.frameTotal = oip->n;
	inputPrm->width = oip->w;
	inputPrm->height = oip->h;
	inputPrm->rate = oip->rate;
	inputPrm->scale = oip->scale;

//表でうっとおしいので省略する
#define NONE  AUO_SIMD_NONE
#define SSE2  AUO_SIMD_SSE2
#define SSE3  AUO_SIMD_SSE3
#define SSSE3 AUO_SIMD_SSSE3
#define SSE41 AUO_SIMD_SSE41
#define SSE42 AUO_SIMD_SSE42
#define AVX   AUO_SIMD_AVX
#define AVX2  AUO_SIMD_AVX2

	static const ConvCSPInfo funcList[] = {
		{ convert_yuy2_to_nv12_avx2, convert_yuy2_to_nv12_i_avx2,  AVX2|AVX       },
		{ convert_yuy2_to_nv12_avx,  convert_yuy2_to_nv12_i_avx,   AVX|SSSE3|SSE2 },
		{ convert_yuy2_to_nv12_sse2, convert_yuy2_to_nv12_i_ssse3, SSSE3|SSE2     },
		{ convert_yuy2_to_nv12_sse2, convert_yuy2_to_nv12_i_sse2,  SSE2           },
		{ convert_yuy2_to_nv12,      convert_yuy2_to_nv12_i,       NONE           },
		{ NULL, NULL, 0 }
	};

	const DWORD availableSIMD = get_availableSIMD();
	for (int i = 0; funcList[i].func[0]; i++) {
		if ((funcList[i].SIMD & availableSIMD) != funcList[i].SIMD)
			continue;

		m_ConvCSPInfo = &funcList[i];
		break;
	}

	enable_enc_control(&m_pause, pe->afs_init, FALSE, timeGetTime(), oip->n);

	if (conf->vid.afs) {
		if (!setup_afsvideo(oip, conf, pe, FALSE)) {
			m_inputMes = _T("raw: 自動フィールドシフトの初期化に失敗しました。\n");
			return 1;
		}
	}

	TCHAR buf[128] = { 0 };
	if (m_ConvCSPInfo->SIMD != NONE) {
		if      (m_ConvCSPInfo->SIMD & AVX2)  _tcscat_s(buf, _countof(buf), "AVX2");
		else if (m_ConvCSPInfo->SIMD & AVX)   _tcscat_s(buf, _countof(buf), "AVX");
		else if (m_ConvCSPInfo->SIMD & SSE42) _tcscat_s(buf, _countof(buf), "SSE4.2");
		else if (m_ConvCSPInfo->SIMD & SSE41) _tcscat_s(buf, _countof(buf), "SSE4.1");
		else if (m_ConvCSPInfo->SIMD & SSSE3) _tcscat_s(buf, _countof(buf), "SSSE3");
		else if (m_ConvCSPInfo->SIMD & SSE3)  _tcscat_s(buf, _countof(buf), "SSE3");
		else if (m_ConvCSPInfo->SIMD & SSE2)  _tcscat_s(buf, _countof(buf), "SSE2");
	}

	CreateInputInfo(_T("auo"), _T("yuy2"), (m_interlaced) ? _T("nv12i") : _T("nv12p"), buf, inputPrm);

	return 0;
}
int AuoInput::LoadNextFrame(EncodeInputSurfaceInfo *surface) {
	if (FALSE != (pe->aud_parallel.abort = oip->func_is_abort()))
		return NVENC_THREAD_ABORT;

	while (m_pause) {
		Sleep(LOG_UPDATE_INTERVAL);
		if (oip->func_is_abort())
			return NVENC_THREAD_ABORT;
		log_process_events();
	}

	if (m_iFrame >= oip->n) {
		oip->func_rest_time_disp(m_iFrame, oip->n);
		release_audio_parallel_events(pe);
		return NVENC_THREAD_FINISHED;
	}

	void *frame = NULL;
	if (conf->vid.afs) {
		BOOL drop = FALSE;
		for (;;) {
			if ((frame = afs_get_video((OUTPUT_INFO *)oip, m_iFrame, &drop, &jitter[m_iFrame + 1])) == NULL) {
				error_afs_get_frame();
				return false;
			}
			if (!drop)
				break;
			jitter[m_iFrame] = DROP_FRAME_FLAG;
			pe->drop_count++;
			m_pStatus->m_sData.frameDrop++;
			m_iFrame++;
			if (m_iFrame >= oip->n) {
				oip->func_rest_time_disp(m_iFrame, oip->n);
				release_audio_parallel_events(pe);
				return false;
			}
		}
	} else {
		if ((frame = oip->func_get_video_ex(m_iFrame, COLORFORMATS[CF_YUY2].FOURCC)) == NULL) {
			error_afs_get_frame();
			return false;
		}
	}
	m_ConvCSPInfo->func[!!m_interlaced](surface->pExtAllocHost, frame, surface->dwWidth, surface->dwWidth * 2, surface->dwCuPitch, surface->dwHeight, surface->dwHeight, m_crop);

	m_iFrame++;
	if (!(m_iFrame & 7))
		aud_parallel_task(oip, pe);

	_InterlockedIncrement(&m_pStatus->m_sData.frameIn);

	uint32_t tm = timeGetTime();
	if (tm - m_tmLastUpdate > 800) {
		m_tmLastUpdate = tm;
		oip->func_rest_time_disp(m_iFrame, oip->n);
		oip->func_update_preview();
	}

	return NVENC_THREAD_RUNNING;
}

CAuoNvEnc::CAuoNvEnc() {

}

CAuoNvEnc::~CAuoNvEnc() {

}

int CAuoNvEnc::InitInput() {
	m_pStatus = new AuoEncodeStatus();
	m_pInput = new AuoInput();
	int ret = m_pInput->Init(&m_InputParam.input, m_pStatus);
	m_pStatus->m_nOutputFPSRate = m_InputParam.input.rate;
	m_pStatus->m_nOutputFPSScale = m_InputParam.input.scale;
	return ret;
}

#pragma warning (push)
#pragma warning (disable:4100)
int CAuoNvEnc::nvPrintf(FILE *fp, int log_level, const char *format, ...) {
	if (log_level < m_log_level)
		return 0;

	log_level = clamp(log_level, LOG_INFO, LOG_ERROR);

	va_list args;
	va_start(args, format);

	int len = _vscprintf(format, args);
	char *const buffer = (char*)malloc((len+1) * sizeof(buffer[0])); // _vscprintf doesn't count terminating '\0'

	vsprintf_s(buffer, len+1, format, args);

	static const char *const LOG_LEVEL_STR[] = { "info", "warning", "error" };
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
#pragma warning(pop)
