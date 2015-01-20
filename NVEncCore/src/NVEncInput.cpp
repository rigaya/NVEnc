//  -----------------------------------------------------------------------------------------
//    NVEnc by rigaya
//  -----------------------------------------------------------------------------------------
//   ソースコードについて
//   ・無保証です。
//   ・本ソースコードを使用したことによるいかなる損害・トラブルについてrigayaは責任を負いません。
//   以上に了解して頂ける場合、本ソースコードの使用、複製、改変、再頒布を行って頂いて構いません。
//  -----------------------------------------------------------------------------------------

#include <io.h>
#include <fcntl.h>
#include <string>
#include <sstream>
#include "nvEncodeAPI.h"
#include "NVEncStatus.h"
#include "NVEncInput.h"
#include "ConvertCSP.h"


int NVEncBasicInput::ParseY4MHeader(char *buf, InputVideoInfo *inputPrm) {
	char *p, *q = NULL;
	for (p = buf; (p = strtok_s(p, " ", &q)) != NULL; ) {
		switch (*p) {
			case 'W':
				{
					char *eptr = NULL;
					int w = strtol(p+1, &eptr, 10);
					if (*eptr == '\0' && w)
						inputPrm->width = w;
				}
				break;
			case 'H':
				{
					char *eptr = NULL;
					int h = strtol(p+1, &eptr, 10);
					if (*eptr == '\0' && h)
						inputPrm->height = h;
				}
				break;
			case 'F':
				{
					int rate = 0, scale = 0;
					if (   (inputPrm->scale == 0 || inputPrm->rate == 0)
						&& sscanf_s(p+1, "%d:%d", &rate, &scale) == 2) {
							if (rate && scale) {
								inputPrm->rate = rate;
								inputPrm->scale = scale;
							}
					}
				}
				break;
			case 'A':
				{
					//int sar_x = 0, sar_y = 0;
					//if (   (info->AspectRatioW == 0 || info->AspectRatioH == 0)
					//	&& sscanf_s(p+1, "%d:%d", &sar_x, &sar_y) == 2) {
					//		if (sar_x && sar_y) {
					//			info->AspectRatioW = (mfxU16)sar_x;
					//			info->AspectRatioH = (mfxU16)sar_y;
					//		}
					//}
				}
				break;
			//case 'I':
			//	switch (*(p+1)) {
			//case 'b':
			//	info->PicStruct = MFX_PICSTRUCT_FIELD_BFF;
			//	break;
			//case 't':
			//case 'm':
			//	info->PicStruct = MFX_PICSTRUCT_FIELD_TFF;
			//	break;
			//default:
			//	break;
			//	}
			//	break;
			case 'C':
				if (   0 != _strnicmp(p+1, "420",      strlen("420"))
					&& 0 != _strnicmp(p+1, "420mpeg2", strlen("420mpeg2"))
					&& 0 != _strnicmp(p+1, "420jpeg",  strlen("420jpeg"))
					&& 0 != _strnicmp(p+1, "420paldv", strlen("420paldv"))) {
					return 1;
				}
				break;
			default:
				break;
		}
		p = NULL;
	}
	return 0;
}

NVEncBasicInput::NVEncBasicInput() {

}

NVEncBasicInput::~NVEncBasicInput() {
	Close();
}

void NVEncBasicInput::setSurfaceInfo(InputVideoInfo *inputPrm) {
	m_stSurface.width     = inputPrm->width;
	m_stSurface.src_pitch = inputPrm->width;
	m_stSurface.height    = inputPrm->height;
	memcpy(&m_stSurface.crop, &inputPrm->crop, sizeof(m_stSurface.crop));
}

int NVEncBasicInput::Init(InputVideoInfo *inputPrm, EncodeStatus *pStatus) {
	Close();

	m_pStatus = pStatus;
	BasicInputInfo *info = reinterpret_cast<BasicInputInfo *>(inputPrm->otherPrm);

	if (0 == inputPrm->filename.compare(_T("-"))) {
		if (_setmode( _fileno( stdin ), _O_BINARY ) == 1) {
			m_inputMes = ((m_bIsY4m) ? _T("y4m") : _T("raw"));
			m_inputMes +=_T(": failed to switch stdin to binary mode.");
			return 1;
		}
		m_fp = stdin;
	} else {
		if (_tfopen_s(&m_fp, inputPrm->filename.c_str(), _T("rb")) || NULL == m_fp) {
			m_inputMes = _T("raw: 入力ファイルのオープンに失敗しました。\n");
			return 1;
		}
	}

	if (info->isY4m) {
		char buf[128] = { 0 };
		if (fread(buf, 1, strlen("YUV4MPEG2"), m_fp) != strlen("YUV4MPEG2")
			|| strcmp(buf, "YUV4MPEG2") != 0
			|| !fgets(buf, sizeof(buf), m_fp)
			|| ParseY4MHeader(buf, inputPrm)) {
			return 1;
		}
	}
	if (NULL == (m_inputBuffer = (uint8_t *)_aligned_malloc(inputPrm->width * inputPrm->height * 3 / 2, 32))) {
		m_inputMes = _T("raw: 入力用バッファの確保に失敗しました。\n");
		return 1;
	}
	m_bIsY4m = info->isY4m;
	
	setSurfaceInfo(inputPrm);
	m_stSurface.src_pitch = inputPrm->width;
	CreateInputInfo((m_bIsY4m) ? _T("y4m") : _T("raw"), _T("yv12"), _T("nv12"), NULL, inputPrm);

	return 0;
}

void NVEncBasicInput::CreateInputInfo(const TCHAR *inputTypeName, const TCHAR *inputCSpName, const TCHAR *outputCSpName, const TCHAR *convSIMD, const InputVideoInfo *inputPrm) {
	std::basic_stringstream<TCHAR> ss;

	ss << inputTypeName << _T(" ");
	ss << _T("(") << inputCSpName << _T(")");
	ss << _T(" -> ") << outputCSpName;
	if (convSIMD && _tcslen(convSIMD)) {
		ss << _T(" [") << convSIMD << _T("]");
	}
	ss << _T(", ");
	ss << inputPrm->width << _T("x") << inputPrm->height << _T(", ");
	ss << inputPrm->rate << _T("/") << inputPrm->scale << _T(" fps");

	m_inputMes = ss.str();
}

void NVEncBasicInput::Close() {
	if (m_fp) {
		fclose(m_fp);
		m_fp = NULL;
	}
	if (m_inputBuffer) {
		_aligned_free(m_inputBuffer);
		m_inputBuffer = NULL;
	}
	m_bIsY4m = false;
}

int NVEncBasicInput::LoadNextFrame(void *dst, int dst_pitch) {

	if (m_bIsY4m) {
		BYTE y4m_buf[8] = { 0 };
		if (fread(y4m_buf, 1, strlen("FRAME"), m_fp) != strlen("FRAME"))
			return 1;
		if (memcmp(y4m_buf, "FRAME", strlen("FRAME")) != NULL)
			return 1;
		int i;
		for (i = 0; fgetc(m_fp) != '\n'; i++)
		if (i >= 64)
			return 1;
	}

	size_t frameSize = m_stSurface.width * m_stSurface.height * 3 / 2;
	if (frameSize != fread(m_inputBuffer, 1, frameSize, m_fp)) {
		return -1;
	}
	convert_yv12_to_nv12(dst, m_inputBuffer, m_stSurface.width, m_stSurface.width, dst_pitch, m_stSurface.height, m_stSurface.height, m_stSurface.crop);

	m_pStatus->m_sData.frameIn++;

	return 0;
}
