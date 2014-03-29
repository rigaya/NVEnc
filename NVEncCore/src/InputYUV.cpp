//  -----------------------------------------------------------------------------------------
//    NVEnc by rigaya
//  -----------------------------------------------------------------------------------------
//   ソースコードについて
//   ・無保証です。
//   ・本ソースコードを使用したことによるいかなる損害・トラブルについてrigayaは責任を負いません。
//   以上に了解して頂ける場合、本ソースコードの使用、複製、改変、再頒布を行って頂いて構いません。
//  -----------------------------------------------------------------------------------------

#include <sstream>
#include "CNVEncoder.h"
#include "NVEncStatus.h"
#include "InputYUV.h"
#include "ConvertCSP.h"

BasicInput::BasicInput() {

}

BasicInput::~BasicInput() {
	Close();
}

int BasicInput::Init(InputVideoInfo *inputPrm, EncodeStatus *pStatus) {
	Close();

	m_pStatus = pStatus;
	BasicInputInfo *info = reinterpret_cast<BasicInputInfo *>(inputPrm->otherPrm);
	memcpy(m_crop, inputPrm->crop, sizeof(m_crop));

	if (_tfopen_s(&m_fp, info->filename, _T("rb")) || NULL == m_fp) {
		m_inputMes = _T("raw: 入力ファイルのオープンに失敗しました。\n");
		return 1;
	}
	if (NULL == (m_inputBuffer = (uint8_t *)_aligned_malloc(inputPrm->width * inputPrm->height * 3 / 2, 32))) {
		m_inputMes = _T("raw: 入力用バッファの確保に失敗しました。\n");
		return 1;
	}
	m_bIsY4m = info->isY4m;

	CreateInputInfo((m_bIsY4m) ? _T("y4m") : _T("raw"), _T("yv12"), _T("nv12"), NULL, inputPrm);

	return 0;
}

void BasicInput::CreateInputInfo(const TCHAR *inputTypeName, const TCHAR *inputCSpName, const TCHAR *outputCSpName, const TCHAR *convSIMD, const InputVideoInfo *inputPrm) {
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
void BasicInput::Close() {
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

int BasicInput::LoadNextFrame(EncodeInputSurfaceInfo *surface) {
	size_t frameSize = surface->dwWidth * surface->dwHeight * 3 / 2;
	if (frameSize != fread(m_inputBuffer, 1, frameSize, m_fp)) {
		return -1;
	}
	convert_yv12_to_nv12(surface->pExtAllocHost, m_inputBuffer, surface->dwWidth, surface->dwWidth, surface->dwCuPitch, surface->dwHeight, surface->dwHeight, m_crop);

	m_pStatus->m_sData.frameIn++;

	return 0;
}
