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


NVEncBasicInput::NVEncBasicInput() {
    memset(&m_sTrimParam, 0, sizeof(m_sTrimParam));
}

NVEncBasicInput::~NVEncBasicInput() {
    Close();
}

#pragma warning(push)
#pragma warning(disable:4100)
int NVEncBasicInput::Init(InputVideoInfo *inputPrm, shared_ptr<EncodeStatus> pStatus) {

    return 0;
}

int NVEncBasicInput::LoadNextFrame(void *dst, int dst_pitch) {
    return 0;
}
#pragma warning(pop)

void NVEncBasicInput::CreateInputInfo(const TCHAR *inputTypeName, const TCHAR *inputCSpName, const TCHAR *outputCSpName, const TCHAR *convSIMD, const InputVideoInfo *inputPrm) {
    std::basic_stringstream<TCHAR> ss;

    ss << inputTypeName;
    ss << _T("(") << inputCSpName << _T(")");
    ss << _T("->") << outputCSpName;
    if (convSIMD && _tcslen(convSIMD)) {
        ss << _T(" [") << convSIMD << _T("]");
    }
    ss << _T(", ");
    ss << inputPrm->width << _T("x") << inputPrm->height << _T(", ");
    ss << inputPrm->rate << _T("/") << inputPrm->scale << _T(" fps");

    m_strInputInfo = ss.str();
}

void NVEncBasicInput::Close() {
    m_pEncSatusInfo.reset();
}
