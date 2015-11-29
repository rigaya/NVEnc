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
#include "NVEncInputRaw.h"

#if RAW_READER

int NVEncInputRaw::ParseY4MHeader(char *buf, InputVideoInfo *inputPrm) {
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
                    int sar_x = 0, sar_y = 0;
                    if ((inputPrm->sar[0] == 0 || inputPrm->sar[1] == 0)
                        && sscanf_s(p+1, "%d:%d", &sar_x, &sar_y) == 2) {
                        if (sar_x && sar_y) {
                            inputPrm->sar[0] = sar_x;
                            inputPrm->sar[1] = sar_y;
                        }
                    }
                }
                break;
            //case 'I':
            //    switch (*(p+1)) {
            //case 'b':
            //    info->PicStruct = MFX_PICSTRUCT_FIELD_BFF;
            //    break;
            //case 't':
            //case 'm':
            //    info->PicStruct = MFX_PICSTRUCT_FIELD_TFF;
            //    break;
            //default:
            //    break;
            //    }
            //    break;
            case 'C':
                if (   0 == _strnicmp(p+1, "420",      strlen("420"))
                    || 0 == _strnicmp(p+1, "420mpeg2", strlen("420mpeg2"))
                    || 0 == _strnicmp(p+1, "420jpeg",  strlen("420jpeg"))
                    || 0 == _strnicmp(p+1, "420paldv", strlen("420paldv"))) {
                    inputPrm->csp = NV_ENC_CSP_YV12;
                } else if (0 == _strnicmp(p+1, "422", strlen("422"))) {
                    inputPrm->csp = NV_ENC_CSP_YUV422;
                } else if (0 == _strnicmp(p+1, "444", strlen("444"))) {
                    inputPrm->csp = NV_ENC_CSP_YUV444;
                } else {
                    return 1;
                }
                break;
            default:
                break;
        }
        p = NULL;
    }
    if (inputPrm->rate > 0 && inputPrm->scale > 0) {
        int fps_gcd = nv_get_gcd(inputPrm->rate, inputPrm->scale);
        inputPrm->rate  /= fps_gcd;
        inputPrm->scale /= fps_gcd;
    }

    return 0;
}

NVEncInputRaw::NVEncInputRaw() {
    m_strReaderName = _T("raw");
}

NVEncInputRaw::~NVEncInputRaw() {
    Close();
}

int NVEncInputRaw::Init(InputVideoInfo *inputPrm, shared_ptr<EncodeStatus> pStatus) {
    Close();

    m_pEncSatusInfo = pStatus;

    if (0 == _tcscmp(inputPrm->filename, _T("-"))) {
        if (_setmode( _fileno( stdin ), _O_BINARY ) == 1) {
            AddMessage(NV_LOG_ERROR, _T("failed to switch stdin to binary mode."));
            return 1;
        }
        m_fp = stdin;
    } else {
        if (_tfopen_s(&m_fp, inputPrm->filename, _T("rb")) || NULL == m_fp) {
            AddMessage(NV_LOG_ERROR, _T("Failed to open input file.\n"));
            return 1;
        }
    }
    
    NV_ENC_CSP inputCsp = NV_ENC_CSP_YV12;
    m_bIsY4m = inputPrm->type == NV_ENC_INPUT_Y4M;
    if (m_bIsY4m) {
        m_strReaderName = _T("y4m");
        char buf[128] = { 0 };
        InputVideoInfo videoInfo = { 0 };
        if (fread(buf, 1, strlen("YUV4MPEG2"), m_fp) != strlen("YUV4MPEG2")
            || strcmp(buf, "YUV4MPEG2") != 0
            || !fgets(buf, sizeof(buf), m_fp)
            || ParseY4MHeader(buf, &videoInfo)) {
            AddMessage(NV_LOG_ERROR, _T("failed to parse y4m header."));
            return 1;
        }
        inputPrm->width = videoInfo.width;
        inputPrm->height = videoInfo.height;
        inputPrm->scale = videoInfo.scale;
        inputPrm->rate = videoInfo.rate;
        memcpy(inputPrm->sar, videoInfo.sar, sizeof(videoInfo.sar));
        inputCsp = videoInfo.csp;
    }
    uint32_t bufferSize = 0;
    switch (inputCsp) {
    case NV_ENC_CSP_NV12:
    case NV_ENC_CSP_YV12:
        bufferSize = inputPrm->width * inputPrm->height * 3 / 2; break;
    case NV_ENC_CSP_YUV422:
        bufferSize = inputPrm->width * inputPrm->height * 2; break;
    case NV_ENC_CSP_YUV444:
        bufferSize = inputPrm->width * inputPrm->height * 3; break;
    default:
        return 1;
    }
    if (NULL == (m_inputBuffer = (uint8_t *)_aligned_malloc(bufferSize, 32))) {
        AddMessage(NV_LOG_ERROR, _T("raw: Failed to allocate input buffer.\n"));
        return 1;
    }

    m_pConvCSPInfo = get_convert_csp_func(inputCsp, inputPrm->csp, false);

    if (nullptr == m_pConvCSPInfo) {
        AddMessage(NV_LOG_ERROR, _T("raw/y4m: invalid colorformat.\n"));
        return 1;
    }
    
    memcpy(&m_sDecParam, inputPrm, sizeof(m_sDecParam));
    m_sDecParam.src_pitch = inputPrm->width;
    CreateInputInfo(m_strReaderName.c_str(), NV_ENC_CSP_NAMES[m_pConvCSPInfo->csp_from], NV_ENC_CSP_NAMES[m_pConvCSPInfo->csp_to], get_simd_str(m_pConvCSPInfo->simd), inputPrm);
    AddMessage(NV_LOG_DEBUG, m_strInputInfo);
    return 0;
}

void NVEncInputRaw::Close() {
    if (m_fp) {
        fclose(m_fp);
        m_fp = NULL;
    }
    if (m_inputBuffer) {
        _aligned_free(m_inputBuffer);
        m_inputBuffer = NULL;
    }
    m_bIsY4m = false;
    m_pEncSatusInfo.reset();
}

int NVEncInputRaw::LoadNextFrame(void *dst, int dst_pitch) {

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

    uint32_t frameSize = 0;
    switch (m_pConvCSPInfo->csp_from) {
    case NV_ENC_CSP_NV12:
    case NV_ENC_CSP_YV12:
        frameSize = m_sDecParam.width * m_sDecParam.height * 3 / 2; break;
    case NV_ENC_CSP_YUV422:
        frameSize = m_sDecParam.width * m_sDecParam.height * 2; break;
    case NV_ENC_CSP_YUV444:
        frameSize = m_sDecParam.width * m_sDecParam.height * 3; break;
    default:
        return 1;
    }
    if (frameSize != fread(m_inputBuffer, 1, frameSize, m_fp)) {
        return -1;
    }
    void *dst_array[3];
    dst_array[0] = dst;
    dst_array[1] = (uint8_t *)dst_array[0] + dst_pitch * (m_sDecParam.height - m_sDecParam.crop.c[1] - m_sDecParam.crop.c[3]);
    dst_array[2] = (uint8_t *)dst_array[1] + dst_pitch * (m_sDecParam.height - m_sDecParam.crop.c[1] - m_sDecParam.crop.c[3]); //YUV444出力時

    const void *src_array[3];
    src_array[0] = m_inputBuffer;
    src_array[1] = (uint8_t *)src_array[0] + m_sDecParam.src_pitch * m_sDecParam.height;
    switch (m_pConvCSPInfo->csp_from) {
    case NV_ENC_CSP_YV12:
        src_array[2] = (uint8_t *)src_array[1] + m_sDecParam.src_pitch * m_sDecParam.height / 4;
        break;
    case NV_ENC_CSP_YUV422:
        src_array[2] = (uint8_t *)src_array[1] + m_sDecParam.src_pitch * m_sDecParam.height / 2;
        break;
    case NV_ENC_CSP_YUV444:
        src_array[2] = (uint8_t *)src_array[1] + m_sDecParam.src_pitch * m_sDecParam.height;
        break;
    case NV_ENC_CSP_NV12:
    default:
        break;
    }

    const int src_uv_pitch = (m_pConvCSPInfo->csp_from == NV_ENC_CSP_YUV444) ? m_sDecParam.src_pitch : m_sDecParam.src_pitch / 2;
    m_pConvCSPInfo->func[0](dst_array, src_array, m_sDecParam.width, m_sDecParam.src_pitch, src_uv_pitch, dst_pitch, m_sDecParam.height, m_sDecParam.height, m_sDecParam.crop.c);

    m_pEncSatusInfo->m_sData.frameIn++;

    auto tm = std::chrono::system_clock::now();
    if (duration_cast<std::chrono::milliseconds>(tm - m_tmLastUpdate).count() > UPDATE_INTERVAL) {
        m_tmLastUpdate = tm;
        m_pEncSatusInfo->UpdateDisplay(tm);
    }

    return 0;
}

#endif
