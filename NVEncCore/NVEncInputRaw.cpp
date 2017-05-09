// -----------------------------------------------------------------------------------------
// NVEnc by rigaya
// -----------------------------------------------------------------------------------------
//
// The MIT License
//
// Copyright (c) 2014-2016 rigaya
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

#include <io.h>
#include <fcntl.h>
#include <string>
#include <sstream>
#include "NVEncStatus.h"
#include "nvEncodeAPI.h"
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
                if (0 == _strnicmp(p+1, "420p9", strlen("420p9"))) {
                    inputPrm->csp = NV_ENC_CSP_YV12_09;
                } else if (0 == _strnicmp(p+1, "420p10", strlen("420p10"))) {
                    inputPrm->csp = NV_ENC_CSP_YV12_10;
                } else if (0 == _strnicmp(p+1, "420p12", strlen("420p12"))) {
                    inputPrm->csp = NV_ENC_CSP_YV12_12;
                }  else if (0 == _strnicmp(p+1, "420p14", strlen("420p14"))) {
                    inputPrm->csp = NV_ENC_CSP_YV12_14;
                }  else if (0 == _strnicmp(p+1, "420p16", strlen("420p16"))) {
                    inputPrm->csp = NV_ENC_CSP_YV12_16;
                } else if (0 == _strnicmp(p+1, "420mpeg2", strlen("420mpeg2"))
                        || 0 == _strnicmp(p+1, "420jpeg",  strlen("420jpeg"))
                        || 0 == _strnicmp(p+1, "420paldv", strlen("420paldv"))
                        || 0 == _strnicmp(p+1, "420",      strlen("420"))) {
                    inputPrm->csp = NV_ENC_CSP_YV12;
                } else if (0 == _strnicmp(p+1, "422", strlen("422"))) {
                    inputPrm->csp = NV_ENC_CSP_YUV422;
                } else if (0 == _strnicmp(p+1, "444p9", strlen("444p9"))) {
                    inputPrm->csp = NV_ENC_CSP_YUV444_09;
                } else if (0 == _strnicmp(p+1, "444p10", strlen("444p10"))) {
                    inputPrm->csp = NV_ENC_CSP_YUV444_10;
                } else if (0 == _strnicmp(p+1, "444p12", strlen("444p12"))) {
                    inputPrm->csp = NV_ENC_CSP_YUV444_12;
                } else if (0 == _strnicmp(p+1, "444p14", strlen("444p14"))) {
                    inputPrm->csp = NV_ENC_CSP_YUV444_14;
                } else if (0 == _strnicmp(p+1, "444p16", strlen("444p16"))) {
                    inputPrm->csp = NV_ENC_CSP_YUV444_16;
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

RGY_ERR NVEncInputRaw::Init(InputVideoInfo *inputPrm, shared_ptr<EncodeStatus> pStatus) {
    Close();

    m_pEncSatusInfo = pStatus;

    if (0 == _tcscmp(inputPrm->filename, _T("-"))) {
        if (_setmode( _fileno( stdin ), _O_BINARY ) == 1) {
            AddMessage(RGY_LOG_ERROR, _T("failed to switch stdin to binary mode."));
            return RGY_ERR_UNKNOWN;
        }
        m_fp = stdin;
    } else {
        if (_tfopen_s(&m_fp, inputPrm->filename, _T("rb")) || NULL == m_fp) {
            AddMessage(RGY_LOG_ERROR, _T("Failed to open input file.\n"));
            return RGY_ERR_FILE_OPEN;
        }
    }
    
    NV_ENC_CSP inputCsp = NV_ENC_CSP_YV12;
    m_bIsY4m = inputPrm->type == RGY_INPUT_FMT_Y4M;
    if (m_bIsY4m) {
        m_strReaderName = _T("y4m");
        char buf[128] = { 0 };
        InputVideoInfo videoInfo;
        memset(&videoInfo, 0, sizeof(videoInfo));

        if (fread(buf, 1, strlen("YUV4MPEG2"), m_fp) != strlen("YUV4MPEG2")
            || strcmp(buf, "YUV4MPEG2") != 0
            || !fgets(buf, sizeof(buf), m_fp)
            || ParseY4MHeader(buf, &videoInfo)) {
            AddMessage(RGY_LOG_ERROR, _T("failed to parse y4m header."));
            return RGY_ERR_INVALID_FORMAT;
        }
        inputPrm->width = videoInfo.width;
        inputPrm->height = videoInfo.height;
        inputPrm->scale = videoInfo.scale;
        inputPrm->rate = videoInfo.rate;
        memcpy(inputPrm->sar, videoInfo.sar, sizeof(videoInfo.sar));
        inputCsp = videoInfo.csp;
    }
    uint32_t bufferSize = 0;
    uint32_t src_pitch = 0;
    switch (inputCsp) {
    case NV_ENC_CSP_NV12:
    case NV_ENC_CSP_YV12:
        src_pitch = inputPrm->width;
        bufferSize = inputPrm->width * inputPrm->height * 3 / 2; break;
    case NV_ENC_CSP_YV12_09:
    case NV_ENC_CSP_YV12_10:
    case NV_ENC_CSP_YV12_12:
    case NV_ENC_CSP_YV12_14:
    case NV_ENC_CSP_YV12_16:
        src_pitch = inputPrm->width * 2;
        bufferSize = inputPrm->width * inputPrm->height * 3; break;
    case NV_ENC_CSP_YUV422:
        src_pitch = inputPrm->width;
        bufferSize = inputPrm->width * inputPrm->height * 2; break;
    case NV_ENC_CSP_YUV444:
        src_pitch = inputPrm->width;
        bufferSize = inputPrm->width * inputPrm->height * 3; break;
    case NV_ENC_CSP_YUV444_09:
    case NV_ENC_CSP_YUV444_10:
    case NV_ENC_CSP_YUV444_12:
    case NV_ENC_CSP_YUV444_14:
    case NV_ENC_CSP_YUV444_16:
        src_pitch = inputPrm->width * 2;
        bufferSize = inputPrm->width * inputPrm->height * 6; break;
    default:
        AddMessage(RGY_LOG_ERROR, _T("Unknown color foramt.\n"));
        return RGY_ERR_INVALID_COLOR_FORMAT;
    }
    if (NULL == (m_inputBuffer = (uint8_t *)_aligned_malloc(bufferSize, 32))) {
        AddMessage(RGY_LOG_ERROR, _T("raw: Failed to allocate input buffer.\n"));
        return RGY_ERR_NULL_PTR;
    }

    m_sConvert = get_convert_csp_func(inputCsp, inputPrm->csp, false);

    if (nullptr == m_sConvert) {
        AddMessage(RGY_LOG_ERROR, _T("raw/y4m: invalid colorformat.\n"));
        return RGY_ERR_INVALID_COLOR_FORMAT;
    }
    
    memcpy(&m_sDecParam, inputPrm, sizeof(m_sDecParam));
    m_sDecParam.src_pitch = src_pitch;
    CreateInputInfo(m_strReaderName.c_str(), NV_ENC_CSP_NAMES[m_sConvert->csp_from], NV_ENC_CSP_NAMES[m_sConvert->csp_to], get_simd_str(m_sConvert->simd), inputPrm);
    AddMessage(RGY_LOG_DEBUG, m_strInputInfo);
    return RGY_ERR_NONE;
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

RGY_ERR NVEncInputRaw::LoadNextFrame(void *dst, int dst_pitch) {

    if (m_bIsY4m) {
        BYTE y4m_buf[8] = { 0 };
        if (fread(y4m_buf, 1, strlen("FRAME"), m_fp) != strlen("FRAME"))
            return RGY_ERR_MORE_DATA;
        if (memcmp(y4m_buf, "FRAME", strlen("FRAME")) != NULL)
            return RGY_ERR_MORE_DATA;
        int i;
        for (i = 0; fgetc(m_fp) != '\n'; i++)
        if (i >= 64)
            return RGY_ERR_MORE_DATA;
    }

    uint32_t frameSize = 0;
    switch (m_sConvert->csp_from) {
    case NV_ENC_CSP_NV12:
    case NV_ENC_CSP_YV12:
        frameSize = m_sDecParam.width * m_sDecParam.height * 3 / 2; break;
    case NV_ENC_CSP_YUV422:
        frameSize = m_sDecParam.width * m_sDecParam.height * 2; break;
    case NV_ENC_CSP_YUV444:
        frameSize = m_sDecParam.width * m_sDecParam.height * 3; break;
    case NV_ENC_CSP_YV12_09:
    case NV_ENC_CSP_YV12_10:
    case NV_ENC_CSP_YV12_12:
    case NV_ENC_CSP_YV12_14:
    case NV_ENC_CSP_YV12_16:
        frameSize = m_sDecParam.width * m_sDecParam.height * 3; break;
    case NV_ENC_CSP_YUV444_09:
    case NV_ENC_CSP_YUV444_10:
    case NV_ENC_CSP_YUV444_12:
    case NV_ENC_CSP_YUV444_14:
    case NV_ENC_CSP_YUV444_16:
        frameSize = m_sDecParam.width * m_sDecParam.height * 6; break;
    default:
        AddMessage(RGY_LOG_ERROR, _T("Unknown color foramt.\n"));
        return RGY_ERR_INVALID_COLOR_FORMAT;
    }
    if (frameSize != fread(m_inputBuffer, 1, frameSize, m_fp)) {
        return RGY_ERR_MORE_DATA;
    }
    void *dst_array[3];
    dst_array[0] = dst;
    dst_array[1] = (uint8_t *)dst_array[0] + dst_pitch * (m_sDecParam.height - m_sDecParam.crop.c[1] - m_sDecParam.crop.c[3]);
    dst_array[2] = (uint8_t *)dst_array[1] + dst_pitch * (m_sDecParam.height - m_sDecParam.crop.c[1] - m_sDecParam.crop.c[3]); //YUV444出力時

    const void *src_array[3];
    src_array[0] = m_inputBuffer;
    src_array[1] = (uint8_t *)src_array[0] + m_sDecParam.src_pitch * m_sDecParam.height;
    switch (m_sConvert->csp_from) {
    case NV_ENC_CSP_YV12:
    case NV_ENC_CSP_YV12_09:
    case NV_ENC_CSP_YV12_10:
    case NV_ENC_CSP_YV12_12:
    case NV_ENC_CSP_YV12_14:
    case NV_ENC_CSP_YV12_16:
        src_array[2] = (uint8_t *)src_array[1] + m_sDecParam.src_pitch * m_sDecParam.height / 4;
        break;
    case NV_ENC_CSP_YUV422:
        src_array[2] = (uint8_t *)src_array[1] + m_sDecParam.src_pitch * m_sDecParam.height / 2;
        break;
    case NV_ENC_CSP_YUV444:
    case NV_ENC_CSP_YUV444_09:
    case NV_ENC_CSP_YUV444_10:
    case NV_ENC_CSP_YUV444_12:
    case NV_ENC_CSP_YUV444_14:
    case NV_ENC_CSP_YUV444_16:
        src_array[2] = (uint8_t *)src_array[1] + m_sDecParam.src_pitch * m_sDecParam.height;
        break;
    case NV_ENC_CSP_NV12:
    default:
        break;
    }

    const int src_uv_pitch = (m_sConvert->csp_from == NV_ENC_CSP_YUV444) ? m_sDecParam.src_pitch : m_sDecParam.src_pitch / 2;
    m_sConvert->func[0](dst_array, src_array, m_sDecParam.width, m_sDecParam.src_pitch, src_uv_pitch, dst_pitch, m_sDecParam.height, m_sDecParam.height, m_sDecParam.crop.c);

    m_pEncSatusInfo->m_sData.frameIn++;
    return m_pEncSatusInfo->UpdateDisplay();
}

#endif
